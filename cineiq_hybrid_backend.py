"""Backend utilities for CINEIQ's SVD + hybrid movie recommender.

This module is designed to plug into the notebook-first workflow that already
exists in this repository. It assumes you already have:

1. A `movies` DataFrame with movie metadata.
2. A `ratings_small` DataFrame with columns like `userId`, `movieId`, `rating`.
3. A content-based function: `recommend_movies(title, top_n)`.
4. A collaborative function: `recommend_collaborative(title, top_n)`.

Typical notebook usage:

```python
from cineiq_hybrid_backend import (
    configure_hybrid_engine,
    hybrid_recommendation,
    load_svd_model,
    train_svd_model,
)

# One-time training step
train_svd_model(ratings_small)

# Or, on later runs:
# load_svd_model()

configure_hybrid_engine(
    movies_df=movies,
    svd_model_instance=load_svd_model(),
    content_recommender=recommend_movies,
    collaborative_recommender=recommend_collaborative,
)

hybrid_recommendation(userId=1, movie_title="toy story", top_n=10)
```
"""

from __future__ import annotations

import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd


# Keep the artifact path inside the project's existing model_artifacts folder.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SVD_MODEL_PATH = PROJECT_ROOT / "model_artifacts" / "svd_model.pkl"

# Module-level dependencies are intentionally configurable so this file can
# attach to the objects already created inside the user's notebook.
MOVIES_DF: Optional[pd.DataFrame] = None
CONTENT_RECOMMENDER: Optional[Callable[[str, int], pd.DataFrame | str]] = None
COLLAB_RECOMMENDER: Optional[Callable[[str, int], pd.DataFrame | str]] = None
SVD_MODEL: Any = None
SVD_RATING_SCALE: Optional[tuple[float, float]] = None


def _import_surprise() -> tuple[Any, Any, Any]:
    """Import Surprise lazily so the module can still be imported without it.

    Surprise is only required for training/loading the SVD model. The hybrid
    score-merging logic can still be unit-tested with a fake model object.
    """

    try:
        from surprise import Dataset, Reader, SVD
    except ImportError as exc:
        raise ImportError(
            "scikit-surprise is required for the SVD stage. Install it in a "
            "Python environment supported by Surprise, then retry."
        ) from exc

    return Dataset, Reader, SVD


def configure_hybrid_engine(
    movies_df: pd.DataFrame,
    svd_model_instance: Any,
    content_recommender: Callable[[str, int], pd.DataFrame | str],
    collaborative_recommender: Callable[[str, int], pd.DataFrame | str],
    svd_rating_scale: Optional[tuple[float, float]] = None,
) -> None:
    """Register the notebook objects that `hybrid_recommendation` depends on.

    This setup function lets the final recommendation API keep the exact
    signature requested by the user:

        hybrid_recommendation(userId, movie_title, top_n=10, weights=(...))
    """

    if not isinstance(movies_df, pd.DataFrame):
        raise TypeError("movies_df must be a pandas DataFrame.")

    if "movieId" not in movies_df.columns:
        raise ValueError("movies_df must contain a 'movieId' column.")

    if not callable(content_recommender):
        raise TypeError("content_recommender must be callable.")

    if not callable(collaborative_recommender):
        raise TypeError("collaborative_recommender must be callable.")

    if svd_model_instance is None:
        raise ValueError("svd_model_instance cannot be None.")

    global MOVIES_DF, CONTENT_RECOMMENDER, COLLAB_RECOMMENDER, SVD_MODEL, SVD_RATING_SCALE
    MOVIES_DF = movies_df.copy()
    CONTENT_RECOMMENDER = content_recommender
    COLLAB_RECOMMENDER = collaborative_recommender
    SVD_MODEL = svd_model_instance
    SVD_RATING_SCALE = svd_rating_scale or _extract_rating_scale_from_model(svd_model_instance)


def train_svd_model(
    ratings_small: pd.DataFrame,
    model_path: str | Path = DEFAULT_SVD_MODEL_PATH,
    *,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    svd_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """Train Surprise SVD on the full ratings DataFrame and persist it with pickle.

    Parameters
    ----------
    ratings_small:
        Filtered ratings DataFrame. At minimum it must contain user, item, and
        rating columns.
    model_path:
        File path where the trained model artifact should be saved.
    user_col / item_col / rating_col:
        Column names used to build the Surprise Dataset.
    svd_kwargs:
        Optional keyword arguments forwarded directly into `surprise.SVD(...)`.

    Returns
    -------
    The trained Surprise SVD model instance.
    """

    Dataset, Reader, SVD = _import_surprise()

    prepared_ratings = _prepare_ratings_frame(
        ratings_small=ratings_small,
        user_col=user_col,
        item_col=item_col,
        rating_col=rating_col,
    )

    rating_scale = _infer_rating_scale(prepared_ratings[rating_col])
    reader = Reader(rating_scale=rating_scale)
    surprise_dataset = Dataset.load_from_df(
        prepared_ratings[[user_col, item_col, rating_col]],
        reader,
    )

    # Full-trainset is appropriate here because the goal is to produce the
    # final inference model that the application will use later.
    trainset = surprise_dataset.build_full_trainset()

    model_config: dict[str, Any] = {
        "n_factors": 100,
        "n_epochs": 30,
        "biased": True,
        "random_state": 42,
        "verbose": False,
    }
    if svd_kwargs:
        model_config.update(svd_kwargs)

    svd_model = SVD(**model_config)
    svd_model.fit(trainset)

    artifact = {
        "model": svd_model,
        "rating_scale": rating_scale,
        "columns": {
            "user_col": user_col,
            "item_col": item_col,
            "rating_col": rating_col,
        },
        "svd_kwargs": model_config,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_ratings": int(len(prepared_ratings)),
        "n_users": int(prepared_ratings[user_col].nunique()),
        "n_items": int(prepared_ratings[item_col].nunique()),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as file_handle:
        pickle.dump(artifact, file_handle)

    global SVD_MODEL, SVD_RATING_SCALE
    SVD_MODEL = svd_model
    SVD_RATING_SCALE = rating_scale

    return svd_model


def load_svd_model(model_path: str | Path = DEFAULT_SVD_MODEL_PATH) -> Any:
    """Load a previously pickled SVD artifact and register it globally."""

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"SVD model artifact not found: {model_path}")

    with model_path.open("rb") as file_handle:
        payload = pickle.load(file_handle)

    if isinstance(payload, dict) and "model" in payload:
        svd_model = payload["model"]
        rating_scale = payload.get("rating_scale")
    else:
        # Backward-compatible fallback in case only the model object was pickled.
        svd_model = payload
        rating_scale = None

    global SVD_MODEL, SVD_RATING_SCALE
    SVD_MODEL = svd_model
    SVD_RATING_SCALE = rating_scale or _extract_rating_scale_from_model(svd_model)

    return svd_model


def hybrid_recommendation(
    userId: Any,
    movie_title: str,
    top_n: int = 10,
    weights: Sequence[float] = (0.3, 0.3, 0.4),
) -> pd.DataFrame:
    """Return the top-N hybrid recommendations for a given user and seed title.

    The function blends:
    1. Content-based TF-IDF similarity.
    2. Item-item Pearson collaborative similarity.
    3. Surprise SVD predicted user rating.

    Notes on cold-start behavior
    ----------------------------
    If `userId` was not seen during SVD training, Surprise can still produce a
    prediction. In that case it falls back to the global mean and any available
    learned item bias, which is a reasonable production default for a new user.
    """

    _ensure_hybrid_dependencies_are_configured()

    if not isinstance(movie_title, str) or not movie_title.strip():
        raise ValueError("movie_title must be a non-empty string.")

    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    tfidf_weight, pearson_weight, svd_weight = _validate_and_normalize_weights(weights)

    try:
        tfidf_result = CONTENT_RECOMMENDER(movie_title, top_n=50)  # type: ignore[misc]
    except Exception as exc:
        raise RuntimeError(
            f"Content-based recommender failed for title '{movie_title}'."
        ) from exc

    try:
        pearson_result = COLLAB_RECOMMENDER(movie_title, top_n=50)  # type: ignore[misc]
    except Exception as exc:
        raise RuntimeError(
            f"Collaborative recommender failed for title '{movie_title}'."
        ) from exc

    tfidf_candidates = _coerce_recommender_output(tfidf_result, source_name="TF-IDF")
    pearson_candidates = _coerce_recommender_output(pearson_result, source_name="Pearson")

    candidate_ids = pd.Index(tfidf_candidates["movieId"]).union(
        pd.Index(pearson_candidates["movieId"])
    )

    if candidate_ids.empty:
        return _empty_hybrid_frame()

    candidates = pd.DataFrame({"movieId": candidate_ids.astype(int)})
    candidates = _remove_seed_movie_from_candidates(candidates, movie_title)

    if candidates.empty:
        return _empty_hybrid_frame()

    candidates = candidates.merge(
        tfidf_candidates.rename(columns={"raw_score": "tfidf_raw_score"}),
        on="movieId",
        how="left",
    )
    candidates = candidates.merge(
        pearson_candidates.rename(columns={"raw_score": "pearson_raw_score"}),
        on="movieId",
        how="left",
    )

    # Missing scores are treated as zero contribution from that signal, exactly
    # as requested for movies absent from one of the recommenders.
    candidates["tfidf_score"] = _normalize_tfidf_scores(candidates["tfidf_raw_score"])
    candidates["pearson_score"] = _normalize_pearson_scores(candidates["pearson_raw_score"])
    candidates["tfidf_raw_score"] = pd.to_numeric(
        candidates["tfidf_raw_score"],
        errors="coerce",
    ).fillna(0.0)
    candidates["pearson_raw_score"] = pd.to_numeric(
        candidates["pearson_raw_score"],
        errors="coerce",
    ).fillna(0.0)

    svd_predictions = _predict_svd_for_candidates(user_id=userId, movie_ids=candidates["movieId"])
    candidates = candidates.merge(svd_predictions, on="movieId", how="left")
    candidates["svd_score"] = _normalize_svd_scores(
        candidates["svd_predicted_rating"],
        rating_scale=_require_rating_scale(),
    )

    candidates["final_hybrid_score"] = (
        candidates["tfidf_score"] * tfidf_weight
        + candidates["pearson_score"] * pearson_weight
        + candidates["svd_score"] * svd_weight
    )

    result = candidates.merge(_movie_metadata_frame(), on="movieId", how="left")

    sort_columns = ["final_hybrid_score", "svd_predicted_rating"]
    ascending = [False, False]

    if "rating_count" in result.columns:
        sort_columns.append("rating_count")
        ascending.append(False)

    result = result.sort_values(sort_columns, ascending=ascending)

    final_columns = [
        "movieId",
        "title",
        "year",
        "genres",
        "tfidf_raw_score",
        "tfidf_score",
        "pearson_raw_score",
        "pearson_score",
        "svd_predicted_rating",
        "svd_score",
        "svd_user_known",
        "svd_item_known",
        "final_hybrid_score",
    ]

    available_columns = [column for column in final_columns if column in result.columns]
    return result[available_columns].head(top_n).reset_index(drop=True)


def _prepare_ratings_frame(
    ratings_small: pd.DataFrame,
    *,
    user_col: str,
    item_col: str,
    rating_col: str,
) -> pd.DataFrame:
    """Validate and clean the ratings DataFrame before feeding Surprise."""

    if not isinstance(ratings_small, pd.DataFrame):
        raise TypeError("ratings_small must be a pandas DataFrame.")

    required_columns = {user_col, item_col, rating_col}
    missing_columns = required_columns.difference(ratings_small.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"ratings_small is missing required columns: {missing}")

    prepared = ratings_small[[user_col, item_col, rating_col]].copy()
    prepared = prepared.dropna(subset=[user_col, item_col, rating_col])
    prepared[rating_col] = pd.to_numeric(prepared[rating_col], errors="coerce")
    prepared = prepared.dropna(subset=[rating_col])

    if prepared.empty:
        raise ValueError("ratings_small does not contain any valid rows after cleaning.")

    if prepared[rating_col].nunique() < 2:
        raise ValueError("At least two distinct rating values are required to train SVD.")

    return prepared


def _infer_rating_scale(ratings: pd.Series) -> tuple[float, float]:
    """Infer the min/max rating scale directly from the training data."""

    min_rating = float(ratings.min())
    max_rating = float(ratings.max())

    if max_rating <= min_rating:
        raise ValueError(
            f"Invalid rating scale inferred from data: ({min_rating}, {max_rating})"
        )

    return (min_rating, max_rating)


def _validate_and_normalize_weights(weights: Sequence[float]) -> tuple[float, float, float]:
    """Ensure weights are usable and normalize them to sum to 1."""

    if len(weights) != 3:
        raise ValueError("weights must contain exactly three values: (tfidf, pearson, svd)")

    try:
        normalized_weights = tuple(float(weight) for weight in weights)
    except (TypeError, ValueError) as exc:
        raise TypeError("All weight values must be numeric.") from exc

    if any(weight < 0 for weight in normalized_weights):
        raise ValueError("weights cannot contain negative values.")

    total_weight = sum(normalized_weights)
    if total_weight <= 0:
        raise ValueError("At least one weight must be greater than zero.")

    return tuple(weight / total_weight for weight in normalized_weights)  # type: ignore[return-value]


def _coerce_recommender_output(result: pd.DataFrame | str, *, source_name: str) -> pd.DataFrame:
    """Convert an external recommender result into a predictable shape."""

    if isinstance(result, str):
        raise ValueError(f"{source_name} recommender returned an error: {result}")

    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"{source_name} recommender must return a pandas DataFrame.")

    required_columns = {"movieId", "similarity"}
    missing_columns = required_columns.difference(result.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{source_name} recommender output is missing columns: {missing}")

    standardized = result[["movieId", "similarity"]].copy()
    standardized["movieId"] = pd.to_numeric(standardized["movieId"], errors="coerce")
    standardized["similarity"] = pd.to_numeric(standardized["similarity"], errors="coerce")
    standardized = standardized.dropna(subset=["movieId", "similarity"])

    if standardized.empty:
        return pd.DataFrame(columns=["movieId", "raw_score"])

    standardized["movieId"] = standardized["movieId"].astype(int)
    standardized = standardized.rename(columns={"similarity": "raw_score"})

    # If upstream functions ever emit duplicates, keep the strongest signal.
    standardized = standardized.sort_values("raw_score", ascending=False)
    standardized = standardized.drop_duplicates(subset=["movieId"], keep="first")

    return standardized.reset_index(drop=True)


def _normalize_tfidf_scores(scores: pd.Series) -> pd.Series:
    """Normalize TF-IDF similarities to the [0, 1] range."""

    numeric_scores = pd.to_numeric(scores, errors="coerce")
    normalized = numeric_scores.clip(lower=0.0, upper=1.0)
    normalized = normalized.where(numeric_scores.notna(), 0.0)
    return normalized.fillna(0.0)


def _normalize_pearson_scores(scores: pd.Series) -> pd.Series:
    """Map Pearson similarities from [-1, 1] into [0, 1]."""

    numeric_scores = pd.to_numeric(scores, errors="coerce")
    normalized = (numeric_scores.clip(lower=-1.0, upper=1.0) + 1.0) / 2.0
    normalized = normalized.where(numeric_scores.notna(), 0.0)
    return normalized.fillna(0.0)


def _normalize_svd_scores(
    predicted_ratings: pd.Series,
    *,
    rating_scale: tuple[float, float],
) -> pd.Series:
    """Normalize SVD rating predictions to the [0, 1] range."""

    min_rating, max_rating = rating_scale
    if max_rating <= min_rating:
        raise ValueError(f"Invalid SVD rating scale: ({min_rating}, {max_rating})")

    numeric_scores = pd.to_numeric(predicted_ratings, errors="coerce")
    clipped_scores = numeric_scores.clip(lower=min_rating, upper=max_rating)
    normalized = (clipped_scores - min_rating) / (max_rating - min_rating)
    return normalized.fillna(0.0)


def _predict_svd_for_candidates(user_id: Any, movie_ids: pd.Series) -> pd.DataFrame:
    """Score each candidate movie with the trained Surprise SVD model."""

    model = _require_svd_model()
    trainset = getattr(model, "trainset", None)

    user_known = _is_known_user(model, user_id)
    if trainset is not None and not user_known:
        warnings.warn(
            f"userId={user_id!r} was not present in the SVD trainset. "
            "Surprise will fall back to a cold-start estimate.",
            RuntimeWarning,
            stacklevel=2,
        )

    prediction_rows = []
    unknown_item_count = 0

    for movie_id in movie_ids.tolist():
        item_known = _is_known_item(model, movie_id)
        if trainset is not None and not item_known:
            unknown_item_count += 1

        prediction = model.predict(uid=user_id, iid=movie_id)
        prediction_rows.append(
            {
                "movieId": int(movie_id),
                "svd_predicted_rating": float(prediction.est),
                "svd_user_known": bool(user_known),
                "svd_item_known": bool(item_known),
            }
        )

    if trainset is not None and unknown_item_count > 0:
        warnings.warn(
            f"{unknown_item_count} candidate movie(s) were not present in the "
            "SVD trainset. Surprise will use fallback estimates for them.",
            RuntimeWarning,
            stacklevel=2,
        )

    return pd.DataFrame(prediction_rows)


def _extract_rating_scale_from_model(model: Any) -> tuple[float, float]:
    """Best-effort extraction of the rating scale from a trained model."""

    trainset = getattr(model, "trainset", None)
    rating_scale = getattr(trainset, "rating_scale", None)

    if (
        isinstance(rating_scale, tuple)
        and len(rating_scale) == 2
        and rating_scale[1] > rating_scale[0]
    ):
        return (float(rating_scale[0]), float(rating_scale[1]))

    # MovieLens-style fallback if the artifact did not store metadata.
    return (0.5, 5.0)


def _is_known_user(model: Any, user_id: Any) -> bool:
    """Check whether a raw user id exists in the trained Surprise trainset."""

    trainset = getattr(model, "trainset", None)
    if trainset is None:
        return False

    try:
        trainset.to_inner_uid(user_id)
        return True
    except ValueError:
        return False


def _is_known_item(model: Any, movie_id: Any) -> bool:
    """Check whether a raw movie id exists in the trained Surprise trainset."""

    trainset = getattr(model, "trainset", None)
    if trainset is None:
        return False

    try:
        trainset.to_inner_iid(movie_id)
        return True
    except ValueError:
        return False


def _remove_seed_movie_from_candidates(candidates: pd.DataFrame, movie_title: str) -> pd.DataFrame:
    """Remove the source movie itself if it appears in the candidate pool."""

    if MOVIES_DF is None:
        return candidates

    title_column = None
    if "clean_title" in MOVIES_DF.columns:
        title_column = "clean_title"
    elif "title" in MOVIES_DF.columns:
        title_column = "title"

    if title_column is None:
        return candidates

    normalized_title = movie_title.strip().lower()
    matching_ids = MOVIES_DF.loc[
        MOVIES_DF[title_column].fillna("").str.strip().str.lower() == normalized_title,
        "movieId",
    ]

    if matching_ids.empty:
        return candidates

    return candidates[~candidates["movieId"].isin(matching_ids.astype(int))].copy()


def _movie_metadata_frame() -> pd.DataFrame:
    """Build a deduplicated metadata view for the final result frame."""

    if MOVIES_DF is None:
        raise RuntimeError("MOVIES_DF has not been configured.")

    title_source = "clean_title" if "clean_title" in MOVIES_DF.columns else "title"
    metadata_columns = ["movieId", title_source]

    for optional_column in ("year", "genres", "rating_count", "rating_mean"):
        if optional_column in MOVIES_DF.columns:
            metadata_columns.append(optional_column)

    metadata = MOVIES_DF[metadata_columns].copy()
    metadata = metadata.drop_duplicates(subset=["movieId"], keep="first")
    metadata = metadata.rename(columns={title_source: "title"})

    return metadata


def _ensure_hybrid_dependencies_are_configured() -> None:
    """Fail fast if the hybrid pipeline has not been initialized."""

    if MOVIES_DF is None:
        raise RuntimeError(
            "movies DataFrame is not configured. Call configure_hybrid_engine(...) first."
        )

    if CONTENT_RECOMMENDER is None:
        raise RuntimeError(
            "Content recommender is not configured. Call configure_hybrid_engine(...) first."
        )

    if COLLAB_RECOMMENDER is None:
        raise RuntimeError(
            "Collaborative recommender is not configured. Call configure_hybrid_engine(...) first."
        )

    if SVD_MODEL is None:
        raise RuntimeError(
            "SVD model is not loaded. Call train_svd_model(...) or load_svd_model(...) first."
        )


def _require_svd_model() -> Any:
    """Return the global SVD model or raise a clear error."""

    if SVD_MODEL is None:
        raise RuntimeError("SVD model is not loaded. Train or load it before inference.")
    return SVD_MODEL


def _require_rating_scale() -> tuple[float, float]:
    """Return the stored rating scale required for SVD score normalization."""

    if SVD_RATING_SCALE is None:
        raise RuntimeError(
            "SVD rating scale is unavailable. Load the model artifact or pass "
            "svd_rating_scale into configure_hybrid_engine(...)."
        )
    return SVD_RATING_SCALE


def _empty_hybrid_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the final schema already defined."""

    return pd.DataFrame(
        columns=[
            "movieId",
            "title",
            "year",
            "genres",
            "tfidf_raw_score",
            "tfidf_score",
            "pearson_raw_score",
            "pearson_score",
            "svd_predicted_rating",
            "svd_score",
            "svd_user_known",
            "svd_item_known",
            "final_hybrid_score",
        ]
    )
