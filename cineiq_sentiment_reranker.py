"""Sentiment-aware re-ranking for CINEIQ hybrid recommendations.

This module applies a lightweight VADER-based post-processing step on top of
the existing `hybrid_recommendation(...)` output. It uses each movie's
`user_tags_top10` text to gently boost or dampen the original
`final_hybrid_score`.

Typical usage
-------------
```python
from cineiq_sentiment_reranker import apply_sentiment_reranking

hybrid_df = hybrid_recommendation(userId=1, movie_title="toy story", top_n=10)
reranked_df = apply_sentiment_reranking(hybrid_df, movies)
```
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd


def apply_sentiment_reranking(
    recommendations_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    sentiment_weight: float = 0.15,
) -> pd.DataFrame:
    """Re-rank hybrid recommendations using VADER sentiment from user tags.

    Parameters
    ----------
    recommendations_df:
        Output of `hybrid_recommendation(...)`. Must contain `movieId` and
        `final_hybrid_score`.
    movies_df:
        Master movie DataFrame containing `movieId` and `user_tags_top10`.
    sentiment_weight:
        Maximum proportional adjustment applied to the hybrid score.
        The multiplier is computed as:

            1.0 + (compound_score * sentiment_weight)

        With the default value of `0.15`, the multiplier range is:
        `[0.85, 1.15]`.

    Returns
    -------
    pd.DataFrame
        A copy of the input recommendations DataFrame, enriched with:
        - `vader_compound_score`
        - `sentiment_multiplier`
        - `sentiment_adjusted_score`

        Results are sorted by `sentiment_adjusted_score` descending.
    """

    _validate_recommendations_frame(recommendations_df)
    _validate_movies_frame(movies_df)
    sentiment_weight = _validate_sentiment_weight(sentiment_weight)

    if recommendations_df.empty:
        empty_result = recommendations_df.copy()
        empty_result["vader_compound_score"] = pd.Series(dtype=float)
        empty_result["sentiment_multiplier"] = pd.Series(dtype=float)
        empty_result["sentiment_adjusted_score"] = pd.Series(dtype=float)
        return empty_result

    analyzer = _get_vader_analyzer()

    result = recommendations_df.copy()
    result["movieId"] = pd.to_numeric(result["movieId"], errors="coerce")
    result["final_hybrid_score"] = pd.to_numeric(
        result["final_hybrid_score"],
        errors="coerce",
    )
    result = result.dropna(subset=["movieId", "final_hybrid_score"]).copy()
    result["movieId"] = result["movieId"].astype(int)

    if result.empty:
        empty_result = recommendations_df.iloc[0:0].copy()
        empty_result["vader_compound_score"] = pd.Series(dtype=float)
        empty_result["sentiment_multiplier"] = pd.Series(dtype=float)
        empty_result["sentiment_adjusted_score"] = pd.Series(dtype=float)
        return empty_result

    tag_lookup = _build_tag_lookup(movies_df)
    result["user_tags_text"] = result["movieId"].map(tag_lookup).fillna("")

    sentiment_scores = result["user_tags_text"].apply(
        lambda text: _compute_vader_compound_score(text, analyzer)
    )
    result["vader_compound_score"] = sentiment_scores.astype(float)
    result["sentiment_multiplier"] = result["vader_compound_score"].apply(
        lambda compound: _compound_to_multiplier(
            compound_score=compound,
            sentiment_weight=sentiment_weight,
        )
    )
    result["sentiment_adjusted_score"] = (
        result["final_hybrid_score"] * result["sentiment_multiplier"]
    )

    result = result.sort_values(
        by=["sentiment_adjusted_score", "final_hybrid_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result.drop(columns=["user_tags_text"])


@lru_cache(maxsize=1)
def _get_vader_analyzer() -> Any:
    """Create and cache the VADER analyzer instance."""

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise ImportError(
            "vaderSentiment is required for sentiment re-ranking. "
            "Install it with: pip install vaderSentiment"
        ) from exc

    return SentimentIntensityAnalyzer()


def _validate_recommendations_frame(recommendations_df: pd.DataFrame) -> None:
    """Ensure the hybrid recommender output has the required schema."""

    if not isinstance(recommendations_df, pd.DataFrame):
        raise TypeError("recommendations_df must be a pandas DataFrame.")

    required_columns = {"movieId", "final_hybrid_score"}
    missing_columns = required_columns.difference(recommendations_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "recommendations_df is missing required columns: "
            f"{missing}"
        )


def _validate_movies_frame(movies_df: pd.DataFrame) -> None:
    """Ensure the master movie frame contains tag text for sentiment analysis."""

    if not isinstance(movies_df, pd.DataFrame):
        raise TypeError("movies_df must be a pandas DataFrame.")

    required_columns = {"movieId", "user_tags_top10"}
    missing_columns = required_columns.difference(movies_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"movies_df is missing required columns: {missing}")


def _validate_sentiment_weight(sentiment_weight: float) -> float:
    """Keep the sentiment adjustment interpretable and bounded."""

    try:
        parsed_weight = float(sentiment_weight)
    except (TypeError, ValueError) as exc:
        raise TypeError("sentiment_weight must be numeric.") from exc

    if parsed_weight < 0 or parsed_weight > 1:
        raise ValueError("sentiment_weight must be between 0 and 1 inclusive.")

    return parsed_weight


def _build_tag_lookup(movies_df: pd.DataFrame) -> dict[int, str]:
    """Create a movieId -> cleaned tag paragraph lookup."""

    tag_frame = movies_df[["movieId", "user_tags_top10"]].copy()
    tag_frame["movieId"] = pd.to_numeric(tag_frame["movieId"], errors="coerce")
    tag_frame = tag_frame.dropna(subset=["movieId"]).copy()
    tag_frame["movieId"] = tag_frame["movieId"].astype(int)
    tag_frame = tag_frame.drop_duplicates(subset=["movieId"], keep="first")
    tag_frame["user_tags_top10"] = tag_frame["user_tags_top10"].apply(_normalize_user_tag_text)
    return dict(zip(tag_frame["movieId"], tag_frame["user_tags_top10"]))


def _normalize_user_tag_text(raw_text: Any) -> str:
    """Convert pipe-separated tags into one VADER-friendly text paragraph."""

    if raw_text is None or pd.isna(raw_text):
        return ""

    cleaned_text = str(raw_text).replace("|", " ").strip()
    if cleaned_text.lower() == "nan":
        return ""

    # Collapse extra whitespace so the analyzer sees a clean sentence-like blob.
    return " ".join(cleaned_text.split())


def _compute_vader_compound_score(tag_text: str, analyzer: Any) -> float:
    """Return the VADER compound sentiment score for a movie's tags."""

    normalized_text = _normalize_user_tag_text(tag_text)
    if not normalized_text:
        return 0.0

    polarity = analyzer.polarity_scores(normalized_text)
    compound_score = float(polarity.get("compound", 0.0))
    return max(-1.0, min(1.0, compound_score))


def _compound_to_multiplier(compound_score: float, sentiment_weight: float) -> float:
    """Map a VADER compound score into a gentle hybrid-score multiplier."""

    bounded_compound = max(-1.0, min(1.0, float(compound_score)))
    return 1.0 + (bounded_compound * sentiment_weight)
