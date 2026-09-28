"""Rule-based explainability utilities for CINEIQ recommendations.

This module builds dynamic explanation strings from the strongest
recommendation signal, shared metadata, and sentiment context.

Typical usage
-------------
```python
from cineiq_explainability import add_explanations

final_df = add_explanations(
    reranked_df,
    search_query="Toy Story",
    search_movie_genres="Adventure|Animation|Children|Comedy|Fantasy",
)
```
"""

from __future__ import annotations

from typing import Any

import pandas as pd


POSITIVE_SENTIMENT_THRESHOLD = 0.6
NEGATIVE_SENTIMENT_THRESHOLD = -0.4


def add_explanations(
    recommendations_df: pd.DataFrame,
    *,
    search_query: str = "",
    search_movie_genres: Any = None,
) -> pd.DataFrame:
    """Attach a dynamic explanation string to each recommendation row.

    Parameters
    ----------
    recommendations_df:
        Recommendation DataFrame containing at least:
        - `tfidf_score`
        - `pearson_score`
        - `svd_score`
        - `vader_compound_score`
    search_query:
        The original movie title the user searched for.
    search_movie_genres:
        The genre string for the searched movie, if available.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with a new `explanation` column.
    """

    _validate_recommendation_schema(recommendations_df)

    if recommendations_df.empty:
        empty_result = recommendations_df.copy()
        empty_result["explanation"] = pd.Series(dtype=str)
        return empty_result

    result = recommendations_df.copy()

    score_columns = ["tfidf_score", "pearson_score", "svd_score", "vader_compound_score"]
    for column in score_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    # Missing scores are treated as neutral/absent signals so the row can still
    # receive an explanation instead of failing halfway through the pipeline.
    result["tfidf_score"] = result["tfidf_score"].fillna(0.0)
    result["pearson_score"] = result["pearson_score"].fillna(0.0)
    result["svd_score"] = result["svd_score"].fillna(0.0)
    result["vader_compound_score"] = result["vader_compound_score"].fillna(0.0)

    normalized_search_query = _normalize_search_query(search_query)
    result["explanation"] = result.apply(
        lambda row: _build_explanation_for_row(
            row,
            search_query=normalized_search_query,
            search_movie_genres=search_movie_genres,
        ),
        axis=1,
    )
    return result


def _validate_recommendation_schema(recommendations_df: pd.DataFrame) -> None:
    """Verify that the DataFrame contains the columns required for explanations."""

    if not isinstance(recommendations_df, pd.DataFrame):
        raise TypeError("recommendations_df must be a pandas DataFrame.")

    required_columns = {
        "tfidf_score",
        "pearson_score",
        "svd_score",
        "vader_compound_score",
    }
    missing_columns = required_columns.difference(recommendations_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            "recommendations_df is missing required columns: "
            f"{missing}"
        )


def _build_explanation_for_row(
    row: pd.Series,
    *,
    search_query: str,
    search_movie_genres: Any,
) -> str:
    """Create one explanation string for a recommendation row."""

    primary_driver = _select_primary_driver(row)
    movie_title = _format_movie_reference(row.get("title"))
    shared_genres = _shared_genres(
        recommendation_genres=row.get("genres"),
        search_movie_genres=search_movie_genres,
    )
    driver_sentence = _driver_specific_explanation(
        primary_driver,
        movie_title=movie_title,
        search_query=search_query,
        shared_genres=shared_genres,
        row=row,
    )

    sentiment_explanation = _sentiment_suffix(row.get("vader_compound_score", 0.0))
    if not sentiment_explanation:
        return driver_sentence

    return f"{driver_sentence} {sentiment_explanation}"


def _select_primary_driver(row: pd.Series) -> str:
    """Find the dominant recommendation signal for a row.

    Tie handling is deterministic: if two or more scores are equal, the first
    signal in the order TF-IDF -> Pearson -> SVD wins.
    """

    scored_drivers = [
        ("tfidf", _safe_float(row.get("tfidf_score", 0.0))),
        ("pearson", _safe_float(row.get("pearson_score", 0.0))),
        ("svd", _safe_float(row.get("svd_score", 0.0))),
    ]
    return max(scored_drivers, key=lambda item: item[1])[0]


def _driver_specific_explanation(
    driver_name: str,
    *,
    movie_title: str,
    search_query: str,
    shared_genres: list[str],
    row: pd.Series,
) -> str:
    """Build a movie-specific explanation sentence for the strongest driver."""

    genre_phrase = _join_with_and(shared_genres)
    variant_index = _template_variant_index(
        driver_name,
        movie_title,
        search_query,
        *shared_genres,
        variant_count=3,
    )

    if driver_name == "tfidf":
        if genre_phrase:
            options = [
                f"Our content analyzer matched {movie_title} to your search due to shared themes like {genre_phrase}.",
                f"{movie_title} stood out in our TF-IDF content model because it overlaps with your search on genres such as {genre_phrase}.",
                f"Content-based matching surfaced {movie_title} after spotting strong thematic overlap around {genre_phrase}.",
            ]
        else:
            options = [
                f"Our content analyzer matched {movie_title} to your search because its themes closely align with what you looked up.",
                f"{movie_title} stood out in our TF-IDF content model thanks to strong content overlap with your search.",
                f"Content-based matching surfaced {movie_title} after its story signals aligned closely with your query.",
            ]
        return options[variant_index]

    if driver_name == "pearson":
        if search_query and genre_phrase:
            options = [
                f"Viewers who liked {search_query} also highly rated {movie_title}, with shared appeal in {genre_phrase}.",
                f"Our Pearson collaborative signal linked {movie_title} to fans of {search_query}, especially around {genre_phrase}.",
                f"People who responded well to {search_query} also tended to enjoy {movie_title}, particularly for its {genre_phrase} elements.",
            ]
        elif search_query:
            options = [
                f"Viewers who liked {search_query} also highly rated {movie_title}.",
                f"Our Pearson collaborative signal linked {movie_title} to fans of {search_query}.",
                f"People who responded well to {search_query} also tended to enjoy {movie_title}.",
            ]
        elif genre_phrase:
            options = [
                f"Similar viewers pushed {movie_title} upward in our collaborative model, with strong overlap in {genre_phrase}.",
                f"Our Pearson taste matching favored {movie_title} for users with patterns like yours, especially around {genre_phrase}.",
                f"Collaborative filtering highlighted {movie_title} because people with similar preferences responded well to its focus on {genre_phrase}.",
            ]
        else:
            options = [
                f"Viewers with taste patterns similar to yours helped push {movie_title} higher in our collaborative model.",
                f"Our Pearson taste matching flagged {movie_title} as a strong pick among users who rate movies like you do.",
                f"Collaborative filtering highlighted {movie_title} because people with similar preferences responded well to it.",
            ]
        return options[variant_index]

    if driver_name == "svd":
        predicted_rating = _optional_predicted_rating(row.get("svd_predicted_rating"))
        if genre_phrase and predicted_rating is not None:
            options = [
                f"Our AI predicts you will highly rate {movie_title} based on your unique taste profile and interest in {genre_phrase}.",
                f"The SVD personalization model sees {movie_title} as a strong fit for you, pairing your interest in {genre_phrase} with an estimated rating of {predicted_rating:.2f}.",
                f"Your latent taste profile points toward {movie_title}, with your interest in {genre_phrase} supporting a predicted rating of {predicted_rating:.2f}.",
            ]
        elif genre_phrase:
            options = [
                f"Our AI predicts you will highly rate {movie_title} based on your unique taste profile and interest in {genre_phrase}.",
                f"The SVD personalization model sees {movie_title} as a strong fit for you, especially around {genre_phrase}.",
                f"Your latent taste profile points toward {movie_title}, with your interest in {genre_phrase} reinforcing that prediction.",
            ]
        elif predicted_rating is not None:
            options = [
                f"Our AI predicts you will highly rate {movie_title} based on your unique taste profile.",
                f"The SVD personalization model sees {movie_title} as a strong fit for you with an estimated rating of {predicted_rating:.2f}.",
                f"Your latent taste profile points toward {movie_title}, with a predicted rating of {predicted_rating:.2f}.",
            ]
        else:
            options = [
                f"Our AI predicts you will highly rate {movie_title} based on your unique taste profile.",
                f"The SVD personalization model sees {movie_title} as a strong fit for your preferences.",
                f"Your latent taste profile points toward {movie_title} as a natural fit for what you usually enjoy.",
            ]
        return options[variant_index]

    raise ValueError(f"Unsupported explanation driver: {driver_name}")


def _shared_genres(
    recommendation_genres: Any,
    search_movie_genres: Any,
    *,
    limit: int = 2,
) -> list[str]:
    """Return the first shared genres between the search and recommendation."""

    recommended_genres = _parse_genres(recommendation_genres)
    if not recommended_genres:
        return []

    search_genre_keys = {
        genre.casefold()
        for genre in _parse_genres(search_movie_genres)
    }
    if not search_genre_keys:
        return []

    return [
        genre
        for genre in recommended_genres
        if genre.casefold() in search_genre_keys
    ][:max(limit, 0)]


def _sentiment_suffix(vader_compound_score: Any) -> str:
    """Return the optional VADER-based explanation suffix."""

    compound_score = _safe_float(vader_compound_score)

    if compound_score > POSITIVE_SENTIMENT_THRESHOLD:
        return (
            "\U0001F525 Audiences are currently raving about it with a "
            f"sentiment score of {compound_score:.2f}!"
        )
    if compound_score < NEGATIVE_SENTIMENT_THRESHOLD:
        return (
            "Audience reactions are more mixed right now with a sentiment "
            f"score of {compound_score:.2f}."
        )
    return ""


def _template_variant_index(*seed_parts: Any, variant_count: int) -> int:
    """Pick a deterministic template variant so similar rows read differently."""

    if variant_count <= 0:
        return 0

    seed_text = "|".join(
        str(part).strip()
        for part in seed_parts
        if part is not None and not pd.isna(part) and str(part).strip()
    )
    if not seed_text:
        return 0

    return sum(ord(character) for character in seed_text) % variant_count


def _format_movie_reference(raw_title: Any) -> str:
    """Return a human-friendly movie reference for explanation sentences."""

    if raw_title is None or pd.isna(raw_title):
        return "this movie"

    cleaned_title = str(raw_title).strip()
    if not cleaned_title or cleaned_title.lower() == "nan":
        return "this movie"

    return f'"{cleaned_title}"'


def _optional_predicted_rating(value: Any) -> float | None:
    """Return the SVD predicted rating when it is present and numeric."""

    if value is None or pd.isna(value):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_genres(raw_genres: Any) -> list[str]:
    """Parse a MovieLens-style pipe-delimited genre string into clean labels."""

    if raw_genres is None or pd.isna(raw_genres):
        return []

    cleaned_genres = str(raw_genres).strip()
    if not cleaned_genres or cleaned_genres.lower() == "nan":
        return []
    if cleaned_genres == "(no genres listed)":
        return []

    deduped_genres: list[str] = []
    seen_genres: set[str] = set()
    for genre in cleaned_genres.split("|"):
        normalized_genre = genre.strip()
        if not normalized_genre:
            continue

        lowered_genre = normalized_genre.casefold()
        if lowered_genre in seen_genres:
            continue

        seen_genres.add(lowered_genre)
        deduped_genres.append(normalized_genre)

    return deduped_genres


def _join_with_and(values: list[str]) -> str:
    """Join one or two short labels into a natural-language phrase."""

    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return f"{values[0]} and {values[1]}"


def _normalize_search_query(search_query: Any) -> str:
    """Normalize the user-entered search query without changing its casing."""

    if search_query is None or pd.isna(search_query):
        return ""

    cleaned_query = str(search_query).strip()
    if cleaned_query.lower() == "nan":
        return ""

    return cleaned_query


def _safe_float(value: Any) -> float:
    """Convert a value into float without raising on null-like inputs."""

    if value is None or pd.isna(value):
        return 0.0

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
