from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from cineiq_explainability import add_explanations
from cineiq_hybrid_backend import (
    configure_hybrid_engine,
    hybrid_recommendation,
    load_svd_model,
)
from cineiq_sentiment_reranker import apply_sentiment_reranking


logger = logging.getLogger("cineiq.api")

PROJECT_ROOT = Path(__file__).resolve().parent
MOVIES_PATH = PROJECT_ROOT / "cleaned_data" / "movies_master.csv"
SVD_MODEL_PATH = PROJECT_ROOT / "model_artifacts" / "svd_model.pkl"
PEARSON_MODEL_PATH = PROJECT_ROOT / "model_artifacts" / "pearson_matrix.pkl"


class RecommendationRequest(BaseModel):
    userId: int
    movie_title: str
    top_n: int = Field(default=10, ge=1)


class RecommendationResponse(BaseModel):
    title: str
    final_hybrid_score: float
    sentiment_adjusted_score: float
    explanation: str


def _load_movies_frame() -> pd.DataFrame:
    if not MOVIES_PATH.exists():
        raise FileNotFoundError(f"Movie metadata file not found: {MOVIES_PATH}")

    movies_df = pd.read_csv(MOVIES_PATH, low_memory=False)

    required_columns = {"movieId", "title", "clean_title", "content_features", "user_tags_top10"}
    missing_columns = required_columns.difference(movies_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"movies_master.csv is missing required columns: {missing}")

    text_columns = ["title", "clean_title", "content_features", "user_tags_top10"]
    for column in text_columns:
        movies_df[column] = movies_df[column].fillna("")

    for column in ("rating_count", "rating_mean"):
        if column in movies_df.columns:
            movies_df[column] = pd.to_numeric(movies_df[column], errors="coerce").fillna(0)

    return movies_df


def _build_content_recommender(movies_df: pd.DataFrame) -> Callable[[str, int], pd.DataFrame | str]:
    content_movies = movies_df.copy()
    content_movies = content_movies[content_movies["content_features"].str.strip() != ""].copy()
    content_movies.reset_index(drop=True, inplace=True)

    if content_movies.empty:
        raise ValueError("movies_master.csv contains no usable rows for content recommendations.")

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10_000,
        ngram_range=(1, 2),
    )
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_movies["content_features"])

    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(tfidf_matrix)

    indexed_movies = content_movies.reset_index().rename(columns={"index": "content_index"})
    indexed_movies["clean_title_key"] = (
        indexed_movies["clean_title"].astype(str).str.lower().str.strip()
    )
    title_lookup = (
        indexed_movies.loc[indexed_movies["clean_title_key"] != ""]
        .drop_duplicates(subset="clean_title_key", keep="first")
        .set_index("clean_title_key")["content_index"]
        .to_dict()
    )

    def recommend_movies(title: str, top_n: int = 10) -> pd.DataFrame | str:
        normalized_title = title.lower().strip()
        if normalized_title not in title_lookup:
            return pd.DataFrame(columns=["movieId", "similarity"])

        content_index = int(title_lookup[normalized_title])
        neighbor_count = min(len(content_movies), max(top_n, 1) + 1)
        distances, neighbors = nn_model.kneighbors(
            tfidf_matrix[content_index],
            n_neighbors=neighbor_count,
        )

        recommendation_indexes = neighbors.flatten()[1:]
        recommendation_distances = distances.flatten()[1:]

        result_columns = [
            column
            for column in (
                "movieId",
                "clean_title",
                "year",
                "genres",
                "rating_count",
                "rating_mean",
                "director",
                "top_cast",
            )
            if column in content_movies.columns
        ]
        recommendation_frame = content_movies.iloc[recommendation_indexes][result_columns].copy()
        recommendation_frame["similarity"] = 1 - recommendation_distances
        sort_columns = ["similarity"]
        ascending = [False]
        if "rating_count" in recommendation_frame.columns:
            sort_columns.append("rating_count")
            ascending.append(False)
        recommendation_frame = recommendation_frame.sort_values(by=sort_columns, ascending=ascending)
        return recommendation_frame.reset_index(drop=True)

    return recommend_movies


def _build_collaborative_recommender(
    movies_df: pd.DataFrame,
) -> Callable[[str, int], pd.DataFrame | str]:
    if not PEARSON_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pearson matrix file not found: {PEARSON_MODEL_PATH}")

    movie_similarity = pd.read_pickle(PEARSON_MODEL_PATH)

    movie_title_frame = movies_df[["movieId", "clean_title"]].copy()
    movie_title_frame["clean_title"] = movie_title_frame["clean_title"].astype(str)
    movie_title_frame["clean_title_key"] = movie_title_frame["clean_title"].str.lower().str.strip()

    movie_id_to_title = (
        movie_title_frame.drop_duplicates(subset="movieId", keep="first")
        .set_index("movieId")["clean_title"]
        .to_dict()
    )
    title_to_movie_id = (
        movie_title_frame.loc[movie_title_frame["clean_title_key"] != ""]
        .drop_duplicates(subset="clean_title_key", keep="first")
        .set_index("clean_title_key")["movieId"]
        .to_dict()
    )

    def recommend_collaborative(movie_title: str, top_n: int = 10) -> pd.DataFrame | str:
        normalized_title = movie_title.lower().strip()
        if normalized_title not in title_to_movie_id:
            return pd.DataFrame(columns=["movieId", "similarity"])

        movie_id = int(title_to_movie_id[normalized_title])
        if movie_id not in movie_similarity.columns:
            return pd.DataFrame(columns=["movieId", "similarity"])

        similarity_scores = movie_similarity[movie_id].dropna().sort_values(ascending=False)
        similarity_scores = similarity_scores[similarity_scores.index != movie_id].head(top_n)

        if similarity_scores.empty:
           return pd.DataFrame(columns=["movieId", "similarity"])

        result = pd.DataFrame(
            {
                "movieId": similarity_scores.index.astype(int),
                "similarity": similarity_scores.values,
            }
        )
        result["title"] = result["movieId"].map(movie_id_to_title)
        return result[["movieId", "title", "similarity"]]

    return recommend_collaborative


def _is_movie_not_found_error(exc: Exception) -> bool:
    return "not found" in str(exc).lower()


def _lookup_movie_genres(movies_df: pd.DataFrame, movie_title: str) -> str:
    """Resolve the searched movie's genre string for explanation generation."""

    if not isinstance(movie_title, str) or not movie_title.strip():
        return ""
    if "genres" not in movies_df.columns:
        return ""

    normalized_title = movie_title.strip().casefold()
    title_columns = [column for column in ("clean_title", "title") if column in movies_df.columns]

    for column in title_columns:
        matches = movies_df.loc[
            movies_df[column].fillna("").astype(str).str.strip().str.casefold() == normalized_title,
            "genres",
        ]
        if matches.empty:
            continue

        genre_value = matches.iloc[0]
        if genre_value is None or pd.isna(genre_value):
            return ""

        cleaned_genres = str(genre_value).strip()
        return "" if cleaned_genres.lower() == "nan" else cleaned_genres

    return ""


def _build_response_payload(result_df: pd.DataFrame) -> list[RecommendationResponse]:
    response_columns = [
        "title",
        "final_hybrid_score",
        "sentiment_adjusted_score",
        "explanation",
    ]

    if result_df.empty:
        return []

    payload = result_df[response_columns].copy()
    payload["title"] = payload["title"].fillna("").astype(str)
    payload["final_hybrid_score"] = pd.to_numeric(
        payload["final_hybrid_score"],
        errors="coerce",
    ).fillna(0.0)
    payload["sentiment_adjusted_score"] = pd.to_numeric(
        payload["sentiment_adjusted_score"],
        errors="coerce",
    ).fillna(0.0)
    payload["explanation"] = payload["explanation"].fillna("").astype(str)

    return [
        RecommendationResponse(
            title=row["title"],
            final_hybrid_score=float(row["final_hybrid_score"]),
            sentiment_adjusted_score=float(row["sentiment_adjusted_score"]),
            explanation=row["explanation"],
        )
        for row in payload.to_dict(orient="records")
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading CineIQ recommendation assets...")

    movies_df = _load_movies_frame()
    svd_model = load_svd_model(SVD_MODEL_PATH)
    content_recommender = _build_content_recommender(movies_df)
    collaborative_recommender = _build_collaborative_recommender(movies_df)

    configure_hybrid_engine(
        movies_df=movies_df,
        svd_model_instance=svd_model,
        content_recommender=content_recommender,
        collaborative_recommender=collaborative_recommender,
    )

    app.state.movies_df = movies_df
    app.state.svd_model = svd_model

    logger.info("CineIQ recommendation engine is ready.")
    yield

    app.state.movies_df = None
    app.state.svd_model = None
    logger.info("CineIQ recommendation engine shut down.")


app = FastAPI(
    title="CineIQ Recommendation Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/api/v1/recommend",
    response_model=list[RecommendationResponse],
    status_code=status.HTTP_200_OK,
)
async def recommend(payload: RecommendationRequest, request: Request) -> list[RecommendationResponse]:
    try:
        hybrid_df = hybrid_recommendation(
            userId=payload.userId,
            movie_title=payload.movie_title,
            top_n=payload.top_n,
        )
        reranked_df = apply_sentiment_reranking(hybrid_df, request.app.state.movies_df)
        search_movie_genres = _lookup_movie_genres(
            request.app.state.movies_df,
            payload.movie_title,
        )
        explained_df = add_explanations(
            reranked_df,
            search_query=payload.movie_title,
            search_movie_genres=search_movie_genres,
        )
        return _build_response_payload(explained_df)
    except (RuntimeError, ValueError) as exc:
        status_code = status.HTTP_404_NOT_FOUND if _is_movie_not_found_error(exc) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
