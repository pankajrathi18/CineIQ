from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RATINGS_PATH = PROJECT_ROOT / "raw_data" / "ml-25m" / "ratings.csv"
OUTPUT_PATH = PROJECT_ROOT / "model_artifacts" / "pearson_matrix.pkl"
CHUNK_SIZE = 250_000
SAMPLE_FRAC = 0.024
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 50
MIN_PERIODS = 20


def load_sampled_ratings(ratings_path: Path) -> pd.DataFrame:
    if not ratings_path.exists():
        raise FileNotFoundError(f"ratings.csv not found: {ratings_path}")

    sampled_chunks: list[pd.DataFrame] = []

    chunk_reader = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating"],
        chunksize=CHUNK_SIZE,
    )

    for chunk_index, chunk in enumerate(chunk_reader):
        sampled_chunk = chunk.sample(
            frac=SAMPLE_FRAC,
            random_state=42 + chunk_index,
        )
        sampled_chunks.append(sampled_chunk)

    if not sampled_chunks:
        raise ValueError("No sampled chunks were produced from ratings.csv.")

    return pd.concat(sampled_chunks, ignore_index=True)


def filter_active_users_and_popular_movies(ratings: pd.DataFrame) -> pd.DataFrame:
    user_counts = ratings["userId"].value_counts()
    movie_counts = ratings["movieId"].value_counts()

    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    popular_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index

    filtered_ratings = ratings[
        ratings["userId"].isin(active_users) & ratings["movieId"].isin(popular_movies)
    ].copy()

    if filtered_ratings.empty:
        raise ValueError(
            "Filtered ratings are empty after applying active user and popular movie thresholds."
        )

    return filtered_ratings


def build_pearson_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
    )

    pearson_matrix = user_movie_matrix.corr(
        method="pearson",
        min_periods=MIN_PERIODS,
    )

    if pearson_matrix.empty:
        raise ValueError("Pearson correlation matrix is empty.")

    return pearson_matrix


def main() -> None:
    print(f"Loading ratings from: {RATINGS_PATH}")
    sampled_ratings = load_sampled_ratings(RATINGS_PATH)
    print(f"Sampled ratings shape: {sampled_ratings.shape}")

    filtered_ratings = filter_active_users_and_popular_movies(sampled_ratings)
    print(f"Filtered ratings shape: {filtered_ratings.shape}")

    pearson_matrix = build_pearson_matrix(filtered_ratings)
    print(f"Pearson matrix shape: {pearson_matrix.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(pearson_matrix, OUTPUT_PATH)
    print(f"Saved Pearson matrix to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
