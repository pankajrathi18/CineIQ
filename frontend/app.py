from __future__ import annotations

import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/api/v1/recommend"
DEFAULT_USER_ID = 1
REQUEST_TIMEOUT_SECONDS = 60


st.set_page_config(page_title="CineIQ Recommender", layout="wide")

st.title("CineIQ Recommender")
st.markdown(
    """
    Explore recommendations powered by CineIQ's hybrid ML engine, which combines
    TF-IDF content similarity, Pearson collaborative filtering, SVD-based
    personalization, sentiment reranking, and explainable AI output.
    """
)
st.caption("This frontend currently queries the backend using demo profile `userId=1`.")


def _format_score(value: float | int | str | None) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or "Unexpected error returned by backend."

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if detail:
            return str(detail)

    return "Unexpected error returned by backend."


with st.form("recommendation_form"):
    movie_title = st.text_input("Movie title", placeholder="Enter a movie title, e.g. Toy Story")
    top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=10)
    submitted = st.form_submit_button("Get Recommendations")


if submitted:
    if not movie_title.strip():
        st.warning("Please enter a movie title before submitting.")
    else:
        request_payload = {
            "userId": DEFAULT_USER_ID,
            "movie_title": movie_title.strip(),
            "top_n": top_n,
        }

        try:
            with st.spinner("Generating recommendations..."):
                response = requests.post(
                    API_URL,
                    json=request_payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

            if response.status_code == 200:
                recommendations = response.json()

                if not recommendations:
                    st.info("No recommendations were returned for this request.")
                else:
                    st.success(f"Found {len(recommendations)} recommendation(s).")

                    for movie in recommendations:
                        st.subheader(movie.get("title", "Untitled Recommendation"))

                        score_col, hybrid_col = st.columns(2)
                        with score_col:
                            st.metric(
                                "Sentiment-Adjusted Score",
                                _format_score(movie.get("sentiment_adjusted_score")),
                            )
                        with hybrid_col:
                            st.metric(
                                "Final Hybrid Score",
                                _format_score(movie.get("final_hybrid_score")),
                            )

                        explanation = str(movie.get("explanation") or "").strip()
                        st.info(explanation or "No explanation available.")
                        st.divider()

            elif response.status_code == 404:
                st.warning(_response_detail(response))
            else:
                st.error(
                    f"Backend request failed with status {response.status_code}: "
                    f"{_response_detail(response)}"
                )

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the FastAPI backend. Please ensure the Docker backend is running."
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
