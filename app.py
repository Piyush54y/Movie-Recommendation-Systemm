import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "movies.csv"
MODEL_FILE = BASE_DIR / "models" / "movie_recommender.pkl"

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide"
)


@st.cache_resource
def build_or_load_model():
    """Load the saved model. If it is absent (common on Streamlit Cloud),
    build it automatically from data/movies.csv."""
    if MODEL_FILE.exists():
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Make sure data/movies.csv is committed "
            "to the GitHub repository."
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "title", "genres", "overview", "keywords",
        "cast", "director", "vote_average", "vote_count"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "The CSV is missing these columns: " + ", ".join(missing)
        )

    text_columns = ["genres", "overview", "keywords", "cast", "director"]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    df["title"] = df["title"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    df["tags"] = (
        df["genres"] + " " +
        df["keywords"] + " " +
        df["cast"] + " " +
        df["director"] + " " +
        df["overview"]
    ).str.lower()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    vectors = vectorizer.fit_transform(df["tags"])
    similarity = cosine_similarity(vectors)

    title_to_index = {
        title.lower(): i for i, title in enumerate(df["title"])
    }

    return {
        "movies": df.to_dict("records"),
        "similarity": similarity,
        "title_to_index": title_to_index
    }


try:
    model = build_or_load_model()
except Exception as e:
    st.error("⚠️ Application setup error")
    st.code(str(e))
    st.info(
        "For Streamlit Cloud, confirm that your GitHub repository contains "
        "both app.py and data/movies.csv."
    )
    st.stop()

movies = model["movies"]
similarity = model["similarity"]
title_to_index = model["title_to_index"]

titles = sorted([m["title"] for m in movies])

st.title("🎬 CineMatch AI")
st.caption("Movie Recommendation System Using Machine Learning")

st.markdown(
    "Select a movie and discover similar movies using "
    "**TF-IDF Vectorization + Cosine Similarity**."
)

selected = st.selectbox("🎥 Choose a movie", titles)

max_recommendations = min(10, max(3, len(titles) - 1))
default_recommendations = min(5, max_recommendations)

count = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=max_recommendations,
    value=default_recommendations
)

if st.button("✨ Recommend Movies", type="primary"):
    idx = title_to_index[selected.lower()]

    scores = list(enumerate(similarity[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)

    recommendations = [
        (movies[i], score)
        for i, score in scores
        if i != idx
    ][:count]

    st.subheader(f"Because you selected: {selected}")

    for rank, (movie, score) in enumerate(recommendations, start=1):
        with st.container(border=True):
            st.markdown(f"### {rank}. {movie['title']}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Similarity", f"{score * 100:.1f}%")
            c2.metric("Rating", f"{float(movie['vote_average']):.1f}/10")
            c3.metric("Votes", f"{int(movie['vote_count']):,}")

            st.write(f"**Genres:** {movie['genres']}")
            st.write(f"**Director:** {movie['director']}")
            st.write(movie["overview"])

st.sidebar.title("📌 Project Information")
st.sidebar.write(
    """
**Technique:** Content-Based Filtering

**ML/NLP:**
- TF-IDF Vectorization
- Cosine Similarity

**Frontend:** Streamlit

**Dataset:** Movie CSV

The app automatically trains the recommendation
model if the saved `.pkl` model is not present.
"""
)
