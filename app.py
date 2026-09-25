import random
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
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
:root {
    --accent: #ff4b6e;
    --accent2: #7c4dff;
}
.hero {
    padding: 34px 34px 28px 34px;
    border-radius: 24px;
    background: linear-gradient(135deg, #17182b 0%, #30204a 55%, #4b1f38 100%);
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 35px rgba(0,0,0,.16);
}
.hero h1 { font-size: 3.2rem; margin: 0; letter-spacing: -2px; }
.hero p { color: #e7e3ed; font-size: 1.05rem; margin-top: 8px; }
.badge {
    display: inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    margin-right: 7px;
    font-size: .82rem;
}
.card {
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 18px;
    padding: 18px;
    height: 100%;
    background: rgba(128,128,128,.045);
    transition: .2s;
}
.card:hover { border-color: var(--accent); transform: translateY(-2px); }
.rank {
    font-size: .78rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 1px;
}
.movie-title { font-size: 1.15rem; font-weight: 750; margin: 5px 0 8px; }
.muted { color: #8b8b98; font-size: .86rem; }
.score {
    display:inline-block;
    padding:5px 9px;
    border-radius:10px;
    background: rgba(124,77,255,.13);
    font-weight:700;
    font-size:.82rem;
}
.genre {
    display:inline-block;
    padding:4px 8px;
    border-radius:8px;
    background: rgba(255,75,110,.10);
    margin:2px;
    font-size:.74rem;
}
.section-title { margin-top: 22px; }
.small-note { font-size:.82rem; color:#888; }
</style>
""", unsafe_allow_html=True)


# ---------- Model ----------
@st.cache_resource
def build_or_load_model():
    if MODEL_FILE.exists():
        try:
            with open(MODEL_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "data/movies.csv was not found. Upload the CSV to the data folder."
        )

    df = pd.read_csv(DATA_FILE)

    required = [
        "title", "genres", "overview", "keywords",
        "cast", "director", "vote_average", "vote_count"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    for col in ["genres", "overview", "keywords", "cast", "director"]:
        df[col] = df[col].fillna("").astype(str)

    df["title"] = df["title"].fillna("").astype(str)
    df = df.drop_duplicates("title").reset_index(drop=True)

    df["tags"] = (
        df["genres"] + " " + df["keywords"] + " " +
        df["cast"] + " " + df["director"] + " " +
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
        "title_to_index": title_to_index,
    }


try:
    model = build_or_load_model()
except Exception as e:
    st.error("⚠️ App setup error")
    st.code(str(e))
    st.stop()

movies = model["movies"]
similarity = model["similarity"]
title_to_index = model["title_to_index"]

titles = sorted([m["title"] for m in movies])


def movie_index(title):
    return title_to_index.get(title.lower().strip())


def recommend(title, n=5):
    idx = movie_index(title)
    if idx is None:
        return []
    scores = list(enumerate(similarity[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        (movies[i], float(score))
        for i, score in scores
        if i != idx
    ][:n]


def genres(movie):
    return [x.strip() for x in str(movie["genres"]).split("|") if x.strip()]


def find_by_genre(genre, n=5):
    candidates = [m for m in movies if genre.lower() in str(m["genres"]).lower()]
    candidates.sort(
        key=lambda m: (
            float(m["vote_average"]),
            int(m["vote_count"])
        ),
        reverse=True
    )
    return candidates[:n]


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="badge">🤖 AI POWERED</div>
    <div class="badge">🧠 TF-IDF + COSINE SIMILARITY</div>
    <div class="badge">🎬 PERSONALIZED DISCOVERY</div>
    <h1>🎬 CineMatch AI</h1>
    <p>Don't search for your next movie. Let AI discover it for you.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎛️ Discovery Lab")
    st.caption("Tune your recommendation experience.")

    mode = st.radio(
        "Explore mode",
        ["✨ For You", "🔥 Top Rated", "🎲 Surprise Me", "🎭 By Genre"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### 📊 Dataset")
    st.metric("Movies", len(movies))
    avg_rating = sum(float(m["vote_average"]) for m in movies) / max(1, len(movies))
    st.metric("Avg. rating", f"{avg_rating:.1f}/10")

    all_genres = sorted({
        g.strip()
        for m in movies
        for g in str(m["genres"]).split("|")
        if g.strip()
    })
    st.caption(f"{len(all_genres)} genres available")

# ---------- Main modes ----------
if mode == "✨ For You":
    st.markdown("## 🔮 Find your next watch")
    st.write("Choose a movie you already like. CineMatch will find movies with similar content.")

    selected = st.selectbox(
        "🎥 Pick a movie",
        titles,
        index=titles.index("3 Idiots") if "3 Idiots" in titles else 0,
        label_visibility="collapsed",
    )

    selected_movie = movies[movie_index(selected)]
    selected_genres = genres(selected_movie)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("⭐ Rating", f"{float(selected_movie['vote_average']):.1f}/10")
    c2.metric("🎭 Genres", len(selected_genres))
    c3.metric("👥 Votes", f"{int(selected_movie['vote_count']):,}")
    c4.metric("🎯 Engine", "Content AI")

    if selected_genres:
        st.markdown(
            "".join(f'<span class="genre">{g}</span>' for g in selected_genres),
            unsafe_allow_html=True
        )

    with st.expander("🧠 Why this movie?", expanded=False):
        st.write(selected_movie["overview"])
        st.caption(
            "The recommendation engine compares genre, keywords, cast, "
            "director and overview using TF-IDF and cosine similarity."
        )

    st.markdown('<div class="section-title"></div>', unsafe_allow_html=True)

    n = st.slider("Recommendations", 3, min(10, max(3, len(titles)-1)), 6)
    if st.button("✨ Find My Movies", type="primary", use_container_width=True):
        st.session_state["results"] = recommend(selected, n)
        st.session_state["source"] = selected

    if "results" in st.session_state and st.session_state.get("source") == selected:
        results = st.session_state["results"]
        st.markdown(f"## 🍿 Because you liked **{selected}**")

        cols = st.columns(3)
        for rank, ((movie, score), col) in enumerate(zip(results, cols * 4), 1):
            with col:
                st.markdown(f"""
                <div class="card">
                    <div class="rank">MATCH #{rank}</div>
                    <div class="movie-title">{movie["title"]}</div>
                    <span class="score">🎯 {score*100:.1f}% match</span>
                    <p class="muted">⭐ {float(movie["vote_average"]):.1f}/10 · 👥 {int(movie["vote_count"]):,} votes</p>
                    <p>{movie["overview"][:220]}{"..." if len(movie["overview"]) > 220 else ""}</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("More details"):
                    st.write(f"**Genres:** {movie['genres']}")
                    st.write(f"**Director:** {movie['director']}")
                    st.write(f"**Cast:** {movie['cast']}")

elif mode == "🔥 Top Rated":
    st.markdown("## 🔥 Top Rated Movies")
    st.caption("Explore highly rated movies in the current dataset.")

    top = sorted(
        movies,
        key=lambda m: (float(m["vote_average"]), int(m["vote_count"])),
        reverse=True
    )[:10]

    cols = st.columns(3)
    for rank, (movie, col) in enumerate(zip(top, cols * 4), 1):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="rank">TOP #{rank}</div>
                <div class="movie-title">{movie['title']}</div>
                <span class="score">⭐ {float(movie['vote_average']):.1f}/10</span>
                <p class="muted">{movie['genres']}</p>
                <p>{movie['overview'][:180]}...</p>
            </div>
            """, unsafe_allow_html=True)

elif mode == "🎲 Surprise Me":
    st.markdown("## 🎲 Surprise Me")
    st.write("Can't decide? Let CineMatch pick a movie from your dataset.")

    if st.button("🎲 Pick Something For Me", type="primary", use_container_width=True):
        pick = random.choice(movies)
        st.session_state["surprise"] = pick

    if "surprise" in st.session_state:
        movie = st.session_state["surprise"]
        st.success(f"Your surprise pick: **{movie['title']}**")
        st.markdown(f"### ⭐ {float(movie['vote_average']):.1f}/10")
        st.write(movie["overview"])
        st.write(f"**Genres:** {movie['genres']}")
        st.write(f"**Director:** {movie['director']}")

        if st.button("🔮 Show Similar Movies"):
            results = recommend(movie["title"], 5)
            st.markdown("### 🍿 You may also like")
            for rec, score in results:
                st.write(f"**{rec['title']}** — {score*100:.1f}% similarity")

elif mode == "🎭 By Genre":
    st.markdown("## 🎭 Browse by Genre")
    genre = st.selectbox("Choose a genre", all_genres)

    results = find_by_genre(genre, 8)
    st.caption(f"Showing {len(results)} movies tagged with {genre}.")

    cols = st.columns(4)
    for movie, col in zip(results, cols * 2):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="movie-title">{movie['title']}</div>
                <span class="score">⭐ {float(movie['vote_average']):.1f}</span>
                <p>{movie['overview'][:160]}...</p>
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.caption(
    "CineMatch AI • MCA Machine Learning Project • "
    "Content-Based Recommendation System"
)
