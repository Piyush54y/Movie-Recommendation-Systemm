import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = Path("data/movies.csv")
MODEL_FILE = Path("models/movie_recommender.pkl")

df = pd.read_csv(DATA_FILE)

required = [
    "title", "genres", "overview", "keywords",
    "cast", "director", "vote_average", "vote_count"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

text_columns = ["genres", "overview", "keywords", "cast", "director"]

for col in text_columns:
    df[col] = df[col].fillna("").astype(str)

# Give structured movie information a combined text representation.
df["tags"] = (
    df["genres"] + " " +
    df["keywords"] + " " +
    df["cast"] + " " +
    df["director"] + " " +
    df["overview"]
).str.lower()

df["title"] = df["title"].fillna("").astype(str)
df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

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

model = {
    "movies": df.to_dict("records"),
    "similarity": similarity,
    "title_to_index": title_to_index,
    "vectorizer": vectorizer,
}

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"Model trained successfully.")
print(f"Movies: {len(df)}")
print(f"Feature matrix: {vectors.shape}")
print(f"Saved to: {MODEL_FILE}")
