# CineMatch AI — Movie Recommendation System

## Overview
A content-based movie recommendation system built with Python, Scikit-learn and Streamlit.

## Machine Learning
1. Movie metadata is combined into a text feature.
2. TF-IDF converts the text into numerical vectors.
3. Cosine similarity measures similarity between movies.
4. The system returns the Top-N most similar movies.

## Dataset
Place the dataset at:

`data/movies.csv`

Required columns:

- title
- genres
- overview
- keywords
- cast
- director
- vote_average
- vote_count

## Installation

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This creates:

`models/movie_recommender.pkl`

## Run the application

```bash
streamlit run app.py
```

## Project Flow

Dataset → Preprocessing → Feature Engineering → TF-IDF → Cosine Similarity → Recommendations → Streamlit UI
