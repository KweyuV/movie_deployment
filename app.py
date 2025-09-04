import streamlit as st
import pandas as pd
import os
from surprise import Dataset, Reader, SVD

st.set_page_config(page_title="Sinema Pamoja Movie Recommendation System", layout="wide")

st.title("Sinema Pamoja Movie Recommendation System")
st.markdown("""
Welcome to the interactive movie recommender for Sinema Pamoja! This app uses the MovieLens dataset and machine learning to suggest movies tailored to your preferences.
""")

# Load data
data_path = "Data/ml-latest-small/"
movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))

# Train SVD model (simple version for demo)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)

# User input
user_id = st.number_input("Enter your User ID:", min_value=int(ratings.userId.min()), max_value=int(ratings.userId.max()), value=int(ratings.userId.min()))

st.write("## Top 5 Recommended Movies For You")

# Recommend top 5 movies not yet rated by user
def recommend_movies(user_id, model, movies, ratings, top_n=5):
    rated_movies = ratings[ratings.userId == user_id].movieId.unique()
    unrated_movies = movies[~movies.movieId.isin(rated_movies)]
    predictions = []
    for movieId in unrated_movies.movieId:
        pred = model.predict(user_id, movieId)
        predictions.append((movieId, pred.est))
    top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]
    return movies[movies.movieId.isin([m[0] for m in top_movies])][["title", "genres"]]

if st.button("Get Recommendations"):
    recs = recommend_movies(user_id, model, movies, ratings)
    st.table(recs)

st.write("---")
st.write("### Data Visualizations")

# Show images if available
image_dir = "images/"
image_files = [
    "distribution_of_ratings.png",
    "ratings_per_user.png",
    "ratings_per_movie.png",
    "movies_by_genre.png",
    "actual_predicted_SVD.png",
    "prediction_error.png",
    "mean_prediction.png",
    "model_comparison.png"
]
for img in image_files:
    img_path = os.path.join(image_dir, img)
    if os.path.exists(img_path):
        st.image(img_path, caption=img.replace("_", " ").replace(".png", "").title(), use_column_width=True)
