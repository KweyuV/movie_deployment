import streamlit as st
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD

# Load data
movies = pd.read_csv('Data/ml-latest-small/movies.csv')
ratings = pd.read_csv('Data/ml-latest-small/ratings.csv')

# Prepare Surprise data
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train SVD model
svd = SVD(n_factors=150, n_epochs=30, lr_all=0.005, reg_all=0.1)
svd.fit(trainset)

# Streamlit UI
st.title('Movie Recommendation System')
user_id = st.number_input('Enter User ID:', min_value=int(ratings['userId'].min()), max_value=int(ratings['userId'].max()), value=int(ratings['userId'].min()))
n_recs = st.slider('Number of recommendations:', 1, 10, 5)

if st.button('Get Recommendations'):
    all_movie_ids = ratings['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].values
    movies_to_predict = np.setdiff1d(all_movie_ids, rated_movies)
    testset = [[user_id, movie_id, 4.] for movie_id in movies_to_predict]
    predictions = svd.test(testset)
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n_recs]
    recs = [(movies[movies['movieId'] == int(pred.iid)]['title'].values[0], pred.est) for pred in top_n]
    st.write('Top Recommendations:')
    for i, (title, rating) in enumerate(recs, 1):
        st.write(f"{i}. {title} (Predicted rating: {rating:.2f})")
