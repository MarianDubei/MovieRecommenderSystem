import numpy as np
import pandas as pd
from utility import create_utility_matrix, rmse
from svd import svd


data = pd.read_csv('../data/ratings7kusers.csv')

data['userId'] = data['userId'].astype('str')
data['movieId'] = data['movieId'].astype('str')

users = data['userId'].unique()
movies = data['movieId'].unique()
dataset = pd.DataFrame(data=data)
utilMat, users_index, items_index = create_utility_matrix(dataset)
svd_out = svd(utilMat, k=10)

while True:
    user = input("Enter user ID: ")
    u_index = users_index[user]
    films_ratings = utilMat.loc[user].copy()
    unwatched_films_ratings = films_ratings[films_ratings.isnull()]
    unwatched_films = list(unwatched_films_ratings.index.values)
    unwatched_films_predicts = []
    for item in unwatched_films:
        if item in items_index:
            i_index = items_index[item]
            unwatched_films_predicts.append([item, svd_out[u_index, i_index]])
        else:
            unwatched_films_predicts.append([item, np.mean(svd_out[u_index, :])])
    unwatched_films_predicts.sort(key=lambda x: x[1], reverse=True)
    print(unwatched_films_predicts)
