import numpy as np
import pandas as pd
from utility import create_utility_matrix, rmse
from svd import svd


data = pd.read_csv('../data/ratings600u.csv')

data['userId'] = data['userId'].astype('str')
data['movieId'] = data['movieId'].astype('str')

users = data['userId'].unique()
movies = data['movieId'].unique()
test = pd.DataFrame(columns=data.columns)
train = pd.DataFrame(columns=data.columns)

test_ratio = 0.2
for u in users:
    temp = data[data['userId'] == u]
    n = len(temp)
    test_size = int(test_ratio * n)

    temp = temp.sort_values('timestamp').reset_index()
    temp.drop('index', axis=1, inplace=True)

    dummy_test = temp.iloc[n - 1 - test_size:]
    dummy_train = temp.iloc[: n - 2 - test_size]

    test = pd.concat([test, dummy_test])
    train = pd.concat([train, dummy_train])

no_of_features = [8, 10, 12, 14, 17]
utilMat, users_index, items_index = create_utility_matrix(train)


for f in no_of_features:
    svd_out = svd(utilMat, k=f)

    pred = []
    for _, row in test.iterrows():
        user = row['userId']
        item = row['movieId']
        u_index = users_index[user]
        if item in items_index:
            i_index = items_index[item]
            pred_rating = svd_out[u_index, i_index]
        else:
            pred_rating = np.mean(svd_out[u_index, :])
        pred.append(pred_rating)

    print(rmse(test['rating'], pred))


