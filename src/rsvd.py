from surprise import SVD
from surprise import Reader
import pandas as pd
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed),
data = pd.read_csv('./ratings7kusers.csv')

data['userId'] = data['userId'].astype('str')
data['movieId'] = data['movieId'].astype('str')

users = data['userId'].unique()  # list of all users
movies = data['movieId'].unique()  # list of all movies
test = pd.DataFrame(columns=data.columns)
train = pd.DataFrame(columns=data.columns)

test_ratio = 0.2  # fraction of data to be used as test set.
i = 0
print(len(users))
for u in users:
    i += 1
    temp = data[data['userId'] == u]
    n = len(temp)
    test_size = int(test_ratio * n)

    temp = temp.sort_values('timestamp').reset_index()
    temp.drop('index', axis=1, inplace=True)

    dummy_test = temp.iloc[n - 1 - test_size:]
    dummy_train = temp.iloc[: n - 2 - test_size]

    test = pd.concat([test, dummy_test])
    train = pd.concat([train, dummy_train])
    if i % 1000 == 0:
        print(i)

print("count train and test")
reader = Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data_test = Dataset.load_from_df(test[['userId', 'movieId', 'rating']], reader).build_full_trainset().build_testset()
data_train = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader).build_full_trainset()

RMSE_tune = {}
n_epochs = [5, 7, 10]  # the number of iteration of the SGD procedure
lr_all = [0.001, 0.002, 0.003] # the learning rate for all parameters
reg_all =  [0.1, 0.2, 0.4] # the regularization term for all parameters

uid = "1"
iid = "1097"
uid2 = "11"
iid2 = "71282"
uid3 = "11"
iid3 = "72641"
uid4 = "11"
iid4 = "82095"

for n in n_epochs:
    for l in lr_all:
        for r in reg_all:
            algo = SVD(n_epochs = n, lr_all = l, reg_all = r)
            algo.fit(data_train)
            predictions = algo.test(data_test)
            RMSE_tune[n,l,r] = accuracy.rmse(predictions)
            print(n, l, r)
            pred = algo.predict(uid, iid, r_ui=4, verbose=True)
            pred = algo.predict(uid2, iid2, r_ui=4, verbose=True)
            pred = algo.predict(uid3, iid3, r_ui=4, verbose=True)
            pred = algo.predict(uid4, iid4, r_ui=1, verbose=True)
            print("-----------------")

algo_real = SVD(n_epochs = 10, lr_all = 0.001, reg_all = 0.2)
algo_real.fit(data_train)
predictions = algo_real.test(data_test)