import numpy as np
import pandas as pd


def create_utility_matrix(data, formatizer={'user': 0, 'item': 1, 'value': 2}):
    item_fie_ld = formatizer['item']
    user_fie_ld = formatizer['user']
    value_fie_ld = formatizer['value']

    user_list = data.iloc[:, user_fie_ld].tolist()
    item_list = data.iloc[:, item_fie_ld].tolist()
    value_list = data.iloc[:, value_fie_ld].tolist()

    users = list(set(data.iloc[:, user_fie_ld]))
    items = list(set(data.iloc[:, item_fie_ld]))
    users_index = {users[i]: i for i in range(len(users))}
    pd_dict = {item: [np.nan for _ in range(len(users))] for item in items}
    for i in range(0, len(data)):
        item = item_list[i]
        user = user_list[i]
        value = value_list[i]
        pd_dict[item][users_index[user]] = value

    X = pd.DataFrame(pd_dict)
    X.index = users

    item_cols = list(X.columns)
    items_index = {item_cols[i]: i for i in range(len(item_cols))}

    return X, users_index, items_index


def rmse(true, pred):
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)
