import numpy as np
import pandas as pd


def get_movielens():
    PATH = "~/develop/dataset/movielens/ml-100k/u.data"
    data = pd.read_csv(
        PATH,
        names=["user_id", "item_id", "rating", "timestamp"],
        header=None,
        delimiter="\t",
    )
    np.random.seed(3)
    msk = np.random.rand(len(data)) < 0.8
    df_train = data[msk].copy()
    df_val = data[~msk].copy()

    return df_train, df_val
