import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_xth_split(x, y, split_num, n_splits=5):

    assert n_splits > split_num >= 0,\
        f'Split num {split_num} is invalid. Must be >= 0, < {n_splits}'

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)

    for i, (train_idx, val_idx) in enumerate(skf.split(x, y)):

        if i == split_num:
            return train_idx, val_idx

        else:
            continue


def bin_paw_train_target(df, bins=10):
    df['bin'] = pd.cut(df['Pawpularity'], bins, labels=False)
    return df
