import numpy as np
import pandas as pd


def append_csv(row: pd.Series, csvfile: str):
    cols = pd.read_csv(csvfile, header=0, nrows=0).columns
    row.reindex(cols).to_frame().T.to_csv(csvfile, mode="a", header=False, index=False)


def norm(a):
    amax = a.max()
    amin = a.min()
    diff = amax - amin
    if diff > 0:
        return (a - amin) / (amax - amin)
    else:
        return a


def quantile(a, q, exclude_zero=True):
    a = a.reshape(-1)
    if exclude_zero:
        a = a[a > 0]
    return np.quantile(a, q)
