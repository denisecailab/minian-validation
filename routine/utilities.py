import pandas as pd


def append_csv(row: pd.Series, csvfile: str):
    cols = pd.read_csv(csvfile, header=0, nrows=0).columns
    row.reindex(cols).to_frame().T.to_csv(csvfile, mode="a", header=False, index=False)
