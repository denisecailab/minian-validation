from typing import Tuple

import numpy as np
import pandas as pd
from read_roi import read_roi_zip
from skimage.draw import polygon


def append_csv(row: pd.Series, csvfile: str):
    cols = pd.read_csv(csvfile, header=0, nrows=0).columns
    row.reindex(cols).to_frame().T.to_csv(csvfile, mode="a", header=False, index=False)


def roi_to_spatial(roi: dict, shape: Tuple, dtype=bool) -> np.ndarray:
    rr, cc = polygon(roi["x"], roi["y"], shape)
    a = np.zeros(shape, dtype=dtype)
    a[rr, cc] = 1
    return a


def convert_rois(roi_path: str, shape: Tuple) -> np.ndarray:
    rois = read_roi_zip(roi_path)
    return np.stack([roi_to_spatial(r, shape) for r in rois.values()])
