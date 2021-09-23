from typing import Tuple

import numpy as np
import pandas as pd
import sparse
import xarray as xr
from read_roi import read_roi_zip
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.draw import polygon
from skimage.registration import phase_cross_correlation

from .minian_functions import centroid, shift_perframe


def roi_to_spatial(roi: dict, shape: Tuple, dtype=bool) -> np.ndarray:
    rr, cc = polygon(roi["y"], roi["x"], shape)
    a = np.zeros(shape, dtype=dtype)
    a[rr, cc] = 1
    return a


def convert_rois(roi_path: str, shape: Tuple) -> np.ndarray:
    rois = read_roi_zip(roi_path)
    return np.stack([roi_to_spatial(r, shape) for r in rois.values()])


def compute_mapping(
    centA: pd.DataFrame, centB: pd.DataFrame, dist_thres=None
) -> pd.DataFrame:
    dist = cdist(centA[["height", "width"]], centB[["height", "width"]])
    idxA, idxB = linear_sum_assignment(dist)
    map_df = pd.DataFrame(
        {
            "uidA": centA["unit_id"].iloc[idxA].values,
            "uidB": centB["unit_id"].iloc[idxB].values,
            "dist": dist[idxA, idxB],
        }
    )
    if dist_thres is not None:
        map_df = map_df[map_df["dist"] < dist_thres].copy()
    return map_df


def compute_jac(A1: xr.DataArray, A2: xr.DataArray) -> np.ndarray:
    assert A1.sizes["unit_id"] == A2.sizes["unit_id"]
    A1 = sparse.COO(A1.values > 0)
    A2 = sparse.COO(A2.values > 0)
    inter = (A1 * A2).sum(axis=(1, 2)).todense().astype(float)
    union = (A1 + A2).sum(axis=(1, 2)).todense().astype(float)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union != 0)


def compute_cos(
    x1: xr.DataArray, x2: xr.DataArray, use_sps=False, centered=True
) -> np.ndarray:
    assert x1.dims == x2.dims
    assert x1.sizes["unit_id"] == x2.sizes["unit_id"]
    axes = tuple(np.arange(1, len(x1.dims)))
    x1 = x1.transpose("unit_id", ...).data
    x2 = x2.transpose("unit_id", ...).data
    if use_sps:
        x1 = x1.map_blocks(sparse.COO)
        x1 = x2.map_blocks(sparse.COO)
    if centered:
        x1 = x1 - x1.mean(axis=axes, keepdims=True)
        x2 = x2 - x2.mean(axis=axes, keepdims=True)
    num = (x1 * x2).sum(axis=axes)
    dem = np.sqrt((x1 ** 2).sum(axis=axes) * (x2 ** 2).sum(axis=axes))
    if use_sps:
        num = num.map_blocks(lambda a: a.todense())
        dem = dem.map_blocks(lambda a: a.todense())
    return num / dem


def compute_f1(Nmap, Ntrue, Nobs):
    precision = Nmap / Nobs
    recall = Nmap / Ntrue
    try:
        return 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return 0


def compute_metrics(
    result_ds: xr.Dataset, true_ds: xr.Dataset
) -> Tuple[float, pd.DataFrame]:
    chk_A = {"height": -1, "width": -1, "unit_id": 30}
    chk_S = {"frame": -1, "unit_id": 30}
    if not result_ds.sizes["unit_id"] > 0:
        return 0, pd.DataFrame()
    A = result_ds["A"].compute().chunk(chk_A)
    A_true = true_ds["A"].compute().chunk(chk_A)
    sumim = A.max("unit_id").compute().transpose("height", "width").values
    sumim_true = A_true.max("unit_id").compute().transpose("height", "width").values
    sh, _, _ = phase_cross_correlation(sumim_true, sumim, upsample_factor=100)
    A = (
        xr.apply_ufunc(
            shift_perframe,
            A,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            kwargs={"sh": sh, "fill": 0},
            dask="parallelized",
        )
        .compute()
        .chunk(chk_A)
    )
    cent = centroid(A)
    cent_true = centroid(A_true)
    mapping = compute_mapping(cent_true, cent, 3)
    f1 = compute_f1(len(mapping), len(cent_true), len(cent))
    Am = A.compute().sel(unit_id=mapping["uidB"].values).chunk(chk_A)
    Am_true = A_true.compute().sel(unit_id=mapping["uidA"].values).chunk(chk_A)
    S = result_ds["S"].compute().sel(unit_id=mapping["uidB"].values).chunk(chk_S)
    S_true = (
        true_ds["S"]
        .compute()
        .sel(unit_id=mapping["uidA"].values)
        .transpose("unit_id", "frame")
        .chunk(chk_S)
    )
    mapping["Acorr"] = compute_cos(Am_true, Am)
    mapping["Scorr"] = compute_cos(S, S_true)
    return f1, mapping
