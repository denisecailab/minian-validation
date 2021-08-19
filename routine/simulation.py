#%% import and definitions
import os

import numba as nb
import numpy as np
import xarray as xr
from cv2 import GaussianBlur
from numpy import random
from scipy.stats import multivariate_normal

from minian_functions import apply_shifts, save_minian, write_video


def gauss_cell(
    height: int, width: int, sz_mean: float, sz_sigma: float, sz_min: float, cent=None
):
    # generate centroid
    if cent is None:
        cent = np.atleast_2d([random.randint(height), random.randint(width)])
    # generate size
    sz_h = np.clip(
        random.normal(loc=sz_mean, scale=sz_sigma, size=cent.shape[0]), sz_min, None
    )
    sz_w = np.clip(
        random.normal(loc=sz_mean, scale=sz_sigma, size=cent.shape[0]), sz_min, None
    )
    # generate grid
    grid = np.moveaxis(np.mgrid[:height, :width], 0, -1)
    A = np.zeros((cent.shape[0], height, width))
    for idx, (c, hs, ws) in enumerate(zip(cent, sz_h, sz_w)):
        A[idx] = multivariate_normal.pdf(grid, mean=c, cov=np.array([[hs, 0], [0, ws]]))
    return A


@nb.jit(nopython=True, nogil=True, cache=True)
def apply_arcoef(s: np.ndarray, g: np.ndarray):
    c = np.zeros_like(s)
    for idx in range(len(g), len(s)):
        c[idx] = s[idx] + c[idx - len(g) : idx] @ g
    return c


def ar_trace(frame: int, pfire: float, g: np.ndarray):
    S = random.binomial(n=1, p=pfire, size=frame).astype(float)
    C = apply_arcoef(S, g)
    return C, S


def exp_trace(frame: int, pfire: float, tau_d: float, tau_r: float, trunc_thres=1e-6):
    S = random.binomial(n=1, p=pfire, size=frame).astype(float)
    t = np.arange(frame)
    v = np.exp(-t / tau_d) - np.exp(-t / tau_r)
    v = v[v > trunc_thres]
    C = np.convolve(S, v, mode="same")
    return C, S


def simulate_data(
    ncell: int,
    dims: dict,
    sz_mean: float,
    sz_sigma: float,
    sz_min: float,
    sp_noise: float,
    tmp_pfire: float,
    tmp_tau_d: float,
    tmp_tau_r: float,
    bg_sigma=0,
    bg_strength=0,
    mo_sigma=0,
    cent=None,
):
    ff, hh, ww = (
        dims["frame"],
        dims["height"],
        dims["width"],
    )
    hh_pad, ww_pad = hh + mo_sigma * 4, ww + mo_sigma * 4
    if cent is None:
        cent = np.stack(
            (
                np.random.randint(0, hh_pad, size=ncell),
                np.random.randint(0, ww_pad, size=ncell),
            ),
            axis=1,
        )
    A = xr.DataArray(
        gauss_cell(
            hh_pad, ww_pad, sz_mean=sz_mean, sz_sigma=sz_sigma, sz_min=sz_min, cent=cent
        ),
        dims=["unit_id", "height", "width"],
        coords={
            "height": np.arange(hh_pad),
            "width": np.arange(ww_pad),
            "unit_id": np.arange(ncell),
        },
        name="A",
    )
    traces = [exp_trace(ff, tmp_pfire, tmp_tau_d, tmp_tau_r) for _ in range(len(cent))]
    C = xr.DataArray(
        np.stack([t[0] for t in traces]),
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(ncell), "frame": np.arange(ff)},
        name="C",
    )
    S = xr.DataArray(
        np.stack([t[1] for t in traces]),
        dims=["unit_id", "frame"],
        coords={"unit_id": np.arange(ncell), "frame": np.arange(ff)},
        name="S",
    )
    Y = C.dot(A).rename("Y")
    Y = Y / Y.max()
    if bg_strength:
        A_bg = xr.apply_ufunc(
            GaussianBlur,
            A,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            vectorize=True,
            kwargs={
                "ksize": (int(hh // 2 * 2 - 1), int(ww // 2 * 2 - 1)),
                "sigmaX": bg_sigma,
            },
        )
        Y_bg = C_noise.dot(A_bg)
        Y_bg = Y_bg / Y_bg.max()
        Y = Y + Y_bg * bg_strength
    shifts = xr.DataArray(
        np.clip(
            random.normal(scale=mo_sigma, size=(ff, 2)),
            a_min=-2 * mo_sigma,
            a_max=2 * mo_sigma,
        ),
        dims=["frame", "shift_dim"],
        coords={"frame": np.arange(ff), "shift_dim": ["height", "width"]},
        name="shifts",
    )
    Y = (
        apply_shifts(Y, shifts)
        .compute()
        .isel(
            height=slice(2 * mo_sigma, -2 * mo_sigma),
            width=slice(2 * mo_sigma, -2 * mo_sigma),
        )
    )
    Y = (Y / Y.max() + random.normal(scale=sp_noise, size=(ff, hh, ww))).rename("Y")
    return (
        Y,
        A.isel(
            height=slice(2 * mo_sigma, -2 * mo_sigma),
            width=slice(2 * mo_sigma, -2 * mo_sigma),
        ),
        C,
        S,
        shifts,
    )


def generate_data(dpath, **kwargs):
    Y, A, C, S, shifts = simulate_data(**kwargs)
    Y = (
        ((Y - Y.min()) / (Y.max() - Y.min()) * 255)
        .astype(np.uint8)
        .chunk({"frame": 500})
    )
    for dat in [Y, A, C, S, shifts]:
        save_minian(dat, dpath=dpath, overwrite=True)
    write_video(Y, os.path.join(dpath, "simulation.mp4"))


#%% main
if __name__ == "__main__":
    generate_data(
        dpath="simulated_data",
        ncell=100,
        dims={"height": 256, "width": 256, "frame": 1000},
        sz_mean=3,
        sz_sigma=0.6,
        sz_min=0.1,
        sp_noise=0.05,
        tmp_pfire=0.01,
        tmp_tau_d=6,
        tmp_tau_r=1,
        bg_sigma=20,
        bg_strength=1,
        mo_sigma=1,
    )
