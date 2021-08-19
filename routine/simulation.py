#%% import and definitions
import os
from typing import List

import numba as nb
import numpy as np
import xarray as xr
from numpy import random
from scipy.stats import multivariate_normal

from minian_functions import apply_shifts, save_minian, write_video


def gauss_cell(
    height: int,
    width: int,
    sz_mean: float,
    sz_sigma: float,
    sz_min: float,
    cent=None,
    norm=True,
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
        pdf = multivariate_normal.pdf(grid, mean=c, cov=np.array([[hs, 0], [0, ws]]))
        if norm:
            pmin, pmax = pdf.min(), pdf.max()
            pdf = (pdf - pmin) / (pmax - pmin)
        A[idx] = pdf
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


def random_walk(n_stp, ndim=1, stp=None, p_stp=None, norm=False):
    if p_stp is None:
        p_stp = np.array([1 / 3] * 3)
    if stp is None:
        stp = np.array([-1, 0, 1])
    stps = random.choice(stp, size=(n_stp, ndim), p=p_stp)
    walk = np.cumsum(stps, axis=0)
    if norm:
        walk = (walk - walk.min(axis=0)) / (walk.max(axis=0) - walk.min(axis=0))
    return walk


def simulate_data(
    ncell: int,
    dims: dict,
    sig_scale: float,
    sz_mean: float,
    sz_sigma: float,
    sz_min: float,
    tmp_pfire: float,
    tmp_tau_d: float,
    tmp_tau_r: float,
    bg_nsrc: int = 0,
    mo_stps: List = [0],
    mo_pstp: List = [1],
    cent=None,
):
    ff, hh, ww = (
        dims["frame"],
        dims["height"],
        dims["width"],
    )
    shifts = xr.DataArray(
        random_walk(ff, ndim=2, stp=mo_stps, p_stp=mo_pstp),
        dims=["frame", "shift_dim"],
        coords={"frame": np.arange(ff), "shift_dim": ["height", "width"]},
        name="shifts",
    )
    pad = np.absolute(shifts).max().astype(int).values
    if cent is None:
        cent = np.stack(
            [
                np.random.randint(pad, pad + hh, size=ncell),
                np.random.randint(pad, pad + ww, size=ncell),
            ],
            axis=1,
        )
    A = xr.DataArray(
        gauss_cell(
            2 * pad + hh,
            2 * pad + ww,
            sz_mean=sz_mean,
            sz_sigma=sz_sigma,
            sz_min=sz_min,
            cent=cent,
        ),
        dims=["unit_id", "height", "width"],
        coords={
            "height": np.arange(2 * pad + hh),
            "width": np.arange(2 * pad + ww),
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
    Y = C.dot(A).rename("Y") * sig_scale
    if bg_nsrc > 0:
        cent_bg = np.stack(
            [
                np.random.randint(pad, pad + hh, size=bg_nsrc),
                np.random.randint(pad, pad + ww, size=bg_nsrc),
            ],
            axis=1,
        )
        A_bg = gauss_cell(
            2 * pad + hh,
            2 * pad + ww,
            sz_mean=sz_mean * 100,
            sz_sigma=sz_sigma * 60,
            sz_min=sz_min,
            cent=cent_bg,
        )
        C_bg = random_walk(ff, ndim=bg_nsrc, norm=True)
        Y_bg = xr.DataArray(
            np.tensordot(C_bg, A_bg, axes=1),
            dims=["frame", "height", "width"],
            coords={
                "frame": np.arange(ff),
                "height": np.arange(2 * pad + hh),
                "width": np.arange(2 * pad + ww),
            },
            name="Y_bg",
        )
        Y = Y + Y_bg
    Y = (
        apply_shifts(Y, shifts)
        .compute()
        .isel(height=slice(pad, -pad), width=slice(pad, -pad))
    )
    Y = (Y + random.normal(scale=0.1, size=(ff, hh, ww))).rename("Y")
    return (
        Y,
        A.isel(height=slice(pad, -pad), width=slice(pad, -pad)),
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
        sig_scale=1,
        sz_mean=3,
        sz_sigma=0.6,
        sz_min=0.1,
        tmp_pfire=0.01,
        tmp_tau_d=6,
        tmp_tau_r=1,
        bg_nsrc=100,
        mo_stps=[-2, -1, 0, 1, 2],
        mo_pstp=[0.02, 0.08, 0.8, 0.08, 0.02],
    )
