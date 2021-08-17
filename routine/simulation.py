import numba as nb
import numpy as np
import xarray as xr
from cv2 import GaussianBlur
from numpy import random

from minian_functions import apply_shifts, save_minian


def gauss_cell(
    height: int, width: int, sigma: float, cov_coef: float, cent=None, nsamp=1000
):
    # generate centroid
    if cent is None:
        cent = (random.randint(height), random.randint(width))
    # generate covariance
    while True:
        cov_var = random.rand(2, 2)
        cov_var = (cov_var + cov_var.T) / 2 * cov_coef
        cov = np.eye(2) * sigma + cov_var
        if np.all(np.linalg.eigvals(cov) > 0):
            break  # ensure cov is positive definite
    # generate samples of coordinates
    crds = np.clip(
        np.round(random.multivariate_normal(cent, cov, size=nsamp)).astype(int),
        0,
        None,
    )
    # generate spatial footprint
    A = np.zeros((height, width))
    for crd in np.unique(crds, axis=0):
        try:
            A[tuple(crd)] = np.sum(np.all(crds == crd, axis=1))
        except IndexError:
            pass
    return A / A.max()


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


def simulate_data(
    ncell: int,
    dims: dict,
    sp_noise: float,
    tmp_noise: float,
    sp_sigma: float,
    sp_cov_coef: float,
    tmp_pfire: float,
    tmp_g_avg: float,
    tmp_g_var: float,
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
        np.stack(
            [
                gauss_cell(hh_pad, ww_pad, sp_sigma, cov_coef=sp_cov_coef, cent=c)
                for c in cent
            ]
        ),
        dims=["unit_id", "height", "width"],
        coords={
            "height": np.arange(hh_pad),
            "width": np.arange(ww_pad),
            "unit_id": np.arange(ncell),
        },
        name="A",
    )
    tmp_g = np.clip(
        random.normal(tmp_g_avg, tmp_g_var, size=ncell), a_min=0.8, a_max=0.95
    )
    traces = [ar_trace(ff, tmp_pfire, np.array([g])) for g in tmp_g]
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
    C_noise = C + random.normal(scale=tmp_noise, size=(ncell, ff))
    Y = C_noise.dot(A).rename("Y")
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


if __name__ == "__main__":
    generate_data(
        dpath="simulated_data",
        ncell=100,
        dims={"height": 100, "width": 100, "frame": 1000},
        sp_noise=0.05,
        tmp_noise=0.08,
        sp_sigma=3,
        sp_cov_coef=2,
        tmp_pfire=0.02,
        tmp_g_avg=0.9,
        tmp_g_var=0.03,
        bg_sigma=20,
        bg_strength=1,
        mo_sigma=1,
    )
