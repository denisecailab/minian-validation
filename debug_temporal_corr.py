#%% imports and definitions
import os
import re

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import StrMethodFormatter
from statsmodels.formula.api import ols

from routine.minian_functions import open_minian
from routine.plotting import ax_tick, format_tick
from routine.utilities import norm
from routine.validation import compute_metrics
from scipy.ndimage import gaussian_filter1d

hv.notebook_extension("bokeh")

IN_DPATH = "./data/simulated/validation/sig1-cell300"
IN_CAIMAN_RESULT_PAT = "caiman_result.nc"
IN_MINIAN_RESULT_PAT = "minian_result"

#%% load data
minian_ds = open_minian(os.path.join(IN_DPATH, IN_MINIAN_RESULT_PAT))
caiman_ds = xr.open_dataset(os.path.join(IN_DPATH, IN_CAIMAN_RESULT_PAT)).transpose(
    "unit_id", "frame", "height", "width"
)
truth_ds = open_minian(os.path.join(IN_DPATH, "simulated"))
f1_minian, mapping_minian = compute_metrics(minian_ds, truth_ds)
f1_caiman, mapping_caiman = compute_metrics(caiman_ds, truth_ds)

#%% plot single cell
opts_im = {"cmap": "viridis"}
opts_cv = {"frame_width": 800}
# uid = mapping_minian.loc[mapping_minian["Scorr"].argmin()]["uidA"]
uid = 0
mp_minian = mapping_minian[mapping_minian["uidA"] == uid].squeeze()
mp_caiman = mapping_caiman[mapping_caiman["uidA"] == uid].squeeze()
A_minian = minian_ds["A"].sel(unit_id=mp_minian["uidB"])
A_caiman = caiman_ds["A"].sel(unit_id=mp_caiman["uidB"])
A_true = truth_ds["A"].sel(unit_id=uid)
# varr = open_minian(os.path.join(IN_DPATH, "varr"))["varr"]
# ya = norm(A_true.dot(varr, ["height", "width"]).compute())
C_minian = norm(minian_ds["C"].sel(unit_id=mp_minian["uidB"])).compute()
C_caiman = norm(caiman_ds["C"].sel(unit_id=mp_caiman["uidB"])).compute()
C_true = norm(truth_ds["C"].sel(unit_id=uid)).compute()
# back = ya - C_true
S_minian = norm(minian_ds["S"].sel(unit_id=mp_minian["uidB"]))
S_caiman = norm(caiman_ds["S"].sel(unit_id=mp_caiman["uidB"]))
S_true = truth_ds["S"].sel(unit_id=uid)
ds_factor = 10
S_true = S_true.coarsen(frame=ds_factor, boundary="trim").mean().compute()
S_minian = S_minian.coarsen(frame=ds_factor, boundary="trim").mean().compute()
S_caiman = S_caiman.coarsen(frame=ds_factor, boundary="trim").mean().compute()
print(
    "minian corr: {} caiman corr: {}".format(
        np.corrcoef(S_true, S_minian)[0, 1], np.corrcoef(S_true, S_caiman)[0, 1]
    )
)
hvres = (
    hv.Curve(S_true, ["frame"], label="C_true").opts(**opts_cv)
    # * hv.Curve(ya, ["frame"], label="raw").opts(**opts_cv)
    # * hv.Curve(back, ["frame"], label="diff").opts(**opts_cv)
    + hv.Curve(S_minian, ["frame"], label="C_minian").opts(**opts_cv)
    * hv.Curve(S_caiman, ["frame"], label="C_caiman").opts(**opts_cv)
).cols(1)
hv.save(hvres, "temp_corr.html")
hvres
