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
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from statsmodels.formula.api import ols

from routine.minian_functions import centroid, open_minian
from routine.plotting import ax_tick, format_tick
from routine.utilities import norm
from routine.validation import compute_metrics

hv.notebook_extension("bokeh")

IN_SIM_DPATH = "./data/simulated/validation/sig1-cell500"
IN_REAL_DPATH = "./data/real/ferdinand"
IN_CAIMAN_RESULT_PAT = "caiman_result.nc"
IN_MINIAN_RESULT_PAT = "minian_result"
IN_SIM_MAP = "./store/validation/mapping_simulated.feather"
IN_REAL_MAP = "./store/validation/mapping_real.feather"

#%% load data
minian_ds_real = open_minian(os.path.join(IN_REAL_DPATH, IN_MINIAN_RESULT_PAT))
caiman_ds_real = xr.open_dataset(
    os.path.join(IN_REAL_DPATH, IN_CAIMAN_RESULT_PAT)
).transpose("unit_id", "frame", "height", "width")
truth_ds_real = open_minian(os.path.join(IN_REAL_DPATH, "truth")).rename(
    A_true="A", C_true="C"
)
minian_ds_sim = open_minian(os.path.join(IN_SIM_DPATH, IN_MINIAN_RESULT_PAT))
caiman_ds_sim = xr.open_dataset(
    os.path.join(IN_SIM_DPATH, IN_CAIMAN_RESULT_PAT)
).transpose("unit_id", "frame", "height", "width")
truth_ds_sim = open_minian(os.path.join(IN_SIM_DPATH, "simulated"))

#%% compute centroids
def pw_corr(ds):
    cent = centroid(ds["A"].dropna("unit_id", how="all"))
    dist_cent = pairwise_distances(
        cent[["height", "width"]], metric="euclidean", n_jobs=-1
    )

    dist_C = pairwise_distances(
        ds["C"].transpose("unit_id", "frame").dropna("unit_id", how="all"),
        metric="correlation",
        n_jobs=-1,
    )
    try:
        dist_S = pairwise_distances(
            ds["S"].transpose("unit_id", "frame").dropna("unit_id", how="all"),
            metric="correlation",
            n_jobs=-1,
        )
    except KeyError:
        dist_S = np.full_like(dist_C, np.nan)
    return pd.DataFrame(
        {
            "dist_centroid": squareform(dist_cent, checks=False),
            "corr_S": 1 - squareform(dist_S, checks=False),
            "corr_C": 1 - squareform(dist_C, checks=False),
        }
    )


minian_corr = pw_corr(minian_ds_real)
caiman_corr = pw_corr(caiman_ds_real)
truth_corr = pw_corr(truth_ds_real)

#%% plot correlations
# sns.kdeplot(
#     x="dist_centroid",
#     y="corr_C",
#     data=truth_corr,
#     fill=True,
#     levels=100,
#     cmap="viridis",
#     thresh=0,
# )
minian_corr["source"] = "minian"
caiman_corr["source"] = "caiman"
truth_corr["source"] = "manual"
data_df = pd.concat([minian_corr, caiman_corr, truth_corr])
sns.set(rc={"figure.dpi": 500})
g = sns.FacetGrid(
    data=data_df, col="source", sharex=True, sharey=True, xlim=(0, 50), ylim=(-0.5, 1)
)
g.map_dataframe(sns.scatterplot, x="dist_centroid", y="corr_C", s=12)
g.savefig("dist_corr.png")
