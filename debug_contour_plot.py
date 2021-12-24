#%% imports
import os

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import pyplot as plt

from routine.minian_functions import open_minian
from routine.utilities import quantile

IN_DPATH = "./data/real/sao"
IN_CAIMAN_RESULT_PAT = "caiman_result.nc"
IN_MINIAN_RESULT_PAT = "minian_result"

#%% load data
truth_ds = open_minian(os.path.join(IN_DPATH, "truth"))
minian_ds = open_minian(os.path.join(IN_DPATH, IN_MINIAN_RESULT_PAT))
caiman_ds = xr.open_dataset(os.path.join(IN_DPATH, IN_CAIMAN_RESULT_PAT))

#%% contour plot
ASPECT = 2.5
WIDTH = 7.87  # 20cm
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIG_SIZE = 11
sns.set(
    rc={
        "figure.figsize": (WIDTH, WIDTH / ASPECT),
        "figure.dpi": 500,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": MEDIUM_SIZE,
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": MEDIUM_SIZE,  # size of faceting titles
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": MEDIUM_SIZE,
        "figure.titlesize": BIG_SIZE,
        "legend.edgecolor": "gray",
        # "axes.linewidth": 0.4,
        # "axes.facecolor": "white",
        "xtick.major.size": 2,
        "xtick.major.width": 0.4,
        "xtick.minor.visible": True,
        "xtick.minor.size": 1,
        "xtick.minor.width": 0.4,
        "ytick.major.size": 2,
        "ytick.major.width": 0.4,
        "ytick.minor.visible": True,
        "ytick.minor.size": 1,
        "ytick.minor.width": 0.4,
        "contour.linewidth": 0.6,
    }
)
sns.set_style("ticks")
fig, ax = plt.subplots(figsize=(WIDTH, WIDTH))
ax.set_axis_off()
xg, yg = np.arange(truth_ds.sizes["width"]), np.arange(truth_ds.sizes["height"])
im = ax.imshow(minian_ds["max_proj"], cmap="viridis")
A_DM = truth_ds["A_DM"].max("unit_id").compute()
A_TF = truth_ds["A_TF"].max("unit_id").compute()
A_minian = minian_ds["A"].max("unit_id").compute()
A_caiman = caiman_ds["A"].max("unit_id").compute()
contours = {
    "DM": ax.contour(xg, yg, A_DM, colors="white", levels=[quantile(A_DM.values, 0.9)]),
    "TF": ax.contour(xg, yg, A_TF, colors="red", levels=[quantile(A_TF.values, 0.9)]),
    "Minian": ax.contour(
        xg,
        yg,
        A_minian,
        colors="white",
        levels=[quantile(A_minian.values, 0.7)],
        linestyles="dashed",
    ),
    "CaImAn": ax.contour(
        xg,
        yg,
        A_caiman,
        colors="red",
        levels=[quantile(A_caiman.values, 0.7)],
        linestyles="dashed",
    ),
}
hd_ls, lb_ls = [], []
for cname, cnt in contours.items():
    hd, _ = cnt.legend_elements()
    hd_ls.append(hd[0])
    lb_ls.append(cname)
ax.legend(hd_ls, lb_ls)
ax.set_xlim(50, 450)
ax.set_ylim(200, 600)
fig.savefig("contour.png")
