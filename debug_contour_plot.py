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
IN_GT_MAPPING = "gt_mapping.csv"

#%% load data
truth_ds = open_minian(os.path.join(IN_DPATH, "truth"))
minian_ds = open_minian(os.path.join(IN_DPATH, IN_MINIAN_RESULT_PAT))
caiman_ds = xr.open_dataset(os.path.join(IN_DPATH, IN_CAIMAN_RESULT_PAT))
gt_mapping = pd.read_csv(os.path.join(IN_DPATH, IN_GT_MAPPING))

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
im = ax.imshow(minian_ds["max_proj"], cmap="Greys_r")
A_true = truth_ds["A_true"].max("unit_id").compute()
A_dm = (
    truth_ds["A_DM"]
    .sel(
        unit_id=sorted(
            list(set(np.arange(truth_ds.sizes["unit_id"])) - set(gt_mapping["uidA"]))
        )
    )
    .max("unit_id")
    .compute()
)
A_tf = (
    truth_ds["A_TF"]
    .sel(
        unit_id=sorted(
            list(set(np.arange(truth_ds.sizes["unit_id"])) - set(gt_mapping["uidB"]))
        )
    )
    .max("unit_id")
    .compute()
)
A_mm = xr.concat([A_dm, A_tf], "unit_id").max("unit_id").compute()
A_minian = minian_ds["A"].max("unit_id").compute()
A_caiman = caiman_ds["A"].max("unit_id").compute()
contours = {
    "Manual-Consensus": ax.contour(xg, yg, A_true, colors="C2", levels=[0.6]),
    "Manual-Mismatch": ax.contour(
        xg,
        yg,
        A_mm,
        colors="C2",
        levels=[0.6],
        linestyles="dashed",
    ),
    "Minian": ax.contour(
        xg, yg, A_minian, colors="C1", levels=[quantile(A_minian.values, 0.7)]
    ),
    "CaImAn": ax.contour(
        xg, yg, A_caiman, colors="C0", levels=[quantile(A_caiman.values, 0.7)]
    ),
}
hd_ls, lb_ls = [], []
for cname, cnt in contours.items():
    hd, _ = cnt.legend_elements()
    hd_ls.append(hd[0])
    lb_ls.append(cname)
ax.legend(hd_ls, lb_ls, title="Source")
ax.set_xlim(75, 525)
ax.set_ylim(150, 600)
ax.invert_yaxis()
fig.savefig("contour.png")
