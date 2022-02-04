"""
script to generate figures for validation results

env: environments/generic.yml
"""

#%% imports and definitions
import os
import re

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from matplotlib.transforms import ScaledTranslation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from statsmodels.formula.api import ols

from routine.minian_functions import open_minian
from routine.plotting import ax_tick, format_tick, it_lab
from routine.utilities import norm, quantile
from routine.validation import compute_metrics

IN_SIM_DPATH = "./data/simulated/validation"
IN_REAL_DPATH = "./data/real"
IN_CAIMAN_RESULT_PAT = "caiman_result.nc"
IN_MINIAN_RESULT_PAT = "minian_result"
OUT_PATH = "./store/validation"
FIG_PATH = "./fig/validation/"
IN_GT_MAPPING = "gt_mapping.csv"
IN_CONT_PATH = "./data/real/sao"
IN_EXP_PATH = "./data/real/ferdinand"
OUT_SIM_PATH = "./output/Figure15"
OUT_DV_PATH = "./output/Figure16"
OUT_REAL_PATH = "./output/Figure17"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

#%% compute_metrics on simulated data
f1_ls = []
mapping_ls = []
for root, dirs, files in os.walk(IN_SIM_DPATH):
    try:
        minian_ds = list(filter(lambda f: re.search(IN_MINIAN_RESULT_PAT, f), dirs))[0]
        caiman_ds = list(filter(lambda f: re.search(IN_CAIMAN_RESULT_PAT, f), files))[0]
        truth_ds = list(filter(lambda f: re.search("simulated$", f), dirs))[0]
    except IndexError:
        continue
    minian_ds = open_minian(os.path.join(root, minian_ds))
    caiman_ds = xr.open_dataset(os.path.join(root, caiman_ds)).transpose(
        "unit_id", "frame", "height", "width"
    )
    truth_ds = open_minian(os.path.join(root, truth_ds))
    f1_minian, mapping_minian = compute_metrics(minian_ds, truth_ds, coarsen_factor=5)
    f1_caiman, mapping_caiman = compute_metrics(caiman_ds, truth_ds, coarsen_factor=5)
    sig, ncell = re.search(r"sig([0-9\.]+)-cell([0-9]+)", root).groups()
    mapping_minian["sig"] = sig
    mapping_caiman["sig"] = sig
    mapping_minian["ncell"] = ncell
    mapping_caiman["ncell"] = ncell
    mapping_minian["pipeline"] = "minian"
    mapping_caiman["pipeline"] = "caiman"
    f1_df = pd.DataFrame(
        {
            "pipeline": ["minian", "caiman"],
            "f1": [f1_minian, f1_caiman],
            "sig": sig,
            "ncell": ncell,
        }
    )
    f1_ls.append(f1_df)
    mapping_ls.extend([mapping_caiman, mapping_minian])
    print(root)
f1_df = pd.concat(f1_ls, ignore_index=True)
mapping_df = pd.concat(mapping_ls, ignore_index=True)
f1_df.astype({"ncell": int, "sig": float}).to_feather(
    os.path.join(OUT_PATH, "f1_simulated.feather")
)
mapping_df.astype({"ncell": int, "sig": float}).to_feather(
    os.path.join(OUT_PATH, "mapping_simulated.feather")
)

#%% plot simulated results
ASPECT = 1.2
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIG_SIZE = 11
WIDTH = 7.87  # 20cm
sns.set(
    rc={
        "figure.figsize": (WIDTH, WIDTH / ASPECT),
        "figure.dpi": 500,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
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
    }
)
sns.set_style("ticks")
metrics = ["Acorr", "Scorr", "Ccorr"]
id_vars = ["ncell", "sig", "pipeline"]
metric_dict = {
    "Acorr": "Spatial Correlation",
    "Scorr": "S Correlation",
    "Ccorr": "Temporal Correlation",
    "f1": "F1 Score",
}
pipeline_dict = {"minian": "Minian", "caiman": "CaImAn"}
ylim_dict = {"Acorr": (0, 1.1), "Scorr": (0, 1.1), "f1": (0, 1.1), "Ccorr": (0, 1.1)}


def set_yaxis(data, set_range=False, **kwargs):
    ax = plt.gca()
    var = data.iloc[0]["variable"]
    ax.set_ylabel(metric_dict[var])
    if set_range:
        ax.set_ylim(ylim_dict[var])


mapping_df = pd.read_feather(
    os.path.join(OUT_PATH, "mapping_simulated.feather")
).replace({"pipeline": pipeline_dict})
metric_df = {
    "median": mapping_df.groupby(id_vars)[metrics]
    .median()
    .reset_index()
    .sort_values(["sig", "ncell"]),
    "worst": mapping_df.groupby(id_vars)[metrics]
    .min()
    .reset_index()
    .sort_values(["sig", "ncell"]),
}
f1_df = pd.read_feather(os.path.join(OUT_PATH, "f1_simulated.feather")).replace(
    {"pipeline": pipeline_dict}
)

for mtype, mdf in metric_df.items():
    df = mdf.merge(f1_df, on=id_vars, validate="one_to_one").melt(id_vars=id_vars)
    if mtype == "median":
        os.makedirs(OUT_SIM_PATH, exist_ok=True)
        df.to_csv(os.path.join(OUT_SIM_PATH, "simulated_validation.csv"), index=False)
    fig = sns.FacetGrid(
        df,
        row="variable",
        col="ncell",
        margin_titles=True,
        legend_out=True,
        row_order=["f1", "Acorr", "Ccorr"],
    )
    fig.map_dataframe(
        sns.lineplot,
        x="sig",
        y="value",
        hue="pipeline",
        hue_order=("CaImAn", "Minian"),
        palette={"Minian": "darkblue", "CaImAn": "red", "Manual": "C2"},
        marker="o",
        legend="full",
    )
    if mtype == "median":
        fig.map_dataframe(set_yaxis, set_range=True)
    else:
        fig.map_dataframe(set_yaxis)
    fig.map_dataframe(ax_tick, x_var="sig")
    fig.map(format_tick, y_formatter=StrMethodFormatter("{x:.2f}"))
    fig.map(it_lab)
    fig.add_legend()
    fig.set_xlabels("Signal Level")
    fig.set_titles(row_template="", col_template="{col_name} cells")
    fig.figure.set_size_inches((WIDTH, WIDTH / ASPECT))
    fig.savefig(os.path.join(FIG_PATH, "simulated-{}.svg".format(mtype)))
    fig.savefig(os.path.join(FIG_PATH, "simulated-{}.png".format(mtype)))
    fig.savefig(os.path.join(FIG_PATH, "simulated-{}.tiff".format(mtype)))

#%% plot deconvolved traces
ASPECT = 1.1
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIG_SIZE = 11
WIDTH = 7.87  # 20cm
sns.set(
    rc={
        "figure.figsize": (WIDTH, WIDTH / ASPECT),
        "figure.dpi": 500,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
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
    }
)
sns.set_style("ticks")
corr_ylim = (0, 1.1)
corr_ylab = "Deconvolved Correlation"
corr_xlab = "Signal Level"
layout = [["100", "300", "500"], ["traces", "traces", "traces"]]
offset_pipeline = 0
offset_unit = 1.1
palette = {"Minian": "darkblue", "CaImAn": "red", "Ground Truth": "C2"}
# process data
mapping_df = pd.read_feather(os.path.join(OUT_PATH, "mapping_simulated.feather"))
mapping_df = mapping_df[mapping_df["pipeline"] == "minian"]
mapping_sub_ls = []
for (sig, ncell), subdf in mapping_df.groupby(["sig", "ncell"]):
    if ncell != 100:
        continue
    row = subdf.sort_values("Scorr").iloc[[int(len(subdf) / 2)]]
    mapping_sub_ls.append(row)
mapping_sub = pd.concat(mapping_sub_ls, ignore_index=True)
# plot
fig, axs = plt.subplot_mosaic(
    layout, figsize=(WIDTH, WIDTH / ASPECT), gridspec_kw={"height_ratios": (1, 1.5)}
)
os.makedirs(OUT_DV_PATH, exist_ok=True)
mapping_df.to_csv(os.path.join(OUT_DV_PATH, "deconvolved_correlation.csv"), index=False)
for ncell, subdf in mapping_df.groupby("ncell"):
    cur_ax = axs[str(ncell)]
    sns.lineplot(
        data=subdf,
        x="sig",
        y="Scorr",
        ax=cur_ax,
        marker="o",
        err_style="band",
        ci="sd",
    )
    cur_ax.set_ylim(corr_ylim)
    cur_ax.set_xlabel(corr_xlab)
    cur_ax.set_title("{} cells".format(ncell))
    ax_tick(data=subdf, ax=cur_ax, x_var="sig")
    format_tick(ax=cur_ax, y_formatter=StrMethodFormatter("{x:.2f}"))
    if ncell == 100:
        cur_ax.set_ylabel(corr_ylab)
    else:
        cur_ax.set_ylabel("")
        cur_ax.set_yticklabels([])
    it_lab(ax=cur_ax)
ax_tr = axs["traces"]
for ir, row in mapping_sub.iterrows():
    sig = row["sig"]
    ncell = row["ncell"]
    if sig == int(sig):
        sig = int(sig)
    minian_ds = open_minian(
        os.path.join(
            IN_SIM_DPATH, "sig{}-cell{}".format(sig, ncell), IN_MINIAN_RESULT_PAT
        )
    )
    truth_ds = open_minian(
        os.path.join(IN_SIM_DPATH, "sig{}-cell{}".format(sig, ncell), "simulated")
    )
    trA = minian_ds["S"].sel(unit_id=row["uidB"]).compute()
    trB = truth_ds["S"].sel(unit_id=row["uidA"]).compute()
    trA.to_netcdf(os.path.join(OUT_DV_PATH, "trace_sig{}_minian.nc".format(sig)))
    trB.to_netcdf(os.path.join(OUT_DV_PATH, "trace_sig{}_truth.nc".format(sig)))
    norm_fac = np.quantile(trA[trB > 0], 0.92)
    trA = np.clip(trA / norm_fac, 0, 1) + ir * offset_unit
    trB = trB + ir * offset_unit + offset_pipeline
    (lineB,) = ax_tr.plot(trB, color=palette["Ground Truth"], linewidth=2)
    (lineA,) = ax_tr.plot(trA, color=palette["Minian"], linewidth=1.5)
    if ir == 0:
        lineA.set_label("Minian")
        lineB.set_label("Ground Truth")
ax_tr.set_ylim(-1.3, len(mapping_sub) * offset_unit + 1.6)
ax_tr.set_xlim(-50, 5000)
legs, labs = ax_tr.get_legend_handles_labels()
ax_tr.legend(legs[::-1], labs[::-1], loc="upper right")
ax_tr.set_yticks(np.arange(len(mapping_sub)) * offset_unit + offset_unit / 2)
ax_tr.set_yticklabels(["Signal Level {}".format(s) for s in mapping_sub["sig"]])
ax_tr.get_yaxis().set_tick_params(width=0)
ax_tr.minorticks_off()
ax_tr.get_xaxis().set_visible(False)
ax_tr.spines["left"].set_lw(0)
ax_tr.spines["bottom"].set_lw(0)
szbar = AnchoredSizeBar(
    ax_tr.transData,
    600,
    "10 sec",
    loc="lower right",
    pad=1,
    sep=4,
    size_vertical=0.06,
    frameon=False,
)
ax0 = axs["100"]
ax0.text(
    0,
    1,
    "A",
    transform=ax0.transAxes + ScaledTranslation(-20 / 72, 10 / 72, fig.dpi_scale_trans),
    va="bottom",
    fontweight="bold",
    fontsize="x-large",
)
ax_tr.text(
    0,
    1,
    "B",
    transform=ax_tr.transAxes
    + ScaledTranslation(-20 / 72, 2 / 72, fig.dpi_scale_trans),
    va="bottom",
    fontweight="bold",
    fontsize="x-large",
)
sns.despine()
fig.tight_layout()
ax_tr.add_artist(szbar)
fig.savefig(os.path.join(FIG_PATH, "deconvolved.svg"))
fig.savefig(os.path.join(FIG_PATH, "deconvolved.png"))
fig.savefig(os.path.join(FIG_PATH, "deconvolved.tiff"))

#%% compute metrics on real datasets
f1_ls = []
mapping_ls = []
for root, dirs, files in os.walk(IN_REAL_DPATH):
    try:
        minian_ds = list(filter(lambda f: re.search(IN_MINIAN_RESULT_PAT, f), dirs))[0]
        caiman_ds = list(filter(lambda f: re.search(IN_CAIMAN_RESULT_PAT, f), files))[0]
        truth_ds = list(filter(lambda f: re.search(r"^truth$", f), dirs))[0]
    except IndexError:
        continue
    minian_ds = open_minian(os.path.join(root, minian_ds))
    caiman_ds = xr.open_dataset(os.path.join(root, caiman_ds)).transpose(
        "unit_id", "frame", "height", "width"
    )
    truth_ds = open_minian(os.path.join(root, truth_ds))
    f1_DM = compute_metrics(A=truth_ds["A_DM"], A_true=truth_ds["A_true"], f1_only=True)
    f1_TF = compute_metrics(A=truth_ds["A_TF"], A_true=truth_ds["A_true"], f1_only=True)
    f1_minian, mapping_minian = compute_metrics(
        A=minian_ds["A"],
        A_true=truth_ds["A_true"],
        C=minian_ds["C"],
        C_true=truth_ds["C_true"],
    )
    f1_caiman, mapping_caiman = compute_metrics(
        A=caiman_ds["A"],
        A_true=truth_ds["A_true"],
        C=caiman_ds["C"],
        C_true=truth_ds["C_true"],
    )
    anm = root.split(os.sep)[-1]
    mapping_minian["animal"] = anm
    mapping_caiman["animal"] = anm
    mapping_minian["source"] = "minian"
    mapping_caiman["source"] = "caiman"
    f1_df = pd.DataFrame(
        {
            "source": ["minian", "caiman", "DM", "TF"],
            "f1": [f1_minian, f1_caiman, f1_DM, f1_TF],
            "animal": anm,
        }
    )
    f1_ls.append(f1_df)
    mapping_ls.extend([mapping_caiman, mapping_minian])
    print(root)
f1_df = pd.concat(f1_ls, ignore_index=True)
mapping_df = pd.concat(mapping_ls, ignore_index=True)
f1_df.to_feather(os.path.join(OUT_PATH, "f1_real.feather"))
mapping_df.to_feather(os.path.join(OUT_PATH, "mapping_real.feather"))

#%% plot real results
ASPECT = 0.7
WIDTH = 7.87  # 20cm
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIG_SIZE = 11
sns.set(
    rc={
        "figure.figsize": (WIDTH, WIDTH / ASPECT),
        "figure.dpi": 500,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
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
    }
)
sns.set_style("ticks")
metrics = ["Acorr", "Ccorr"]
id_vars = ["animal", "source"]
metric_dict = {
    "f1": "F1 Score",
    "Acorr": "Spatial Correlation",
    "Ccorr": "Temporal Correlation",
}
layout = [["contour", "contour", "contour"], ["f1", "Acorr", "Ccorr"]]
source_dict = {"minian": "Minian", "caiman": "CaImAn", "DM": "Manual", "TF": "Manual"}
palette = {"Minian": "C0", "CaImAn": "C1", "Manual": "C2"}
ylim_dict = {"Acorr": (0, 1), "Ccorr": (0, 1), "f1": (0, 1)}
# transform data for plotting
mapping_df = pd.read_feather(os.path.join(OUT_PATH, "mapping_real.feather")).replace(
    {"source": source_dict}
)
f1_df = pd.read_feather(os.path.join(OUT_PATH, "f1_real.feather")).replace(
    {"source": source_dict}
)
truth_ds = open_minian(os.path.join(IN_CONT_PATH, "truth"))
minian_ds = open_minian(os.path.join(IN_CONT_PATH, IN_MINIAN_RESULT_PAT))
caiman_ds = xr.open_dataset(os.path.join(IN_CONT_PATH, IN_CAIMAN_RESULT_PAT))
gt_mapping = pd.read_csv(os.path.join(IN_CONT_PATH, IN_GT_MAPPING))
metric_df = mapping_df.groupby(id_vars)[metrics].median().reset_index()
data_df = f1_df.merge(metric_df, on=id_vars, how="left")
xg, yg = np.arange(truth_ds.sizes["width"]), np.arange(truth_ds.sizes["height"])
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
# plot
fig, axs = plt.subplot_mosaic(
    layout,
    figsize=(WIDTH, WIDTH / ASPECT),
    gridspec_kw={"width_ratios": (3, 2, 2), "height_ratios": (2.6, 1)},
)
ax_cnt = axs["contour"]
im = ax_cnt.imshow(minian_ds["max_proj"], cmap="Greys_r")
contours = {
    "Manual-Consensus": ax_cnt.contour(
        xg, yg, A_true, colors=palette["Manual"], levels=[0.6]
    ),
    "Manual-Mismatch": ax_cnt.contour(
        xg,
        yg,
        A_mm,
        colors=palette["Manual"],
        levels=[0.6],
        linestyles="dashed",
    ),
    "Minian": ax_cnt.contour(
        xg,
        yg,
        A_minian,
        colors=palette["Minian"],
        levels=[quantile(A_minian.values, 0.7)],
    ),
    "CaImAn": ax_cnt.contour(
        xg,
        yg,
        A_caiman,
        colors=palette["CaImAn"],
        levels=[quantile(A_caiman.values, 0.7)],
    ),
}
hd_ls, lb_ls = [], []
for cname, cnt in contours.items():
    hd, _ = cnt.legend_elements()
    hd_ls.append(hd[0])
    lb_ls.append(cname)
ax_cnt.legend(hd_ls, lb_ls, title="Source")
ax_cnt.set_xlim(75, 525)
ax_cnt.set_ylim(150, 600)
ax_cnt.invert_yaxis()
ax_cnt.set_axis_off()
for var, vname in metric_dict.items():
    cur_ax = axs[var]
    sns.barplot(
        data=data_df[data_df[var].notnull()],
        x="source",
        y=var,
        capsize=0.2,
        dodge=False,
        hue="source",
        hue_order=["Minian", "CaImAn", "Manual"],
        palette=palette,
        ax=cur_ax,
    )
    sns.swarmplot(
        data=data_df[data_df[var].notnull()],
        x="source",
        y=var,
        size=6,
        alpha=0.8,
        hue="source",
        hue_order=["Minian", "CaImAn", "Manual"],
        palette=palette,
        edgecolor="gray",
        linewidth=2,
        ax=cur_ax,
    )
    cur_ax.set_xlabel("Source")
    cur_ax.set_ylabel(vname)
    cur_ax.set_ylim(ylim_dict[var])
    cur_ax.get_legend().remove()
axs["contour"].text(
    0,
    1,
    "A",
    transform=axs["contour"].transAxes
    + ScaledTranslation(-20 / 72, 10 / 72, fig.dpi_scale_trans),
    va="bottom",
    fontweight="bold",
    fontsize="x-large",
)
axs["f1"].text(
    0,
    1,
    "B",
    transform=axs["f1"].transAxes
    + ScaledTranslation(-40 / 72, 10 / 72, fig.dpi_scale_trans),
    va="bottom",
    fontweight="bold",
    fontsize="x-large",
)
sns.despine()
fig.tight_layout(h_pad=1, w_pad=2)
fig.savefig(os.path.join(FIG_PATH, "real.svg"))
fig.savefig(os.path.join(FIG_PATH, "real.png"))

#%% stats on real validation
metrics = ["Acorr", "Ccorr"]
id_vars = ["animal", "source"]
source_dict = {"minian": "Minian", "caiman": "CaImAn", "DM": "Manual", "TF": "Manual"}
mapping_df = pd.read_feather(os.path.join(OUT_PATH, "mapping_real.feather")).replace(
    {"source": source_dict}
)
metric_df = mapping_df.groupby(id_vars)[metrics].median().reset_index()
f1_df = pd.read_feather(os.path.join(OUT_PATH, "f1_real.feather")).replace(
    {"source": source_dict}
)
lm_f1 = ols("f1 ~ source", f1_df).fit()
lm_A = ols("Acorr ~ source", metric_df).fit()
lm_S = ols("Ccorr ~ source", metric_df).fit()
print("F1 score")
print(lm_f1.summary())
print("Spatial correlation")
print(lm_A.summary())
print("Temporal correlation")
print(lm_S.summary())

#%% compute metric on pipeline comparison
f1_ls = []
mapping_ls = []
for root, dirs, files in os.walk(IN_REAL_DPATH):
    try:
        minian_ds = list(filter(lambda f: re.search(IN_MINIAN_RESULT_PAT, f), dirs))[0]
        caiman_ds = list(filter(lambda f: re.search(IN_CAIMAN_RESULT_PAT, f), files))[0]
    except IndexError:
        continue
    minian_ds = open_minian(os.path.join(root, minian_ds))
    caiman_ds = xr.open_dataset(os.path.join(root, caiman_ds)).transpose(
        "unit_id", "frame", "height", "width"
    )
    f1, mapping = compute_metrics(
        result_ds=caiman_ds, true_ds=minian_ds, dist_thres=15, register=False
    )
    anm = root.split(os.sep)[-1]
    mapping["animal"] = anm
    f1 = pd.Series({"f1": f1, "animal": anm})
    f1_ls.append(f1)
    mapping_ls.append(mapping)
    print(root)
f1_df = pd.concat(f1_ls, axis="columns").T
mapping_df = pd.concat(mapping_ls, ignore_index=True)
f1_df.to_feather(os.path.join(OUT_PATH, "f1_pipeline.feather"))
mapping_df.to_feather(os.path.join(OUT_PATH, "mapping_pipeline.feather"))

#%% plot pipeline comparison
ASPECT = 0.8
WIDTH = 5.51  # 14cm
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIG_SIZE = 11
sns.set(
    rc={
        "figure.figsize": (WIDTH, WIDTH / ASPECT),
        "figure.dpi": 500,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
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
        "contour.linewidth": 1,
    }
)
sns.set_style("ticks")
metrics = ["Acorr", "Ccorr"]
id_vars = ["animal", "source"]
metric_dict = {
    "f1": "F1 Score",
    "Acorr": "Spatial Correlation",
    "Ccorr": "Temporal Correlation",
}
layout = [["image"], ["traces"]]
source_dict = {"minian": "Minian", "caiman": "CaImAn", "DM": "Manual", "TF": "Manual"}
palette = {"Minian": "darkblue", "CaImAn": "red", "Manual": "C2"}
ylim_dict = {"Acorr": (0, 1), "Ccorr": (0, 1), "f1": (0, 1)}
max_proj_range = (0, 20)
offset_pipeline = 0
offset_unit = 1.2
nunits = 5
brt_offset = 0
q_clip = 0.98
# transform data for plotting
mapping_df = pd.read_feather(
    os.path.join(OUT_PATH, "mapping_pipeline.feather")
).replace({"source": source_dict})
f1_df = pd.read_feather(os.path.join(OUT_PATH, "f1_pipeline.feather")).replace(
    {"source": source_dict}
)
minian_ds = open_minian(os.path.join(IN_EXP_PATH, IN_MINIAN_RESULT_PAT))
caiman_ds = xr.open_dataset(os.path.join(IN_EXP_PATH, IN_CAIMAN_RESULT_PAT))
max_proj = np.clip(
    minian_ds["max_proj"].compute(), a_min=max_proj_range[0], a_max=max_proj_range[1]
)
A_minian = minian_ds["A"].max("unit_id").compute()
A_caiman = caiman_ds["A"].max("unit_id").compute()
A_minian = A_minian.clip(0, A_minian.quantile(q_clip))
A_caiman = A_caiman.clip(0, A_caiman.quantile(q_clip))
xg, yg = np.arange(minian_ds.sizes["width"]), np.arange(minian_ds.sizes["height"])
anm = IN_EXP_PATH.split(os.sep)[-1]
mapping_sub = (
    mapping_df[mapping_df["animal"] == anm]
    .sort_values("Ccorr", ascending=False)[:nunits]
    .copy()
    .reset_index()
)
mapping_agg = mapping_df.groupby("animal").median()
print("F1: {}, sem: {}".format(f1_df["f1"].mean(), f1_df["f1"].sem()))
print(
    "A corr: {}, sem: {}".format(
        mapping_agg["Acorr"].mean(), mapping_agg["Acorr"].sem()
    )
)
print(
    "C corr: {}, sem: {}".format(
        mapping_agg["Ccorr"].mean(), mapping_agg["Ccorr"].sem()
    )
)
# plot
fig, axs = plt.subplot_mosaic(
    layout, figsize=(WIDTH, WIDTH / ASPECT), gridspec_kw={"height_ratios": (1.6, 1)}
)
ax_im = axs["image"]
os.makedirs(OUT_REAL_PATH, exist_ok=True)
A_minian.to_netcdf(os.path.join(OUT_REAL_PATH, "spatial_footprint_minian.nc"))
A_caiman.to_netcdf(os.path.join(OUT_REAL_PATH, "spatial_footprint_caiman.nc"))
im_minian = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_blue_0_44_c57).to_rgba(A_minian.values)
    + brt_offset,
    0,
    1,
)
im_caiman = np.clip(
    cm.ScalarMappable(cmap=cc.m_linear_ternary_red_0_50_c52).to_rgba(A_caiman.values)
    + brt_offset,
    0,
    1,
)
im_ovly = np.clip(im_minian + im_caiman, 0, 1)
hsv_minian = rgb_to_hsv(im_minian[:, :, :-1])
hsv_caiman = rgb_to_hsv(im_caiman[:, :, :-1])
hsv_ovly = rgb_to_hsv(im_ovly[:, :, :-1])
hminian = hsv_minian[:, :, 0]
hcaiman = hsv_caiman[:, :, 0]
hovly = hsv_ovly[:, :, 0]
hcaiman_sh = hcaiman + 1
hminian_sh = np.where(hminian < 0.5, hminian + 1, hminian)
hovly_sh = np.where(hovly < 0.5, hovly + 1, hovly)
# hsv_ovly[:, :, 0] = np.where(
#     hovly > hminian,
#     hminian - ((hovly - hminian) / (1 + hcaiman - hminian) * (hminian - hcaiman)),
#     hcaiman + ((hcaiman - hovly) / (1 + hcaiman - hminian) * (hminian - hcaiman)),
# )
hdist = np.min(
    np.stack([np.abs(hovly_sh - hminian_sh), np.abs(hovly_sh - hcaiman_sh)]),
    axis=0,
)
hsv_ovly[:, :, 1] = np.clip(hsv_ovly[:, :, 1] * (1 - hdist / 0.4), 0, 1)
legends = [
    Line2D(
        [0],
        [0],
        marker="o",
        linewidth=0,
        label="Minian",
        markerfacecolor="darkblue",
        mew=0,
        markersize=10,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        linewidth=0,
        label="CaImAn",
        markerfacecolor="red",
        mew=0,
        markersize=10,
    ),
]
ax_im.imshow(hsv_to_rgb(hsv_ovly))
ax_im.set_xlim(0, 550)
ax_im.set_ylim(100, 600)
ax_im.set_axis_off()
ax_im.invert_yaxis()
ax_im.legend(handles=legends, facecolor="dimgray", labelcolor="white")
ax_tr = axs["traces"]
for ir, row in mapping_sub.iterrows():
    trA = norm(minian_ds["C"].sel(unit_id=row["uidA"]))
    trB = norm(caiman_ds["C"].sel(unit_id=row["uidB"]))
    trA.to_netcdf(os.path.join(OUT_REAL_PATH, "trace_cell{}_minian.nc".format(ir + 1)))
    trB.to_netcdf(os.path.join(OUT_REAL_PATH, "trace_cell{}_caiman.nc".format(ir + 1)))
    trA = trA + ir * offset_unit
    trB = trB + ir * offset_unit + offset_pipeline
    (lineB,) = ax_tr.plot(trB, color=palette["CaImAn"], linewidth=2)
    (lineA,) = ax_tr.plot(trA, color=palette["Minian"], linewidth=1.5)
    if ir == 0:
        lineA.set_label("Minian")
        lineB.set_label("CaImAn")
szbar = AnchoredSizeBar(
    ax_tr.transData,
    900,
    "30 sec",
    loc="lower right",
    pad=0,
    sep=4,
    size_vertical=0.06,
    frameon=False,
)
ax_tr.set_ylim(-1.2, nunits * offset_unit + 1.4)
legs, labs = ax_tr.get_legend_handles_labels()
ax_tr.legend(legs[::-1], labs[::-1], loc="upper right")
sns.despine(ax=ax_tr)
# ax_tr.set_axis_off()
ax_tr.set_yticks(np.arange(nunits) * offset_unit + offset_unit / 4)
ax_tr.set_yticklabels(["Cell {}".format(i) for i in range(nunits, 0, -1)])
ax_tr.get_yaxis().set_tick_params(width=0)
ax_tr.minorticks_off()
ax_tr.get_xaxis().set_visible(False)
ax_tr.spines["left"].set_lw(0)
ax_tr.spines["bottom"].set_lw(0)
ax_im.text(
    0,
    1,
    "A",
    transform=ax_im.transAxes
    + ScaledTranslation(-20 / 72, 10 / 72, fig.dpi_scale_trans),
    va="bottom",
    fontweight="bold",
    fontsize="x-large",
)
ax_tr.text(
    0,
    1,
    "B",
    transform=ax_tr.transAxes
    + ScaledTranslation(-10 / 72, 2 / 72, fig.dpi_scale_trans),
    va="bottom",
    fontweight="bold",
    fontsize="x-large",
)
fig.tight_layout(h_pad=1)
ax_tr.add_artist(szbar)
fig.savefig(os.path.join(FIG_PATH, "pipeline.svg"))
fig.savefig(os.path.join(FIG_PATH, "pipeline.png"))
fig.savefig(os.path.join(FIG_PATH, "pipeline.tiff"))
