"""
script to generate figures for validation results

env: environments/generic.yml
"""

#%% imports and definitions
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.ticker import StrMethodFormatter
from statsmodels.formula.api import ols

from routine.minian_functions import open_minian
from routine.validation import compute_metrics
from routine.plotting import ax_tick, format_tick

IN_SIM_DPATH = "./data/simulated/validation"
IN_REAL_DPATH = "./data/real"
IN_CAIMAN_RESULT_PAT = "caiman_result.nc"
IN_MINIAN_RESULT_PAT = "minian_result"
OUT_PATH = "./store/validation"
FIG_PATH = "./fig/validation/"

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
    f1_minian, mapping_minian = compute_metrics(minian_ds, truth_ds)
    f1_caiman, mapping_caiman = compute_metrics(caiman_ds, truth_ds)
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
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIG_SIZE = 11
WIDTH = 7.87  # 20cm
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
    }
)
sns.set_style("ticks")
metrics = ["Acorr", "Scorr"]
id_vars = ["ncell", "sig", "pipeline"]
metric_dict = {
    "Acorr": "Spatial Correlation",
    "Scorr": "Temporal Correlation",
    "f1": "F1 Score",
}
pipeline_dict = {"minian": "Minian", "caiman": "CaImAn"}
ylim_dict = {"Acorr": (0, 1.1), "Scorr": (0, 1.1), "f1": (0, 1.1)}


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
    fig = sns.FacetGrid(
        df,
        row="variable",
        col="ncell",
        margin_titles=True,
        legend_out=True,
        row_order=["f1", "Acorr", "Scorr"],
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
    fig.add_legend()
    fig.set_xlabels("Signal Level")
    fig.set_titles(row_template="", col_template="{col_name} cells")
    fig.figure.set_size_inches((WIDTH, WIDTH / ASPECT))
    fig.savefig(os.path.join(FIG_PATH, "simulated-{}.svg".format(mtype)))
    fig.savefig(os.path.join(FIG_PATH, "simulated-{}.png".format(mtype)))

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
        S=minian_ds["C"].sel(frame=slice(0, 9999)),
        S_true=truth_ds["C_true"].sel(frame=slice(0, 9999)),
    )
    f1_caiman, mapping_caiman = compute_metrics(
        A=caiman_ds["A"],
        A_true=truth_ds["A_true"],
        S=caiman_ds["C"].sel(frame=slice(0, 9999)),
        S_true=truth_ds["C_true"].sel(frame=slice(0, 9999)),
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
    }
)
sns.set_style("ticks")
metrics = ["Acorr", "Scorr"]
id_vars = ["animal", "source"]
metric_dict = {
    "Acorr": "Spatial Correlation",
    "Scorr": "Temporal Correlation",
    "f1": "F1 Score",
}
source_dict = {"minian": "Minian", "caiman": "CaImAn", "DM": "Manual", "TF": "Manual"}
ylim_dict = {"Acorr": (0.6, 1), "Scorr": (0.6, 1), "f1": (0.1, 1)}


def set_yaxis(data, set_range=False, **kwargs):
    ax = plt.gca()
    var = data.iloc[0]["variable"]
    ax.set_ylabel(metric_dict[var])
    if set_range:
        ax.set_ylim(ylim_dict[var])


mapping_df = pd.read_feather(os.path.join(OUT_PATH, "mapping_real.feather")).replace(
    {"source": source_dict}
)
metric_df = {
    "median": mapping_df.groupby(id_vars)[metrics].median().reset_index(),
    "worst": mapping_df.groupby(id_vars)[metrics].min().reset_index(),
}
f1_df = pd.read_feather(os.path.join(OUT_PATH, "f1_real.feather")).replace(
    {"source": source_dict}
)
for mtype, mdf in metric_df.items():
    df = f1_df.merge(mdf, on=id_vars, how="left").melt(id_vars=id_vars).dropna()
    fig = sns.FacetGrid(
        df,
        col="variable",
        legend_out=True,
        col_order=["f1", "Acorr", "Scorr"],
        sharey=False,
        sharex=False,
        gridspec_kws={"width_ratios": (3, 2, 2)},
    )
    fig.map_dataframe(
        sns.barplot,
        x="source",
        y="value",
        capsize=0.2,
        hue="source",
        hue_order=["Minian", "CaImAn", "Manual"],
        dodge=False,
        palette={"Minian": "darkblue", "CaImAn": "red", "Manual": "C2"},
    )
    fig.map_dataframe(
        sns.swarmplot,
        x="source",
        y="value",
        size=6,
        alpha=0.8,
        hue="source",
        hue_order=["Minian", "CaImAn", "Manual"],
        palette={"Minian": "darkblue", "CaImAn": "red", "Manual": "C2"},
        edgecolor="gray",
        linewidth=2,
    )
    if mtype == "median":
        fig.map_dataframe(set_yaxis, set_range=True)
    else:
        fig.map_dataframe(set_yaxis)
    fig.set_xlabels("Source")
    fig.set_titles(col_template="")
    fig.figure.set_size_inches((WIDTH, WIDTH / ASPECT))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_PATH, "real-{}.svg".format(mtype)))
    fig.savefig(os.path.join(FIG_PATH, "real-{}.png".format(mtype)))

#%% stats on real validation
metrics = ["Acorr", "Scorr"]
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
lm_S = ols("Scorr ~ source", metric_df).fit()
print("F1 score")
print(lm_f1.summary())
print("Spatial correlation")
print(lm_A.summary())
print("Temporal correlation")
print(lm_S.summary())
