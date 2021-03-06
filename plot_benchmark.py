"""
script to generate figures for benchmark results

env: environments/generic.yml
"""

#%% imports and definitions
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import EngFormatter

from routine.plotting import ax_tick, format_tick, it_lab

IN_DPATH = "./data/simulated/benchmark"
IN_PROF_FILES = {
    "minian": "minian_prof.csv",
    "caiman": "caiman_prof.csv",
    "minian-vis": "minian_vis_prof.csv",
    "caiman-vis": "caiman_vis_prof.csv",
}
IN_DIR_PATTERN = r"fm(?P<nfm>[0-9]+)-cell(?P<ncell>[0-9]+)"
FIG_BENCH_ALL = "./fig/benchmark/overall"
OUT_CSV_PATH = "./output/Figure18"

os.makedirs(FIG_BENCH_ALL, exist_ok=True)


def prof_metric(df: pd.DataFrame):
    return pd.Series(
        {
            "duration": df.iloc[-1]["timestamp"] - df.iloc[0]["timestamp"],
            "max_mem": df["mem_sum"].max(),
        }
    )


#%% load benchmark results
df_ls = []
for root, dirs, files in os.walk(IN_DPATH):
    csvf = list(filter(lambda f: f in list(IN_PROF_FILES.values()), files))
    if not csvf:
        continue
    match = re.search(IN_DIR_PATTERN, root.split(os.sep)[-1])
    for pipeline, prof_file in IN_PROF_FILES.items():
        try:
            prof_df = pd.read_csv(os.path.join(root, prof_file))
        except FileNotFoundError:
            continue
        prof_df = prof_df.groupby("phase").apply(prof_metric).reset_index()
        prof_df["pipeline"] = pipeline
        prof_df["nfm"] = match.group("nfm")
        prof_df["ncell"] = match.group("ncell")
        df_ls.append(prof_df)
prof_df = pd.concat(df_ls, ignore_index=True)

#%% plot overall performance
ASPECT = 1.4
SMALL_SIZE = 8
MEDIUM_SIZE = 11
BIG_SIZE = 11
WIDTH = 5.51  # 14cm
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
id_vars = ["pipeline", "nfm", "ncell"]
val_vars = ["duration", "max_mem"]
metric_dict = {"duration": "Run Time (minutes)", "max_mem": "Peak Memory (MB)"}
pipeline_dict = {
    "minian": "Minian",
    "caiman": "CaImAn",
    "minian-vis": "Minian with visualization",
    "caiman-vis": "CaImAn with visualization",
}
pallete = {
    "Minian": "darkblue",
    "CaImAn": "red",
    "Minian with visualization": "orange",
    "CaImAn with visualization": "purple",
}
dash_dict = {
    "Minian": "",
    "CaImAn": "",
    "Minian with visualization": (2, 2),
    "CaImAn with visualization": (2, 2),
}
mk_dict = {
    "Minian": "o",
    "CaImAn": "o",
    "Minian with visualization": "X",
    "CaImAn with visualization": "X",
}


def rename_axis(data, **kwargs):
    ax = plt.gca()
    ax.set_ylabel(metric_dict[data.iloc[0]["variable"]])


prof_agg = (
    prof_df.groupby(id_vars)
    .agg({"duration": "sum", "max_mem": "max"})
    .reset_index()
    .astype({"ncell": int, "nfm": int})
    .sort_values(["pipeline", "ncell", "nfm"])
    .replace({"pipeline": pipeline_dict})
)
prof_agg["duration"] = prof_agg["duration"] / 60
prof_agg = prof_agg.melt(id_vars=id_vars)
os.makedirs(OUT_CSV_PATH, exist_ok=True)
prof_agg.to_csv(os.path.join(OUT_CSV_PATH, "benchmark.csv"), index=False)
fig = sns.FacetGrid(
    prof_agg,
    row="variable",
    col="ncell",
    margin_titles=True,
    legend_out=True,
    sharey="row",
)
fig.map_dataframe(
    sns.lineplot,
    x="nfm",
    y="value",
    hue="pipeline",
    hue_order=list(pipeline_dict.values()),
    style="pipeline",
    dashes=dash_dict,
    markers=mk_dict,
    palette=pallete,
    marker="o",
    legend="full",
)
fig.map_dataframe(rename_axis)
fig.map_dataframe(ax_tick, x_var="nfm")
fig.map(format_tick, x_formatter=EngFormatter(), y_formatter=EngFormatter())
fig.map(it_lab)
fig.figure.set_size_inches((WIDTH, WIDTH / ASPECT))
fig.set_xlabels("Frame Number")
fig.set_titles(row_template="", col_template="{col_name} cells")
fig.figure.tight_layout()
fig.add_legend()
fig.savefig(os.path.join(FIG_BENCH_ALL, "master.svg"))
fig.savefig(os.path.join(FIG_BENCH_ALL, "master.png"))
fig.savefig(os.path.join(FIG_BENCH_ALL, "master.tiff"))
