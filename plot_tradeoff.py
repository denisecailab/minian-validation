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

IN_DPATH = "./store/tradeoff"
IN_PROF_PATTERN = r"^(?P<pipeline>minian|caiman)_prof_ps(?P<nps>[0-9]+)\.csv$"
FIG_TRADEOFF = "./fig/benchmark/tradeoff"

os.makedirs(FIG_TRADEOFF, exist_ok=True)


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
    for fn in files:
        match = re.search(IN_PROF_PATTERN, fn)
        if match is not None:
            prof_df = pd.read_csv(os.path.join(root, fn))
            prof_df = prof_df.groupby("phase").apply(prof_metric).reset_index()
            prof_df["pipeline"] = match.group("pipeline")
            prof_df["nps"] = match.group("nps")
            df_ls.append(prof_df)
prof_df = pd.concat(df_ls, ignore_index=True)

#%% plot tradeoff
ASPECT = 1.8
SMALL_SIZE = 9
MEDIUM_SIZE = 10
BIG_SIZE = 11
WIDTH = 3.94  # 10cm
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
id_vars = ["pipeline", "nps"]
pipeline_dict = {"minian": "Minian", "caiman": "CaImAn"}
prof_agg = (
    prof_df.groupby(id_vars)
    .agg({"duration": "sum", "max_mem": "max"})
    .reset_index()
    .astype({"nps": int})
    .sort_values(["pipeline", "nps"])
    .replace({"pipeline": pipeline_dict})
)
prof_agg["duration"] = prof_agg["duration"] / 60
prof_agg = prof_agg[prof_agg["nps"] <= 5]
fig, ax = plt.subplots()
sns.lineplot(
    data=prof_agg,
    x="duration",
    y="max_mem",
    hue="pipeline",
    hue_order=("CaImAn", "Minian"),
    palette={"Minian": "darkblue", "CaImAn": "red"},
    marker="o",
    legend="brief",
    ax=ax,
)
ax.set_xlabel("Run Time (minutes)")
ax.set_ylabel("Peak Memory (MB)")
ax.xaxis.set_major_formatter(EngFormatter())
ax.yaxis.set_major_formatter(EngFormatter())
ax.legend(
    bbox_to_anchor=(1, 0.5),
    loc="center left",
    framealpha=0,
)
fig.tight_layout()
fig.savefig(os.path.join(FIG_TRADEOFF, "master.svg"))
fig.savefig(os.path.join(FIG_TRADEOFF, "master.png"))
