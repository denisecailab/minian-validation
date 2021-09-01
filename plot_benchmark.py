"""
script to generate figures for benchmark results

env: environments/environment-generic.yml
"""

#%% imports and definitions
import os
import re

import pandas as pd
import plotly.express as px

IN_DPATH = "./data"
IN_PROF_FILES = {"minian": "minian_prof.csv", "caiman": "caiman_prof.csv"}
IN_DIR_PATTERN = r"fm(?P<nfm>[0-9]+)-cell(?P<ncell>[0-9]+)"
FIG_BENCH_ALL = "./fig/benchmark/overall"

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
id_vars = ["pipeline", "nfm", "ncell"]
val_vars = ["duration", "max_mem"]
facet_dim = "ncell"
prof_agg = (
    prof_df.groupby(id_vars)
    .agg({"duration": "sum", "max_mem": "max"})
    .reset_index()
    .astype({"ncell": int, "nfm": int})
    .sort_values(["pipeline", "ncell", "nfm"])
)
for variable in val_vars:
    fig = px.line(
        prof_agg,
        x="nfm",
        y=variable,
        color="pipeline",
        facet_col=facet_dim,
        markers=True,
    )
    fig.write_image(os.path.join(FIG_BENCH_ALL, "{}.pdf".format(variable)))
