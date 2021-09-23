"""
script to generate figures for validation results

env: environments/generic.yml
"""

#%% imports and definitions
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

from routine.minian_functions import open_minian
from routine.validation import compute_metrics

IN_SIM_DPATH = "./data/simulated/validation"
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
metrics = ["Acorr", "Scorr"]
id_vars = ["ncell", "sig", "pipeline"]
mapping_df = pd.read_feather(os.path.join(OUT_PATH, "mapping_simulated.feather"))
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
f1_df = pd.read_feather(os.path.join(OUT_PATH, "f1_simulated.feather"))
for mtype, mdf in metric_df.items():
    df = mdf.merge(f1_df, on=id_vars, validate="one_to_one").melt(id_vars=id_vars)
    fig = px.line(
        df,
        x="sig",
        y="value",
        facet_col="ncell",
        facet_row="variable",
        color="pipeline",
        markers=True,
    )
    nrow, _ = fig._get_subplot_rows_columns()
    for r in nrow:
        fig.update_yaxes(matches="y" + str(r) * 3, row=r)
    fig.update_yaxes(range=[0.8, 1.1])
    fig.write_image(os.path.join(FIG_PATH, "simulated-{}.pdf".format(mtype)))
