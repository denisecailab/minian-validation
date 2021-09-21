"""
script to generate ground truth based on manual scoring

env: environments/generic.yml
"""
#%% imports and definitions
import os
import re

import dask.array as darr
import numpy as np
import pandas as pd
import plotly.express as px
import sparse
import xarray as xr

from routine.minian_functions import centroid, open_minian, save_minian
from routine.validation import compute_f1, compute_mapping, convert_rois

IN_DPATH = "./data/real"
IN_ROI_FILES_PAT = r"^manual_score_([A-Z]+).zip$"
IN_PS_DS = "preprocess_ds"
OUT_DS = "truth"

#%% load manual scoring and generate GT
for root, dirs, files in os.walk(IN_DPATH):
    if IN_PS_DS in dirs:
        ps_ds = open_minian(os.path.join(root, IN_PS_DS))
    else:
        continue
    manual_ds = dict()
    for f in files:
        match = re.search(IN_ROI_FILES_PAT, f)
        if match:
            labeler = match.group(1)
            A = xr.DataArray(
                convert_rois(
                    os.path.join(root, f), (ps_ds.sizes["height"], ps_ds.sizes["width"])
                ),
                dims=["unit_id", "height", "width"],
            )
            manual_ds["A_" + labeler] = A.assign_coords(
                height=ps_ds.coords["height"].values,
                width=ps_ds.coords["width"].values,
                unit_id=np.arange(A.sizes["unit_id"]),
            )
    assert len(manual_ds) == 2
    vals = list(manual_ds.values())
    centA = centroid(vals[0])
    centB = centroid(vals[1])
    mapping = compute_mapping(centA, centB, 10)
    A_true = xr.concat(
        [
            vals[0]
            .sel(unit_id=mapping["uidA"].values)
            .assign_coords(unit_id=np.arange(len(mapping))),
            vals[1]
            .sel(unit_id=mapping["uidB"].values)
            .assign_coords(unit_id=np.arange(len(mapping))),
        ],
        "agg",
    ).mean("agg")
    C_true = xr.DataArray(
        darr.tensordot(
            sparse.COO(A_true.transpose("unit_id", "height", "width").data),
            ps_ds["Y"].transpose("height", "width", "frame").data,
            axes=2,
        ).compute(),
        dims=["unit_id", "frame"],
        coords={
            "unit_id": A_true.coords["unit_id"],
            "frame": ps_ds["Y"].coords["frame"],
        },
    )
    manual_ds["A_true"] = A_true
    manual_ds["C_true"] = C_true
    for name, arr in manual_ds.items():
        save_minian(arr.rename(name), os.path.join(root, OUT_DS), overwrite=True)
