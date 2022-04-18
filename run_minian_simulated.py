"""
script to run minian pipeline on datasets

env: environments/minian.yml
"""

import io
import os
import re
import shutil
import traceback
import warnings
from contextlib import redirect_stdout
from copy import deepcopy

import numpy as np

from routine.pipeline_minian import minian_process
from routine.profiling import PipelineProfiler

DPATH = "./data/simulated/validation"
MINIAN_INT_PATH = "~/var/minian-validation/intermediate"

MINIAN_PARAMS = {
    "save_minian": {
        "meta_dict": None,
        "overwrite": True,
    },
    "load_videos": {
        "pattern": ".*\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    },
    "denoise": {"method": "median", "ksize": 5},
    "background_removal": {"method": "tophat", "wnd": 10},
    "estimate_motion": {
        "dim": "frame",
        "aggregation": "max",
        "alt_error": 5,
        "upsample": 10,
    },
    "seeds_init": {
        "wnd_size": 1000,
        "method": "rolling",
        "stp_size": 500,
        "max_wnd": 10,
        "diff_thres": 5,
    },
    "pnr_refine": {"noise_freq": 0.06, "thres": 2},
    # "ks_refine": {"sig": 0.05},
    "seeds_merge": {"thres_dist": 2, "thres_corr": 0.95, "noise_freq": 0.06},
    "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
    "init_merge": {"thres_corr": 0.8},
    "get_noise": {"noise_range": (0.06, 0.5)},
    "first_spatial": {
        "dl_wnd": 3,
        "sparse_penal": 1e-4,
        "size_thres": (5, None),
    },
    "first_temporal": {
        "noise_freq": 0.06,
        "sparse_penal": 0.5,
        "p": 2,
        "add_lag": 100,
        "jac_thres": 0.2,
        "med_wd": None,
        "use_smooth": False,
    },
    "first_merge": {"thres_corr": 0.6},
    "second_spatial": {
        "dl_wnd": 3,
        "sparse_penal": 1e-3,
        "size_thres": (5, None),
    },
    "second_temporal": {
        "noise_freq": 0.06,
        "sparse_penal": 0.5,
        "p": 2,
        "add_lag": 100,
        "jac_thres": 0.4,
        "med_wd": None,
        "use_smooth": False,
    },
}
PARAM_PER_SIG = {"0.2": {"seeds_init": {"diff_thres": 4}, "pnr_refine": {"thres": 1.5}}}

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    DPATH = os.path.abspath(DPATH)
    MINIAN_INT_PATH = os.path.abspath(os.path.expanduser(MINIAN_INT_PATH))
    for root, dirs, files in os.walk(DPATH):
        avifiles = list(filter(lambda f: f.endswith(".avi"), files))
        if not avifiles:
            continue
        params = deepcopy(MINIAN_PARAMS)
        params["save_minian"]["dpath"] = os.path.join(root, "minian_result")
        fmatch = re.search(r"sig([0-9\.]+)-cell([0-9]+)", root)
        if fmatch:
            sig, ncell = fmatch.groups()
            try:
                params.update(PARAM_PER_SIG[sig])
            except KeyError:
                pass
        profiler = PipelineProfiler(
            proc=os.getpid(),
            interval=0.2,
            outpath=os.path.join(root, "minian_prof.csv"),
            nchild=10,
        )
        shutil.rmtree(MINIAN_INT_PATH, ignore_errors=True)
        try:
            with redirect_stdout(io.StringIO()):
                A, C, S = minian_process(
                    root, MINIAN_INT_PATH, 4, params, profiler, glow_rm=False
                )
            print("minian success: {}".format(root))
            print("ncells: {}".format(A.sizes["unit_id"]))
        except Exception as err:
            print("minian failed: {}".format(root))
            with open(os.path.join(root, "minian_error"), "w") as txtf:
                traceback.print_exception(None, err, err.__traceback__, file=txtf)
