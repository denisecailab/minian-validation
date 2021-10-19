"""
script to run minian pipeline on datasets

env: environments/minian.yml
"""

import os
import shutil

import numpy as np

from routine.pipeline_minian import minian_process
from routine.profiling import PipelineProfiler

DPATH = "./data/simulated/benchmark/fm28000-cell500"
MINIAN_INT_PATH = "~/var/minian-validation/intermediate"
MINIAN_OUTPATH = "~/var/minian-validation/minian_result"
PROF_OUTPATH = "./store/tradeoff"
NPS_LS = np.arange(16) + 1

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
    "background_removal": {"method": "tophat", "wnd": 5},
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
    "pnr_refine": {"noise_freq": 0.2, "thres": 1.5},
    # "ks_refine": {"sig": 0.05},
    "seeds_merge": {"thres_dist": 3, "thres_corr": 0.9, "noise_freq": 0.2},
    "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.2},
    "init_merge": {"thres_corr": 0.8},
    "get_noise": {"noise_range": (0.2, 0.5)},
    "first_spatial": {
        "dl_wnd": 3,
        "sparse_penal": 1e-4,
        "update_background": True,
        "size_thres": (1, None),
    },
    "first_temporal": {
        "noise_freq": 0.2,
        "sparse_penal": 0.1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.2,
        "med_wd": 1000,
        "use_smooth": False,
    },
    "first_merge": {"thres_corr": 0.6},
    "second_spatial": {
        "dl_wnd": 3,
        "sparse_penal": 1e-3,
        "update_background": True,
        "size_thres": (1, None),
    },
    "second_temporal": {
        "noise_freq": 0.2,
        "sparse_penal": 0.1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.4,
        "med_wd": 1000,
        "use_smooth": False,
    },
}

if __name__ == "__main__":
    DPATH = os.path.abspath(DPATH)
    MINIAN_INT_PATH = os.path.abspath(os.path.expanduser(MINIAN_INT_PATH))
    MINIAN_OUTPATH = os.path.abspath(os.path.expanduser(MINIAN_OUTPATH))
    MINIAN_PARAMS["save_minian"]["dpath"] = MINIAN_OUTPATH
    os.makedirs(PROF_OUTPATH, exist_ok=True)
    for nps in NPS_LS:
        profiler = PipelineProfiler(
            proc=os.getpid(),
            interval=0.2,
            outpath=os.path.join(PROF_OUTPATH, "minian_prof_ps{}.csv".format(nps)),
            nchild=20,
        )
        shutil.rmtree(MINIAN_INT_PATH, ignore_errors=True)
        shutil.rmtree(MINIAN_OUTPATH, ignore_errors=True)
        try:
            minian_process(
                DPATH, MINIAN_INT_PATH, nps, MINIAN_PARAMS, profiler, glow_rm=False
            )
            print("minian success: {}".format(nps))
        except Exception as e:
            print("minian failed: {}".format(nps))
            with open(os.path.join(PROF_OUTPATH, "minian_error"), "w") as txtf:
                txtf.write(str(e))
