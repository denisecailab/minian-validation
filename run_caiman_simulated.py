"""
script to run caiman pipeline on datasets

env: environments/caiman.yml
"""

import logging
import os
import shutil
import warnings

from routine.pipeline_caiman import caiman_process
from routine.profiling import PipelineProfiler

DPATH = "./data/simulated/benchmark"
CAIMAN_INT_PATH = "~/var/minian-validation/intermediate-cm"


MC_DICT = {
    "fr": 60,  # movie frame rate
    "decay_time": 0.1,  # length of a typical transient in seconds
    "pw_rigid": False,  # flag for pw-rigid motion correction
    "max_shifts": (20, 20),  # maximum allowed rigid shift
    "gSig_filt": (3, 3),  # size of filter, in general gSig (see below)
    "strides": (48, 48),  # start a new patch for pw-rigid mc every x pixels
    "overlaps": (24, 24),  # overlap between pathes (size of patch strides+overlaps)
    "max_deviation_rigid": 3,  # maximum deviation allowed for patch with respect to rigid shifts
    "border_nan": "copy",
}
PARAM_DICT = {
    "method_init": "corr_pnr",  # use this for 1 photon
    "K": None,  # upper bound on number of components per patch, in general None for 1p data
    "gSig": (3, 3),  # width of a 2D gaussian kernel, which approximates a neuron
    "gSiz": (13, 13),  # average diameter of a neuron, in general 4*gSig+1
    "merge_thr": 0.7,  # merging threshold, max correlation allowed
    "p": 1,  # order of the autoregressive system
    "tsub": 1,  # downsampling factor in time for initialization
    "ssub": 1,  # downsampling factor in space for initialization
    "rf": 40,  # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    "stride": 20,  # amount of overlap between the patches in pixels
    "only_init": True,  # set it to True to run CNMF-E
    "nb": 0,  # number of background components (rank) if positive
    "nb_patch": 0,  # number of background components (rank) per patch if gnb>0
    "method_deconvolution": "oasis",  # could use 'cvxpy' alternatively
    "low_rank_background": None,  # None leaves background of each patch intact
    "update_background_components": True,  # sometimes setting to False improve the results
    "min_corr": 0.8,  # min peak value from correlation image
    "min_pnr": 5,  # min peak to noise ration from PNR image
    "normalize_init": False,  # just leave as is
    "center_psf": True,  # leave as is for 1 photon
    "ssub_B": 2,  # additional downsampling factor in space for background
    "ring_size_factor": 1.4,  # radius of ring is gSiz*ring_size_factor
    "del_duplicates": True,  # whether to remove duplicates from initialization
    "memory_efficient": True,
}
QUALITY_DICT = {
    "min_SNR": 2.5,  # adaptive way to set threshold on the transient size
    "rval_thr": 0.85,  # threshold on space consistency
    "use_cnn": False,
}

if __name__ == "__main__":
    DPATH = os.path.abspath(DPATH)
    CAIMAN_INT_PATH = os.path.abspath(os.path.expanduser(CAIMAN_INT_PATH))
    logging.basicConfig(force=True)
    logging.getLogger().setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    for root, dirs, files in os.walk(DPATH, followlinks=True):
        avifiles = list(filter(lambda f: f.endswith(".avi"), files))
        if not avifiles:
            continue
        profiler = PipelineProfiler(
            proc=os.getpid(),
            interval=0.2,
            outpath=os.path.join(root, "caiman_prof.csv"),
            nchild=20,
        )
        shutil.rmtree(CAIMAN_INT_PATH, ignore_errors=True)
        os.makedirs(CAIMAN_INT_PATH, exist_ok=True)
        try:
            caiman_process(
                root,
                root,
                CAIMAN_INT_PATH,
                16,
                MC_DICT,
                PARAM_DICT,
                QUALITY_DICT,
                profiler,
                copy_to_int=True,
            )
            print("caiman sucess: {}".format(root))
        except Exception as e:
            print("caiman failed: {}".format(root))
            with open(os.path.join(root, "caiman_error"), "w") as txtf:
                txtf.write(str(e))
