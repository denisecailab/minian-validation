#!/usr/bin/env python

"""
Adapted from default caiman 1p pipeline:
https://github.com/flatironinstitute/CaImAn/blob/721c47b0e945d2a40b9ebb007b1baa61b47be1f8/demos/general/demo_pipeline_cnmfE.py

Complete pipeline for motion correction, source extraction, and deconvolution
of one photon microendoscopic calcium imaging data using the CaImAn package.
The demo demonstrates how to use the params, MotionCorrect and cnmf objects
for processing 1p microendoscopic data. The analysis pipeline is similar as in
the case of 2p data processing with core difference being the usage of the
CNMF-E algorithm for source extraction (as opposed to plain CNMF). Check
the companion paper for more details.

You can also run a large part of the pipeline with a single method
(cnmf.fit_file) See inside for details.

Demo is also available as a jupyter notebook (see demo_pipeline_cnmfE.ipynb)
"""

import os
import re
from shutil import copyfile

import caiman as cm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf import params as params
from natsort import natsorted

from .profiling import PipelineProfiler


def caiman_process(
    dpath,
    outpath,
    intpath,
    n_processes,
    mc_dict,
    params_dict,
    quality_dict,
    profiler: PipelineProfiler = None,
    vpat: str = r".*\.avi",
    copy_to_int=False,
):
    fnames = natsorted(
        [os.path.join(dpath, v) for v in os.listdir(dpath) if re.search(vpat, v)]
    )
    if copy_to_int:
        fnames_new = []
        for f in fnames:
            fnew = os.path.join(intpath, os.path.relpath(f, dpath))
            copyfile(f, fnew)
            fnames_new.append(fnew)
        fnames = fnames_new
        print("done copying data")
    # setup
    if profiler is not None:
        profiler.change_phase("setup")
        profiler.start()
    try:
        cm.stop_server()
    except:
        pass
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend="local",
        n_processes=n_processes,
        single_thread=False,
        ignore_preexisting=True,
    )
    dpath = os.path.normpath(os.path.abspath(dpath))
    outpath = os.path.normpath(os.path.abspath(outpath))
    os.makedirs(outpath, exist_ok=True)
    mc_dict["fnames"] = fnames
    opts = params.CNMFParams(params_dict=mc_dict)
    # do motion correction rigid
    if profiler is not None:
        profiler.change_phase("motion-correction")
    pw_rigid = mc_dict["pw_rigid"]
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group("motion"))
    mc.motion_correct(save_movie=True)
    print("mc done")
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(
            np.maximum(np.max(np.abs(mc.x_shifts_els)), np.max(np.abs(mc.y_shifts_els)))
        ).astype(np.int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        plt.subplot(1, 2, 1)
        plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2)
        plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(["x shifts", "y shifts"])
        plt.xlabel("frames")
        plt.ylabel("pixels")
    bord_px = 0 if mc_dict["border_nan"] == "copy" else bord_px
    fname_new = cm.save_memmap(
        fname_mc, base_name="memmap_", order="C", border_to_0=bord_px
    )
    print("mmapping done")
    # load memory mappable file
    if profiler is not None:
        profiler.change_phase("initialization")
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order="F")
    # initialization
    Ain = None  # possibility to seed with predetermined binary masks
    params_dict["dims"] = dims
    params_dict["border_pix"] = bord_px
    opts.change_params(params_dict=params_dict)
    # compute some summary images (correlation and peak to noise)
    # cn_filter, pnr = cm.summary_images.correlation_pnr(
    #     images[::1], gSig=params_dict["gSig"][0], swap_dim=False
    # )
    # inspect_correlation_pnr(cn_filter, pnr)
    # cnmf
    print("initialization done")
    if profiler is not None:
        profiler.change_phase("cnmf")
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
    cnm.fit(images)
    print("cnmf done")
    # post-hoc curating
    if profiler is not None:
        profiler.change_phase("post-hoc")
    cnm.params.set("quality", quality_dict)
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    print(" ***** ")
    print("Number of total components: ", len(cnm.estimates.C))
    print("Number of accepted components: ", len(cnm.estimates.idx_components))
    save_caiman(cnm.estimates, outpath)
    # terminate
    cm.stop_server(dview=dview)
    if profiler is not None:
        profiler.terminate()
    files = [f for f in os.listdir(dpath) if f.endswith(".mmap")]
    for f in files:
        os.remove(os.path.join(dpath, f))


def save_caiman(cnm, dpath, dsname="caiman_result.nc"):
    A = xr.DataArray(
        cnm.A.toarray().reshape(tuple([*cnm.dims, -1]))[:, :, cnm.idx_components],
        dims=["width", "height", "unit_id"],
    ).transpose("unit_id", "height", "width")
    C = xr.DataArray(cnm.C[cnm.idx_components, :], dims=["unit_id", "frame"])
    S = xr.DataArray(cnm.S[cnm.idx_components, :], dims=["unit_id", "frame"])
    ds = xr.Dataset({"A": A, "C": C, "S": S})
    for d in ds.dims:
        ds = ds.assign_coords({d: np.arange(ds.sizes[d])})
    ds.to_netcdf(os.path.join(dpath, dsname), mode="w")
    return ds


if __name__ == "__main__":
    mc_dict = {
        "fr": 10,  # movie frame rate
        "decay_time": 0.4,  # length of a typical transient in seconds
        "pw_rigid": False,  # flag for pw-rigid motion correction
        "max_shifts": (5, 5),  # maximum allowed rigid shift
        "gSig_filt": (3, 3),  # size of filter, in general gSig (see below)
        "strides": (48, 48),  # start a new patch for pw-rigid mc every x pixels
        "overlaps": (24, 24),  # overlap between pathes (size of patch strides+overlaps)
        "max_deviation_rigid": 3,  # maximum deviation allowed for patch with respect to rigid shifts
        "border_nan": "copy",
    }
    params_dict = {
        "method_init": "corr_pnr",  # use this for 1 photon
        "K": None,  # upper bound on number of components per patch, in general None for 1p data
        "gSig": (3, 3),  # width of a 2D gaussian kernel, which approximates a neuron
        "gSiz": (13, 13),  # average diameter of a neuron, in general 4*gSig+1
        "merge_thr": 0.7,  # merging threshold, max correlation allowed
        "p": 1,  # order of the autoregressive system
        "tsub": 2,  # downsampling factor in time for initialization
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
        "min_pnr": 10,  # min peak to noise ration from PNR image
        "normalize_init": False,  # just leave as is
        "center_psf": True,  # leave as is for 1 photon
        "ssub_B": 2,  # additional downsampling factor in space for background
        "ring_size_factor": 1.4,  # radius of ring is gSiz*ring_size_factor
        "del_duplicates": True,  # whether to remove duplicates from initialization
    }
    quality_dict = {
        "min_SNR": 2.5,  # adaptive way to set threshold on the transient size
        "rval_thr": 0.85,  # threshold on space consistency
        "use_cnn": False,
    }
    profiler = PipelineProfiler(
        proc=os.getpid(), interval=0.5, outpath="caiman_prof.csv", nchild=20
    )
    caiman_process(
        "simulated_data",
        "simulated_data",
        16,
        mc_dict,
        params_dict,
        quality_dict,
        profiler=profiler,
    )
