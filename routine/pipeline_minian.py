import os

import numpy as np
from dask.distributed import Client, LocalCluster
from memory_profiler import memory_usage
from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_spatial,
    update_temporal,
)
from minian.initialization import (
    initA,
    initbf,
    initC,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
)
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.utilities import (
    TaskAnnotation,
    get_optimal_chk,
    load_videos,
    open_minian,
    save_minian,
)

from .profiling import PipelineProfiler


def minian_process(dpath, intpath, n_workers, param, profiler: PipelineProfiler):
    # setup
    profiler.change_phase("setup")
    profiler.start()
    dpath = os.path.abspath(os.path.expanduser(dpath))
    intpath = os.path.abspath(os.path.expanduser(intpath))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit="4GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    varr = load_videos(dpath, **param["load_videos"])
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    # preprocessing
    profiler.change_phase("preprocessing")
    varr_ref = varr
    # varr_min = varr_ref.min("frame").compute()
    # varr_ref = varr_ref - varr_min
    varr_ref = denoise(varr_ref, **param["denoise"])
    varr_ref = remove_background(varr_ref, **param["background_removal"])
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)
    # motion-correction
    profiler.change_phase("motion-correction")
    motion = estimate_motion(varr_ref, **param["estimate_motion"])
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), **param["save_minian"]
    )
    Y = apply_transform(varr_ref, motion, fill=0)
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )
    # initilization
    profiler.change_phase("initialization")
    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param["save_minian"]
    ).compute()
    seeds = seeds_init(Y_fm_chk, **param["seeds_init"])
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param["pnr_refine"])
    # seeds = ks_refine(Y_hw_chk, seeds, **param["ks_refine"])
    # seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds[seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param["seeds_merge"])
    A_init = initA(
        Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param["initialize"]
    )
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)
    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    try:
        A, C = unit_merge(A_init, C_init, **param["init_merge"])
    except:
        A, C = A_init, C_init
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    b, f = initbf(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)
    # cnmf
    profiler.change_phase("cnmf")
    sn_spatial = get_noise_fft(Y_hw_chk, **param["get_noise"])
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
    ## first iteration
    A_new, b_new, f_new, mask = update_spatial(
        Y_hw_chk, A, b, C, f, sn_spatial, **param["first_spatial"]
    )
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = C.sel(unit_id=A.coords["unit_id"].values)
    C_chk = C_chk.sel(unit_id=A.coords["unit_id"].values)
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param["first_temporal"]
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)
    ## merge
    try:
        A_mrg, C_mrg, [sig_mrg] = unit_merge(
            A, C, [C + b0 + c0], **param["first_merge"]
        )
    except:
        A_mrg, C_mrg, sig_mrg = A, C, C + b0 + c0
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_mrg_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)
    ## second iteration
    A_new, b_new, f_new, mask = update_spatial(
        Y_hw_chk, A, b, sig, f, sn_spatial, **param["second_spatial"]
    )
    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = C.sel(unit_id=A.coords["unit_id"].values)
    C_chk = C_chk.sel(unit_id=A.coords["unit_id"].values)
    YrA = save_minian(
        compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "frame": -1},
    )
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
        A, C, YrA=YrA, **param["second_temporal"]
    )
    C = save_minian(
        C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )
    S = save_minian(
        S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    b0 = save_minian(
        b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    c0 = save_minian(
        c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
    )
    A = A.sel(unit_id=C.coords["unit_id"].values)
    # save result
    A = save_minian(A.rename("A"), **param["save_minian"])
    C = save_minian(C.rename("C"), **param["save_minian"])
    S = save_minian(S.rename("S"), **param["save_minian"])
    c0 = save_minian(c0.rename("c0"), **param["save_minian"])
    b0 = save_minian(b0.rename("b0"), **param["save_minian"])
    b = save_minian(b.rename("b"), **param["save_minian"])
    f = save_minian(f.rename("f"), **param["save_minian"])
    client.close()
    cluster.close()
    profiler.terminate()


def preprocess_data(dpath, intpath, param, subset=None):
    # setup
    dpath = os.path.abspath(os.path.expanduser(dpath))
    intpath = os.path.abspath(os.path.expanduser(intpath))
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = intpath
    varr = load_videos(dpath, **param["load_videos"])
    varr = varr.sel(subset)
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )
    # preprocessing
    vmin = varr.min("frame").compute()
    varr = varr - vmin
    varr = denoise(varr, **param["denoise"])
    varr = remove_background(varr.astype(float), **param["background_removal"])
    varr = save_minian(varr.rename("varr_ref"), dpath=intpath, overwrite=True)
    max_fm_before = varr.max("frame").compute()
    # motion-correction
    motion = estimate_motion(varr, **param["estimate_motion"])
    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}),
        dpath=intpath,
        overwrite=True,
    )
    varr_mc = apply_transform(varr, motion, fill=0)
    varr_mc = save_minian(varr_mc.rename("varr_mc"), intpath, overwrite=True)
    min_fm = varr_mc.min("frame").compute()
    max_fm = varr_mc.max("frame").compute()
    vmin = min_fm.min()
    vmax = max_fm.max()
    Y = (varr_mc - min_fm) * (255 / (vmax - vmin))
    Y = save_minian(Y.astype(np.uint8).rename("Y"), intpath, overwrite=True)
    return Y, motion, max_fm_before, max_fm


if __name__ == "__main__":
    param = {
        "save_minian": {
            "dpath": "./data/simulated_result",
            "meta_dict": dict(session=-1, animal=-2),
            "overwrite": True,
        },
        "load_videos": {
            "pattern": ".*\.avi$",
            "dtype": np.uint8,
            "downsample": dict(frame=1, height=1, width=1),
            "downsample_strategy": "subset",
        },
        "denoise": {"method": "median", "ksize": 7},
        "background_removal": {"method": "tophat", "wnd": 15},
        "estimate_motion": {"dim": "frame"},
        "seeds_init": {
            "wnd_size": 1000,
            "method": "rolling",
            "stp_size": 500,
            "max_wnd": 15,
            "diff_thres": 2,
        },
        "pnr_refine": {"noise_freq": 0.06, "thres": 1},
        "ks_refine": {"sig": 0.05},
        "seeds_merge": {"thres_dist": 5, "thres_corr": 0.8, "noise_freq": 0.06},
        "initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06},
        "init_merge": {"thres_corr": 0.8},
        "get_noise": {"noise_range": (0.06, 0.5)},
        "first_spatial": {
            "dl_wnd": 5,
            "sparse_penal": 0.01,
            "update_background": True,
            "size_thres": (25, None),
        },
        "first_temporal": {
            "noise_freq": 0.06,
            "sparse_penal": 1,
            "p": 1,
            "add_lag": 20,
            "jac_thres": 0.2,
        },
        "first_merge": {"thres_corr": 0.8},
        "second_spatial": {
            "dl_wnd": 5,
            "sparse_penal": 0.01,
            "update_background": True,
            "size_thres": (25, None),
        },
        "second_temporal": {
            "noise_freq": 0.06,
            "sparse_penal": 1,
            "p": 1,
            "add_lag": 20,
            "jac_thres": 0.4,
        },
    }
    profiler = PipelineProfiler(
        proc=os.getpid(), interval=0.5, outpath="minian_prof.csv", nchild=20
    )
    minian_process(
        "simulated_data",
        "~/var/minian-validation/intermediate",
        16,
        param,
        profiler=profiler,
    )
