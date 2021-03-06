import itertools as itt
import os

import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from minian.cnmf import (
    compute_trace,
    get_noise_fft,
    unit_merge,
    update_background,
    update_spatial,
    update_temporal,
)
from minian.initialization import initA, initC, pnr_refine, seeds_init, seeds_merge
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.utilities import (
    TaskAnnotation,
    get_optimal_chk,
    load_videos,
    open_minian,
    save_minian,
)
from minian.visualization import (
    generate_videos,
    visualize_preprocess,
    visualize_spatial_update,
    visualize_temporal_update,
    write_video,
)

from .profiling import PipelineProfiler


def minian_process(
    dpath,
    intpath,
    n_workers,
    param,
    profiler: PipelineProfiler,
    glow_rm=True,
    visualization=False,
):
    # setup
    profiler.change_phase("setup")
    profiler.start()
    hv.notebook_extension("bokeh")
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
    if glow_rm:
        varr_min = varr_ref.min("frame").compute()
        varr_ref = varr_ref - varr_min
    if visualization:
        visualize_preprocess(
            varr_ref.isel(frame=0).compute(),
            denoise,
            method=["median"],
            ksize=[5, 7, 9],
        )
        visualize_preprocess(
            varr_ref.isel(frame=0).compute(),
            remove_background,
            method=["tophat"],
            wnd=[10, 15, 20],
        )
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
    if visualization:
        vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
        write_video(vid_arr, "minian_mc.mp4", dpath)
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
    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)
    # cnmf
    profiler.change_phase("cnmf")
    sn_spatial = get_noise_fft(Y_hw_chk, **param["get_noise"])
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)
    ## first iteration
    if visualization:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C.sel(unit_id=units).persist()
        sprs_ls = [0.005, 0.01, 0.05]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_mask, cur_norm = update_spatial(
                Y_hw_chk,
                A_sub,
                C_sub,
                sn_spatial,
                in_memory=True,
                dl_wnd=param["first_spatial"]["dl_wnd"],
                sparse_penal=cur_sprs,
            )
            if cur_A.sizes["unit_id"]:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk,
        A,
        C,
        sn_spatial,
        **param["first_spatial"],
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
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
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    if visualization:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C_chk.sel(unit_id=units).persist()
        p_ls = [1]
        sprs_ls = [0.1, 0.5, 1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = (
            compute_trace(Y_fm_chk, A_sub, b, C_sub, f)
            .persist()
            .chunk({"unit_id": 1, "frame": -1})
        )
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(
            p_ls, sprs_ls, add_ls, noise_ls
        ):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print(
                "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(
                    cur_p, cur_sprs, cur_add, cur_noise
                )
            )
            cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                A_sub,
                C_sub,
                YrA=YrA,
                sparse_penal=cur_sprs,
                p=cur_p,
                use_smooth=True,
                add_lag=cur_add,
                noise_freq=cur_noise,
            )
            (
                YA_dict[ks],
                C_dict[ks],
                S_dict[ks],
                g_dict[ks],
                sig_dict[ks],
                A_dict[ks],
            ) = (
                YrA.compute(),
                cur_C.compute(),
                cur_S.compute(),
                cur_g.compute(),
                (cur_C + cur_b0 + cur_c0).compute(),
                A_sub.compute(),
            )
        hv_res = visualize_temporal_update(
            YA_dict,
            C_dict,
            S_dict,
            g_dict,
            sig_dict,
            A_dict,
            kdims=["p", "sparse penalty", "additional lag", "noise frequency"],
        )
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
    if visualization:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C.sel(unit_id=units).persist()
        sprs_ls = [0.005, 0.01, 0.05]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_mask, cur_norm = update_spatial(
                Y_hw_chk,
                A_sub,
                C_sub,
                sn_spatial,
                in_memory=True,
                dl_wnd=param["first_spatial"]["dl_wnd"],
                sparse_penal=cur_sprs,
            )
            if cur_A.sizes["unit_id"]:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])
    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, C, sn_spatial, **param["second_spatial"]
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"),
        intpath,
        overwrite=True,
    )
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)
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
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    if visualization:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C_chk.sel(unit_id=units).persist()
        p_ls = [1]
        sprs_ls = [0.1, 0.5, 1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = (
            compute_trace(Y_fm_chk, A_sub, b, C_sub, f)
            .persist()
            .chunk({"unit_id": 1, "frame": -1})
        )
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(
            p_ls, sprs_ls, add_ls, noise_ls
        ):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print(
                "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(
                    cur_p, cur_sprs, cur_add, cur_noise
                )
            )
            cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                A_sub,
                C_sub,
                YrA=YrA,
                sparse_penal=cur_sprs,
                p=cur_p,
                use_smooth=True,
                add_lag=cur_add,
                noise_freq=cur_noise,
            )
            (
                YA_dict[ks],
                C_dict[ks],
                S_dict[ks],
                g_dict[ks],
                sig_dict[ks],
                A_dict[ks],
            ) = (
                YrA.compute(),
                cur_C.compute(),
                cur_S.compute(),
                cur_g.compute(),
                (cur_C + cur_b0 + cur_c0).compute(),
                A_sub.compute(),
            )
        hv_res = visualize_temporal_update(
            YA_dict,
            C_dict,
            S_dict,
            g_dict,
            sig_dict,
            A_dict,
            kdims=["p", "sparse penalty", "additional lag", "noise frequency"],
        )
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
    if visualization:
        generate_videos(varr, Y_fm_chk, A=A, C=C_chk, vpath=dpath)
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
    return A, C, S


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
