"""
script to preprocess real data for hand-scoring

env: environments/minian.yml
"""

import os
import shutil

import ffmpeg
import numpy as np
import holoviews as hv

from routine.minian_functions import write_video, save_minian
from routine.pipeline_minian import preprocess_data
from distributed import LocalCluster, Client
from minian.utilities import TaskAnnotation
from natsort import natsorted

IN_DPATH = "data/real/raw"
INT_PATH = "~/var/minian-validation/intermediate"
WORKER_PATH = "~/var/dask-worker-space"
OUT_DPATH = "data/real/preprocessed"

MINIAN_PARAMS = {
    "load_videos": {
        "pattern": ".*\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=1, width=1),
        "downsample_strategy": "subset",
    },
    "denoise": {"method": "median", "ksize": 3},
    "background_removal": {"method": "uniform", "wnd": 50},
    "estimate_motion": {"dim": "frame", "aggregation": "max", "alt_error": 5},
}

if __name__ == "__main__":
    hv.notebook_extension("bokeh")
    IN_DPATH = os.path.abspath(IN_DPATH)
    INT_PATH = os.path.abspath(os.path.expanduser(INT_PATH))
    WORKER_PATH = os.path.abspath(os.path.expanduser(WORKER_PATH))
    for root, dirs, files in os.walk(IN_DPATH, followlinks=True):
        avifiles = list(filter(lambda f: f.endswith(".avi"), files))
        if not avifiles:
            continue
        cluster = LocalCluster(
            n_workers=16,
            memory_limit="4GB",
            resources={"MEM": 1},
            threads_per_worker=2,
            dashboard_address="0.0.0.0:12345",
            local_directory=WORKER_PATH,
        )
        annt_plugin = TaskAnnotation()
        cluster.scheduler.add_plugin(annt_plugin)
        client = Client(cluster)
        shutil.rmtree(INT_PATH, ignore_errors=True)
        Y, motion, maxfm_bf, maxfm_aft = preprocess_data(
            root, INT_PATH, MINIAN_PARAMS, subset={"frame": slice(None, 20 * 60 * 30)}
        )
        outpath = os.path.join(OUT_DPATH, os.path.relpath(root, IN_DPATH))
        os.makedirs(outpath, exist_ok=True)
        opts_im = {
            "frame_width": 608,
            "frame_height": 608,
            "cmap": "viridis",
            "show_title": True,
        }
        mc_result = hv.Image(maxfm_bf.rename("before"), ["width", "height"]).opts(
            **opts_im
        ) + hv.Image(maxfm_aft.rename("after"), ["width", "height"]).opts(**opts_im)
        hv.save(mc_result, os.path.join(outpath, "mc_result.html"))
        del Y.encoding["chunks"]
        del motion.encoding["chunks"]
        Y = save_minian(
            Y.chunk({"frame": 1000}).rename("Y"),
            dpath=os.path.join(outpath, "preprocess_ds"),
            overwrite=True,
        )
        motion = save_minian(
            motion.chunk({"frame": 1000}).rename("motion"),
            dpath=os.path.join(outpath, "preprocess_ds"),
            overwrite=True,
        )
        vchunked = write_video(
            Y,
            vpath=outpath,
            vname="chunked",
            vext="avi",
            options={"r": "30", "pix_fmt": "gray", "vcodec": "rawvideo"},
            chunked=True,
        )
        ffmpeg.input("concat:" + "|".join(natsorted(vchunked))).output(
            os.path.join(outpath, "concat.avi"), vcodec="copy"
        ).run(overwrite_output=True)
        [os.remove(vf) for vf in vchunked]
        client.close()
        cluster.close()
