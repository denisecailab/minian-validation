import os

import numpy as np
from dask.distributed import Client, LocalCluster
from minian.utilities import TaskAnnotation, get_optimal_chk, load_videos, save_minian

DPATH = "./data/simulated/validation/sig1.8-cell300"

if __name__ == "__main__":
    cluster = LocalCluster(
        n_workers=16,
        memory_limit="4GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address="0.0.0.0:12345",
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    varr = load_videos(DPATH, r"simulated.*\.avi$", dtype=np.uint8)
    chk, _ = get_optimal_chk(varr, dtype=float)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        os.path.join(DPATH, "varr"),
        overwrite=True,
    )
