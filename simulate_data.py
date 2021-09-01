"""
script to generate simulated data

env: environments/environment-generic.yml
"""

import itertools as itt
import os

import dask as da
from distributed import Client, LocalCluster

from routine.simulation import generate_data

if __name__ == "__main__":
    da.config.set(
        **{
            # "optimization.fuse.ave-width": 5,
            # "optimization.fuse.subgraphs": True,
            "distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0,
            "distributed.worker.memory.target": 0.9,
            "distributed.worker.memory.spill": 0.95,
            "distributed.worker.memory.pause": 0.98,
            "distributed.worker.memory.terminate": 0.99,
        }
    )
    out_path = "data/simulated"
    frame_ls = [4000, 12000, 20000, 28000]
    ncell_ls = [100, 300, 500]

    os.makedirs(out_path, exist_ok=True)
    cluster = LocalCluster(
        n_workers=4,
        memory_limit="10GB",
        threads_per_worker=1,
        dashboard_address="0.0.0.0:12345",
    )
    client = Client(cluster)
    for nfm, ncell in itt.product(frame_ls, ncell_ls):
        print("generating {} cells with {} frames".format(ncell, nfm))
        generate_data(
            dpath=os.path.join(out_path, "fm{}-cell{}".format(nfm, ncell)),
            ncell=ncell,
            dims={"height": 512, "width": 512, "frame": nfm},
            sig_scale=1,
            sz_mean=3,
            sz_sigma=0.6,
            sz_min=0.1,
            tmp_pfire=0.01,
            tmp_tau_d=6,
            tmp_tau_r=1,
            bg_nsrc=100,
            bg_tmp_var=2,
            mo_stp_var=0.05,
            mo_cons_fac=1,
            post_offset=1,
            post_gain=50,
        )
    client.close()
    cluster.close()
