"""
script to generate simulated data

env: environments/generic.yml
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
    out_path = "data/simulated/validation"
    sig_ls = [0.2, 0.4, 0.6, 1, 1.4, 1.8, 2, 5, 8, 10]
    ncell_ls = [100, 300, 500]

    os.makedirs(out_path, exist_ok=True)
    cluster = LocalCluster(
        n_workers=6,
        memory_limit="10GB",
        threads_per_worker=1,
        dashboard_address="0.0.0.0:12345",
    )
    client = Client(cluster)
    for sig, ncell in itt.product(sig_ls, ncell_ls):
        print("generating {} cells with signal level {}".format(ncell, sig))
        generate_data(
            dpath=os.path.join(out_path, "sig{}-cell{}".format(sig, ncell)),
            ncell=ncell,
            dims={"height": 512, "width": 512, "frame": 20000},
            sig_scale=sig,
            sz_mean=15,
            sz_sigma=5,
            sz_min=3,
            tmp_pfire=0.005,
            tmp_tau_d=60,
            tmp_tau_r=5,
            bg_nsrc=300,
            bg_tmp_var=0.2,
            bg_cons_fac=0.05,
            bg_smth_var=60,
            mo_stp_var=1,
            mo_cons_fac=0.2,
            post_offset=1,
            post_gain=60 / max(sig, 1),
        )
    client.close()
    cluster.close()
