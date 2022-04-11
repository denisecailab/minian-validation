import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage

from .utilities import append_csv


def prof_loop(proc, interval, outpath, states) -> None:
    while True:
        if not states["START"]:
            continue
        mem_all = pd.Series(
            np.array(
                memory_usage(proc=proc, include_children=True, timestamps=True)
            ).squeeze(),
            index=["mem_sum", "timestamp"],
        )
        mem_chld = np.array(memory_usage(proc=proc, multiprocess=True)).squeeze()
        mem_chld = pd.Series(
            mem_chld, index=["mem_chld{}".format(i) for i in range(len(mem_chld))]
        )
        row = pd.concat([mem_all, mem_chld])
        row["mem_swap"] = psutil.swap_memory().used / (1024**2)
        row["phase"] = states["phase"]
        append_csv(row, outpath)
        time.sleep(interval)
        if states["TERMINATE"]:
            break


class PipelineProfiler:
    def __init__(self, proc, interval, outpath, nchild: int = 1) -> None:
        self._manager = mp.Manager()
        self.states = self._manager.dict()
        self.states["START"] = False
        self.states["TERMINATE"] = False
        self.states["phase"] = None
        memdf = pd.DataFrame(
            columns=["timestamp", "phase", "mem_sum", "mem_swap"]
            + ["mem_chld{}".format(i) for i in range(nchild)]
        )
        memdf.to_csv(outpath, index=False)
        self._loop = mp.Process(
            target=prof_loop, args=(proc, interval, outpath, self.states)
        )
        self._loop.start()

    def start(self) -> None:
        self.states["START"] = True

    def terminate(self) -> None:
        self.states["TERMINATE"] = True
        self._loop.join()

    def change_phase(self, ph: str) -> None:
        self.states["phase"] = ph
