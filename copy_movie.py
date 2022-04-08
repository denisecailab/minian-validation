import os
import re
import shutil

IN_DPATH = "./data/demo_concat"
NREPEAT = 6
LENREPEAT = 10

for avif in os.listdir(IN_DPATH):
    match = re.search(r"msCam(\d+)\.avi", avif)
    if match:
        idx = int(match.group(1))
    else:
        continue
    if idx <= LENREPEAT:
        for irpt in range(1, NREPEAT + 1):
            newfn = "msCam{}.avi".format(idx + irpt * LENREPEAT)
            shutil.copyfile(os.path.join(IN_DPATH, avif), os.path.join(IN_DPATH, newfn))
