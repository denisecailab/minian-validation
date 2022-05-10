import os
import shutil
from zipfile import ZipFile

import requests
from tqdm.auto import tqdm

FILES = {
    "35073556": "./data/raw/grus.zip",
    "35073661": "./data/raw/ferdinand.zip",
    "35073781": "./data/raw/rhea.zip",
    "35075416": "./data/raw/eridanus.zip",
    "35076292": "./data/raw/sao.zip",
    "35076526": "./data/raw/umbriel.zip",
}


def download_file(url, output):
    with requests.get(url, stream=True) as r:
        total_length = int(r.headers.get("Content-Length", 0))
        if not total_length > 0:
            raise ValueError("Failed to download from {}".format(url))
        with tqdm.wrapattr(
            r.raw, "read", total=total_length, desc="downloading {}".format(output)
        ) as raw:
            with open(output, "wb") as out:
                shutil.copyfileobj(raw, out)


def download_figshare(file_id, output):
    download_file(
        "https://api.figshare.com/v2/file/download/{}".format(file_id), output
    )


if __name__ == "__main__":
    for fid, output in FILES.items():
        download_figshare(fid, output)
        if output.endswith(".zip"):
            outdir = output.rstrip(".zip")
            try:
                os.remove(outdir)
            except OSError:
                pass
            print("extracting {}".format(outdir))
            with ZipFile(output, "r") as zip:
                zip.extractall(outdir)
            os.remove(output)
