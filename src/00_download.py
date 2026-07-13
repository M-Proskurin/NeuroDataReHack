"""Stage 0 — access dandiset 000447 via the DANDI Python API + pynwb.

Two access modes:

  * **stream** (default, no local copy) — lazily read an NWB file straight from
    the DANDI S3 store with `remfile`. Datasets stay on the remote; only the
    bytes you index are fetched. This is the recommended way to explore and to
    feed stage-1 extraction without downloading ~33 GiB.

  * **download** — pull whole assets (or the full dandiset) into data/raw/ with
    the DANDI Python API, for offline / repeated heavy access.

CLI:
    pixi run python src/00_download.py --list                 # list asset paths
    pixi run python src/00_download.py --stream <asset-path>  # open + print lazily
    pixi run python src/00_download.py --download <asset-path> # one asset to data/raw/
    pixi run python src/00_download.py --download-all          # full dandiset (~33 GiB)

Programmatic use (e.g. from src/01_extraction.py):
    from importlib import import_module
    dl = import_module("00_download")
    with dl.stream_nwb("sub-XX/sub-XX_....nwb") as nwb:
        units = nwb.units.to_dataframe()   # only the bytes touched are fetched

Raw NWB files under data/raw/ are gitignored.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Iterator

import h5py
import remfile
from dandi.dandiapi import DandiAPIClient
from dandi.download import DownloadExisting, download
from pynwb import NWBHDF5IO
from pynwb.file import NWBFile

from config import DANDISET_ID, DATA_RAW

DEFAULT_VERSION = "draft"


def list_asset_paths(dandiset_id: str = DANDISET_ID,
                     version: str = DEFAULT_VERSION) -> list[str]:
    """Return the paths of all assets in the dandiset."""
    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(dandiset_id, version)
        return [asset.path for asset in dandiset.get_assets()]


def get_s3_url(filepath: str, dandiset_id: str = DANDISET_ID,
               version: str = DEFAULT_VERSION) -> str:
    """Resolve the streamable S3 URL for one asset path within the dandiset."""
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, version).get_asset_by_path(filepath)
        return asset.get_content_url(follow_redirects=1, strip_query=True)


@contextmanager
def stream_nwb(filepath: str, dandiset_id: str = DANDISET_ID,
               version: str = DEFAULT_VERSION) -> Iterator[NWBFile]:
    """Lazily open a remote NWB file for reading; yields an NWBFile.

    Uses remfile -> h5py -> NWBHDF5IO so datasets are read on demand (indexing
    `series.data[a:b]` fetches only those bytes). All data access must happen
    inside the `with` block; the I/O and remote handles close on exit.
    """
    s3_url = get_s3_url(filepath, dandiset_id, version)
    rem_file = remfile.File(s3_url)
    h5_file = h5py.File(rem_file, "r")
    io = NWBHDF5IO(file=h5_file, mode="r")
    try:
        yield io.read()
    finally:
        io.close()
        h5_file.close()
        rem_file.close()


def download_asset(filepath: str, dandiset_id: str = DANDISET_ID,
                   version: str = DEFAULT_VERSION, skip_existing: bool = True):
    """Download a single asset into data/raw/ (returns the download dir)."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    url = get_s3_url(filepath, dandiset_id, version)
    existing = DownloadExisting("skip") if skip_existing else DownloadExisting("overwrite")
    download(url, str(DATA_RAW), existing=existing)
    return DATA_RAW


def download_all(dandiset_id: str = DANDISET_ID, version: str = DEFAULT_VERSION):
    """Download the entire dandiset into data/raw/ (~33 GiB)."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    url = f"https://dandiarchive.org/dandiset/{dandiset_id}/{version}"
    download(url, str(DATA_RAW), existing=DownloadExisting("skip"))
    return DATA_RAW


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                       help="list all asset paths in the dandiset")
    group.add_argument("--stream", metavar="ASSET_PATH",
                       help="lazily open an NWB asset and print a summary")
    group.add_argument("--download", metavar="ASSET_PATH",
                       help="download one asset into data/raw/")
    group.add_argument("--download-all", action="store_true",
                       help="download the whole dandiset into data/raw/ (~33 GiB)")
    args = parser.parse_args()

    if args.list:
        for path in list_asset_paths():
            print(path)
    elif args.stream:
        with stream_nwb(args.stream) as nwb:
            print(nwb)
            if nwb.units is not None:
                print(f"\n{len(nwb.units)} sorted units")
    elif args.download:
        out = download_asset(args.download)
        print(f"downloaded {args.download} -> {out}")
    elif args.download_all:
        out = download_all()
        print(f"downloaded dandiset {DANDISET_ID} -> {out}")


if __name__ == "__main__":
    main()
