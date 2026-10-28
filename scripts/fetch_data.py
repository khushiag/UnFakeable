"""Download and/or unpack a dataset into the folder layout the pipeline expects.

Works with any of:
  * a Google Drive FILE link (e.g. the FaceForensics++.zip / Celeb-DF.zip shared in the team Drive)
  * a direct https:// link to a .zip / .tar / .txt
  * a file you already downloaded with the browser (path to the .zip on your disk)

Examples (run from the project folder with the venv active):
  python scripts/fetch_data.py "https://drive.google.com/file/d/<ID>/view?usp=sharing" --as ffpp
  python scripts/fetch_data.py "C:/Users/you/Downloads/Celeb-DF.zip" --as celebdf
  python scripts/fetch_data.py "C:/Users/you/Downloads/List_of_testing_videos.txt" --as celebdf

--as ffpp     -> data/FaceForensics++   (needs original_sequences/ + manipulated_sequences/ inside)
--as celebdf  -> data/Celeb-DF-v2       (needs Celeb-real/ Celeb-synthesis/ YouTube-real/ + the test list)

Google Drive notes: the link must be shared as "Anyone with the link"; very popular files can hit Drive's
download quota (then use the browser and pass the downloaded file instead). Drive FOLDER links are limited to
50 files by gdown, so download dataset folders as zips (right-click > Download in Drive) instead.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

TARGET = {"ffpp": Path("data") / "FaceForensics++", "celebdf": Path("data") / "Celeb-DF-v2"}
MARKER = {"ffpp": "original_sequences", "celebdf": "Celeb-synthesis"}


def _download(src: str, tmp: Path) -> Path:
    if "drive.google.com" in src or "docs.google.com" in src:
        try:
            import gdown
        except ImportError:
            sys.exit("gdown is missing: pip install gdown")
        if "/folders/" in src:
            out = tmp / "drive_folder"
            gdown.download_folder(src, output=str(out), quiet=False, remaining_ok=True)
            return out
        out = gdown.download(src, output=str(tmp) + "/", quiet=False, fuzzy=True)
        if not out:
            sys.exit("Drive download failed (not shared publicly, or quota exceeded). Download it in the browser "
                     "and pass the downloaded file path instead.")
        return Path(out)
    import requests

    name = src.split("?")[0].rstrip("/").split("/")[-1] or "download.bin"
    out = tmp / name
    with requests.get(src, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(out, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 2**20:8.1f} / {total / 2**20:.1f} MB", end="")
    print()
    return out


def _find(root: Path, marker: str):
    if (root / marker).exists():
        return root
    for depth in ("*", "*/*", "*/*/*"):
        for c in sorted(root.glob(depth)):
            if c.is_dir() and (c / marker).exists():
                return c
    return None


def _place(src: Path, kind: str) -> None:
    dest = TARGET[kind]
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        print(f"Copying folder {src} -> {dest}")
        shutil.copytree(src, dest, dirs_exist_ok=True)
    elif zipfile.is_zipfile(src):
        print(f"Extracting {src.name} -> {dest} (this can take a few minutes)")
        with zipfile.ZipFile(src) as z:
            z.extractall(dest)
    elif tarfile.is_tarfile(src):
        print(f"Extracting {src.name} -> {dest}")
        with tarfile.open(src) as t:
            t.extractall(dest, filter="data")
    elif src.suffix.lower() == ".txt" and kind == "celebdf":
        root = _find(dest, MARKER[kind]) or dest
        shutil.copy2(src, root / src.name)
        print(f"Copied {src.name} -> {root}")
        return
    else:
        sys.exit(f"Don't know how to unpack {src} (expected .zip, .tar, a folder, or the Celeb-DF test list .txt)")
    root = _find(dest, MARKER[kind])
    if root is None:
        print(f"[warning] {MARKER[kind]}/ not found under {dest} yet - fine if this was only part of the dataset.")
    else:
        print(f"Dataset root detected at: {root}")
        stray = _find(dest, "List_of_testing_videos.txt") if kind == "celebdf" else None
        if stray is not None and stray != root and not (root / "List_of_testing_videos.txt").exists():
            shutil.copy2(stray / "List_of_testing_videos.txt", root / "List_of_testing_videos.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Drive link, https URL, or path to a downloaded file/folder")
    ap.add_argument("--as", dest="kind", choices=sorted(TARGET), required=True)
    ap.add_argument("--keep-download", action="store_true", help="keep the downloaded archive in data/_downloads")
    args = ap.parse_args()
    local = Path(args.source.strip('"'))
    if local.exists():
        _place(local, args.kind)
    else:
        tmp = Path("data") / "_downloads"
        tmp.mkdir(parents=True, exist_ok=True)
        got = _download(args.source, tmp)
        _place(got, args.kind)
        if not args.keep_download:
            shutil.rmtree(got) if got.is_dir() else got.unlink(missing_ok=True)
    print("\nNow check what the pipeline sees:\n  python -m unfakeable.cli check-data --config configs/default.yaml")


if __name__ == "__main__":
    main()
