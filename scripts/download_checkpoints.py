"""Download the released checkpoints and put them where the configs expect them.

The weights live on the GitHub release, not in the repository. This fetches them,
checks their MD5 against checkpoints/MANIFEST.json, and lays them out as:

    checkpoints/spectrostream_48khz_500k.pth
    checkpoints/dac_48khz_12cb/dac/{weights,metadata}.pth
    checkpoints/bwe_spectrostream_multirate/25k/transformer.pth
    checkpoints/bwe_dac_multirate/25k/transformer.pth

which is what conf/bwe/*.yml (codec_ckpt) and --save_path/--tag point at.

    python scripts/download_checkpoints.py            # everything
    python scripts/download_checkpoints.py --only bwe_spectrostream_multirate_25k.pth
"""
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

REPO = "Multi-Rate-BWE-by-Token-Completion/Multi-rate-BWE"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}\n  -> {dst}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as fh:
        shutil.copyfileobj(r, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="checkpoints/MANIFEST.json")
    ap.add_argument("--base_url", default=None,
                    help="default: the GitHub release named by the manifest's release_tag")
    ap.add_argument("--only", nargs="*", help="asset names to fetch (default: all)")
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    base = a.base_url or f"https://github.com/{REPO}/releases/download/{man['release_tag']}"

    for name, info in man["files"].items():
        if a.only and name not in a.only:
            continue
        dst = Path(info["install_to"])
        if name.endswith(".pth"):
            assets = [(f"{base}/{name}", dst)]
        else:   # the DAC codec folder ships as two files
            assets = [(f"{base}/{name}.weights.pth", dst),
                      (f"{base}/{name}.metadata.pth", dst.parent / "metadata.pth")]
        for url, target in assets:
            if target.exists() and md5(target) == info["md5"]:
                print(f"{name}: already present and verified"); continue
            fetch(url, target)
        got = md5(dst)
        ok = got == info["md5"]
        print(f"{name}: md5 {'OK' if ok else 'MISMATCH ' + got}")
        if not ok:
            raise SystemExit(f"checksum mismatch for {name}; delete {dst} and retry")


if __name__ == "__main__":
    main()
