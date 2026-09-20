"""Export the released checkpoints: strip optimiser state, keep what inference needs.

The training checkpoints carry the optimiser and scheduler state, which triples
their size and is useless for anyone who only wants to run the models. This
writes inference-only copies and a manifest with sizes and MD5 sums.

    python scripts/export_checkpoints.py --out dist/ \
        --sps_codec   /path/to/runs/codec_spectrostream/500k/generator.pth \
        --sps_bwe     /path/to/runs/bwe_sps_multirate/25k/transformer.pth \
        --dac_bwe     /path/to/runs/bwe_dac_multirate/25k/transformer.pth \
        --dac_codec   /path/to/runs/codec_dac_48khz

What is kept:
  SpectroStream codec / predictors   model.pth and tracker.pth (the step count)
  DAC codec folder                   dac/weights.pth and dac/metadata.pth, which
                                     is all DAC.load_from_folder(package=False)
                                     needs; the torch.package copy of the code is
                                     dropped because the code is in this repo.

The result is what the GitHub release carries. Resuming training from these is not
possible: the optimiser state is gone, by design.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

KEEP = ["model.pth", "tracker.pth", "metadata.pth"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slim_pth(src, dst):
    ck = torch.load(src, map_location="cpu")
    out = {k: v for k, v in ck.items() if k in KEEP}
    assert "model.pth" in out, f"{src} has no model.pth (keys: {list(ck)})"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    step = out.get("tracker.pth", {}).get("step")
    return {"file": dst.name, "from": str(src), "step": step,
            "dropped": sorted(set(ck) - set(out))}


def slim_dac(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "dac").mkdir(exist_ok=True)
    for name in ["weights.pth", "metadata.pth"]:
        shutil.copy(Path(src) / "dac" / name, dst / "dac" / name)
    kw = torch.load(dst / "dac" / "metadata.pth", map_location="cpu").get("kwargs", {})
    return {"file": dst.name + "/dac/{weights,metadata}.pth", "from": str(src), "kwargs": kw,
            "dropped": ["package.pth", "optimizer.pth", "scheduler.pth", "tracker.pth"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    ap.add_argument("--sps_codec")
    ap.add_argument("--sps_bwe")
    ap.add_argument("--dac_bwe")
    ap.add_argument("--dac_codec")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {}
    jobs = [("spectrostream_48khz_500k.pth", a.sps_codec, slim_pth),
            ("bwe_spectrostream_multirate_25k.pth", a.sps_bwe, slim_pth),
            ("bwe_dac_multirate_25k.pth", a.dac_bwe, slim_pth),
            ("dac_48khz_12cb", a.dac_codec, slim_dac)]
    for name, src, fn in jobs:
        if not src:
            continue
        info = fn(Path(src), out / name)
        target = out / name if fn is slim_pth else out / name / "dac" / "weights.pth"
        info["bytes"] = target.stat().st_size
        info["md5"] = md5(target)
        manifest[name] = info
        print(f"{name:40s} {info['bytes'] / 1e6:8.1f} MB  md5 {info['md5']}")

    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
