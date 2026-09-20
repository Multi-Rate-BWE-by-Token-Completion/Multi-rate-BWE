"""Build the evaluation sets: full-band references and their band-limited partners.

Two steps, both reproducible from a corpus folder:

  excerpts   draw N excerpts of a fixed duration from a corpus and write them as
             sample_<i>_sr48000.wav, with metadata.csv recording the source file.
             The paper uses 1000 excerpts of 2.5 s from the 50 MUSDB18 test
             tracks (20 per track), and 1000 from OrchideaSOL out of domain.

  rates      for each input rate, write the band-limited partner
             sample_<i>_sr<rate>.wav next to a copy of the reference.
             Band-limiting is a resample down, so the partner carries the
             resampling filter's transition band rather than a brick-wall edge;
             the synthesis script resamples it back up to 48 kHz.

Examples
--------
    # 1000 excerpts of 2.5 s from the MUSDB18 test split
    python scripts/make_testsets.py excerpts \
        --source /path/to/musdb18/test/Mixtures \
        --output samples/input_25s_48 --n 1000 --duration 2.5

    # band-limited partners at every input rate of the paper
    python scripts/make_testsets.py rates \
        --src samples/input_25s_48 --dst_prefix samples/input_25s 8000 16000 24000 32000
"""
import argparse
import csv
import shutil
from pathlib import Path

from audiotools import AudioSignal
from audiotools.core import util
from audiotools.data.datasets import AudioDataset, AudioLoader


def excerpts(a):
    """Draw excerpts with audiotools' salient-excerpt sampler.

    The excerpt of item i is drawn with util.random_state(i), so the set is
    reproducible from the corpus and the item count alone.
    """
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    loader = AudioLoader(sources=[a.source])
    dataset = AudioDataset(loader, a.sample_rate, duration=a.duration, n_examples=a.n,
                           loudness_cutoff=a.loudness_cutoff, num_channels=1)
    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "original"])
        w.writeheader()
        for i in range(a.n):
            item = dataset[i]
            sig = item["signal"]
            path = out / f"sample_{i}_sr{a.sample_rate}.wav"
            sig.write(path)
            w.writerow({"path": str(path), "original": str(sig.path_to_input_file)})
            if i % 100 == 0:
                print(f"  {i}/{a.n}", flush=True)
    print(f"{out}: {a.n} excerpts of {a.duration}s at {a.sample_rate} Hz", flush=True)


def rates(a):
    src = Path(a.src)
    refs = [p for p in sorted(util.find_audio(str(src))) if "sr48000" in p.name]
    assert refs, f"no sr48000 references under {src}"
    for cut in a.rates:
        dst = Path(f"{a.dst_prefix}_{cut // 1000}-48")
        dst.mkdir(parents=True, exist_ok=True)
        for p in refs:
            s = AudioSignal(str(p))
            s.write(dst / p.name)                                    # full-band reference
            lo = s.clone().resample(cut)
            lo.write(dst / p.name.replace("sr48000", f"sr{cut}"))    # band-limited partner
        meta = src / "metadata.csv"
        if meta.exists():
            shutil.copy(meta, dst / "metadata.csv")
        print(f"{dst}: {len(refs)} pairs at {cut} Hz", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("excerpts", help="draw excerpts from a corpus")
    e.add_argument("--source", required=True, help="folder of audio files")
    e.add_argument("--output", required=True)
    e.add_argument("--n", type=int, default=1000)
    e.add_argument("--duration", type=float, default=2.5)
    e.add_argument("--sample_rate", type=int, default=48000)
    e.add_argument("--loudness_cutoff", type=float, default=-40.0)
    e.set_defaults(func=excerpts)

    r = sub.add_parser("rates", help="write band-limited partners at each rate")
    r.add_argument("--src", required=True, help="folder of sr48000 references")
    r.add_argument("--dst_prefix", default="samples/input_25s")
    r.add_argument("rates", nargs="+", type=int)
    r.set_defaults(func=rates)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
