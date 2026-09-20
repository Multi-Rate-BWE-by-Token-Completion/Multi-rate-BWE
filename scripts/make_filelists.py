"""Build the CSV file lists the training configs read, with the paper's bandwidth filter.

The predictors are trained on a mixture assembled to carry as much high-frequency
energy as possible (paper Sec. 3.1). Jamendo is filtered before entering it: about
half of it is MP3-sourced and brick-wall lowpassed, which teaches a bandwidth
extension model that high frequencies are absent. We keep the files whose
effective bandwidth reaches at least 19 kHz (39% of the corpus).

Effective bandwidth ("cliff") is measured on a 20 s window from the middle of the
file: a Welch-style average power spectrum (n_fft 8192, hop 4096, Hann), smoothed
over 9 bins, and the highest frequency still within 35 dB of the 0.5-2 kHz band
level.

Each source becomes <name>.csv with a single "path" column, which audiotools reads
directly. With --holdout, a source is split into train_<name>.csv and
val_<name>.csv after a seeded shuffle.

Example (the sources used in the paper)
---------------------------------------
    python scripts/make_filelists.py --outdir filelists \
        --source jamendo_hq='/data/jamendo/audio/*/*.mp3' \
        --min-cliff jamendo_hq=19 --holdout jamendo_hq=300 \
        --source train_musdb='/data/musdb18/train/Mixtures/*.wav' \
        --source val_musdb='/data/musdb18/test/Mixtures/*.wav' \
        --source train_medleydb='/data/MedleyDB/train/**/*.wav' \
        --source val_medleydb='/data/MedleyDB/test/**/*.wav' \
        --source train_enst_drums='/data/ENST-drums/train/**/*.wav' \
        --source val_enst_drums='/data/ENST-drums/test/**/*.wav'

Those are the names conf/bwe/*.yml reads: a train_/val_ pair per corpus, with the
Jamendo pair produced by --holdout.
"""
import argparse
import csv
import glob
import os
import random
import warnings
from multiprocessing import Pool

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

WINDOW_S = 20.0
CLIFF_DB = -35.0          # re. the 0.5-2 kHz band level


def cliff_khz(path):
    """Effective bandwidth of one file, in kHz. Returns None if unreadable."""
    try:
        with sf.SoundFile(path) as h:
            sr, n = h.samplerate, h.frames
            if n < sr:
                return None
            want = int(WINDOW_S * sr)
            start = max(0, n // 2 - want // 2)
            h.seek(start)
            y = h.read(frames=min(want, n - start), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if not np.isfinite(y).all() or np.abs(y).max() < 1e-6:
            return None

        n_fft = 8192
        if len(y) < n_fft:
            return None
        step = n_fft // 2
        frames = 1 + (len(y) - n_fft) // step
        win = np.hanning(n_fft).astype(np.float32)
        acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        for i in range(frames):
            seg = y[i * step: i * step + n_fft] * win
            acc += np.abs(np.fft.rfft(seg)) ** 2
        p = acc / max(frames, 1)
        f = np.linspace(0, sr / 2, len(p))

        ref = p[(f > 500) & (f < 2000)].mean() + 1e-20
        db = 10 * np.log10(p / ref + 1e-20)
        k = 9
        sm = np.convolve(db, np.ones(k) / k, mode="same")
        above = np.nonzero(sm > CLIFF_DB)[0]
        return f[above.max()] / 1000.0 if len(above) else 0.0
    except Exception:                                   # unreadable / corrupt
        return None


def _kv(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="NAME=GLOB",
                    help="a named glob of audio files; repeatable")
    ap.add_argument("--min-cliff", action="append", metavar="NAME=KHZ",
                    help="keep only files whose effective bandwidth reaches KHZ "
                         "(the paper uses jamendo_hq=19); repeatable")
    ap.add_argument("--holdout", action="append", metavar="NAME=N",
                    help="split a source into train_/val_ with N validation files")
    ap.add_argument("--outdir", default="filelists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    sources, filters, holdouts = _kv(a.source), _kv(a.min_cliff), _kv(a.holdout)
    os.makedirs(a.outdir, exist_ok=True)

    def write(name, paths):
        path = os.path.join(a.outdir, f"{name}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path"])
            for p in paths:
                w.writerow([p])
        print(f"  {name:<26} {len(paths):>7} files -> {path}")

    for name, pattern in sources.items():
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            print(f"  {name:<26} NO MATCH for {pattern!r}")
            continue
        if name in filters:
            keep_khz = float(filters[name])
            with Pool(a.workers) as pool:
                cliffs = pool.map(cliff_khz, files, chunksize=16)
            kept = [f for f, c in zip(files, cliffs) if c is not None and c >= keep_khz]
            print(f"  {name}: bandwidth filter >= {keep_khz} kHz keeps "
                  f"{len(kept)}/{len(files)} ({100 * len(kept) / len(files):.0f}%)")
            files = kept
        if name in holdouts:
            n_val = int(holdouts[name])
            random.Random(a.seed).shuffle(files)
            write(f"val_{name}", files[:n_val])
            write(f"train_{name}", files[n_val:])
        else:
            write(name, files)


if __name__ == "__main__":
    main()
