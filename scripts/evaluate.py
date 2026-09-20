"""Objective metrics for a folder of outputs, against the reference test set.

Writes metrics.csv inside the output folder, one row per file, with the five
metrics reported in the paper:

    visqol-audio  ViSQOL v3, audio mode (the main metric)
    mel           multi-resolution mel distance: two resolutions, window lengths
                  {2048, 512} and {150, 80} mel bins, log10 of the squared
                  magnitude plus a linear term, L1, summed over resolutions
    stft          multi-resolution STFT distance, windows {2048, 512}
    waveform      waveform L1
    sisdr         scale-invariant SDR, stored as a LOSS (negate for dB)

Only the sr48000 rows are the reconstructions; the band-limited rows are written
by the synthesis script so that every reference has a partner, and are ignored by
the analysis.

Usage:
    python scripts/evaluate.py --input samples/input_25s_16-48 \
        --output samples/out_16k/predicted_k1 --n_proc 32
"""
import csv
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

import argbind
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from audiotools.ml.decorators import Tracker

sys.path.append(os.getcwd())
from sps.nn.loss import L1Loss, MelSpectrogramLoss_DAC, MultiScaleSTFTLoss, SISDRLoss  # noqa: E402


@dataclass
class State:
    stft_loss: MultiScaleSTFTLoss
    mel_loss: MelSpectrogramLoss_DAC
    waveform_loss: L1Loss
    sisdr_loss: SISDRLoss


def get_metrics(signal_path, recons_path, state):
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)
    x = signal.clone()
    y = recons.clone()
    return {
        "mel": state.mel_loss(x, y),
        "stft": state.stft_loss(x, y),
        "waveform": state.waveform_loss(x, y),
        "sisdr": state.sisdr_loss(x, y),
        "visqol-audio": metrics.quality.visqol(x, y),
        "recons_path": str(recons_path),
        "path": str(signal.path_to_file),
    }


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "samples/input_25s_16-48",
    output: str = "samples/output",
    n_proc: int = 8,
    sample_rates: List[int] = [16000, 48000],
):
    tracker = Tracker()
    state = State(
        waveform_loss=L1Loss(),
        stft_loss=MultiScaleSTFTLoss(),
        mel_loss=MelSpectrogramLoss_DAC(),
        sisdr_loss=SISDRLoss(),
    )

    audio_files = util.find_audio(input)
    audio_files.sort()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    @tracker.track("metrics", len(audio_files))
    def record(future, writer):
        o = future.result()
        for k, v in o.items():
            if torch.is_tensor(v):
                o[k] = v.item()
        writer.writerow(o)
        o.pop("path")
        return o

    futures = []
    with tracker.live:
        with open(output / "metrics.csv", "w") as csvfile:
            with ProcessPoolExecutor(n_proc, mp.get_context("fork")) as pool:
                for f in audio_files:
                    if not any(f"sr{sr}" in f.name for sr in sample_rates):
                        continue
                    futures.append(pool.submit(get_metrics, f, output / f.name, state))

                keys = list(futures[0].result().keys())
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()
                for future in futures:
                    record(future, writer)

        tracker.done("test", f"N={len(futures)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
