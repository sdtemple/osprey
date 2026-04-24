#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import Compose, Gain, PitchShift


COLOR_TO_BETA = {
    "violet": -2.0,
    "blue": -1.0,
    "white": 0.0,
    "pink": 1.0,
    "brown": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a colored-noise audio dataset with random color, amplitude, and pitch variation."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Where generated WAV files are written")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Output sample rate in Hz")
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=30.0,
        help="Target clip length in seconds; actual value is evenly adjusted to exactly fill total time",
    )
    parser.add_argument(
        "--total-minutes",
        type=float,
        default=45.0,
        help="Total simulated time in minutes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def resolve_segmentation(args: argparse.Namespace) -> tuple[int, float, float]:
    target_segment_seconds = float(args.segment_seconds)
    total_seconds = float(args.total_minutes) * 60.0
    n_clips = max(1, int(round(total_seconds / target_segment_seconds)))
    segment_seconds = total_seconds / n_clips
    return n_clips, segment_seconds, total_seconds


def generate_colored_noise(num_samples: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Generate colored noise where power spectral density follows 1/f^beta."""
    freqs = np.fft.rfftfreq(num_samples, d=1.0)
    freqs[0] = 1.0
    scale = np.power(freqs, -beta / 2.0)

    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=freqs.shape[0])
    spectrum = scale * (np.cos(random_phases) + 1j * np.sin(random_phases))
    signal = np.fft.irfft(spectrum, n=num_samples)
    signal = signal.astype(np.float32)

    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal /= peak
    return signal


def main() -> None:
    args = parse_args()
    if args.segment_seconds <= 0:
        raise ValueError("--segment-seconds must be > 0")
    if args.total_minutes <= 0:
        raise ValueError("--total-minutes must be > 0")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    n_clips, segment_seconds, simulated_seconds = resolve_segmentation(args)
    samples_per_clip = int(round(segment_seconds * args.sample_rate))

    rng = np.random.default_rng(args.seed)
    augmenter = Compose(
        [
            Gain(min_gain_db=-20.0, max_gain_db=-1.0, p=1.0),
            PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1.0),
        ]
    )
    colors = list(COLOR_TO_BETA.keys())

    for idx in range(n_clips):
        color = str(rng.choice(colors))
        beta = COLOR_TO_BETA[color]
        raw = generate_colored_noise(samples_per_clip, beta, rng)

        # Additional random amplitude scaling before audiomentations transforms.
        raw *= float(rng.uniform(0.25, 1.0))
        augmented = augmenter(samples=raw, sample_rate=args.sample_rate)

        augmented = np.clip(augmented, -1.0, 1.0).astype(np.float32)
        out_name = f"noise_{idx:05d}_{color}.wav"
        sf.write(output_dir / out_name, augmented, args.sample_rate)

    print(
        f"Generated {n_clips} files x {segment_seconds:.2f}s "
        f"(~{simulated_seconds / 60.0:.2f} minutes) in {output_dir}"
    )


if __name__ == "__main__":
    main()