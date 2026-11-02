# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-input and same-latent streaming checks for FP32 LiDAR attention tiles."""

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from cosmos_interactive.lidar import LidarDecoder, LidarEncoder, LidarTokenizer


@torch.inference_mode()
def compare(
    checkpoint: str, *, device: str, version: Literal["v1p1", "v1p2"]
) -> dict:
    """Compare both tiles through cache eviction, variable chunks and reset."""
    generator = torch.Generator(device=device).manual_seed(0)
    raw = torch.rand(1, 3, 13, 128, 1800, device=device, generator=generator)
    raw[:, 0] = raw[:, 0] * 95 + 5
    raw[:, 2] = 1
    reference = LidarTokenizer.from_checkpoint(
        checkpoint, device=device, version=version, attention_block_size=64
    )
    candidate = LidarTokenizer.from_checkpoint(
        checkpoint, device=device, version=version, attention_block_size=32
    )
    rows = []
    for counts in ((1,) * 13, (1, 1, 1, 2, 1, 1, 2, 1, 1, 2)):
        encoders = [LidarEncoder(t) for t in (reference, candidate)]
        decoders = [LidarDecoder(t) for t in (reference, candidate)]
        start = 0
        first = None
        for index, count in enumerate((*counts, 1)):
            reset = index == len(counts)
            if reset:
                for stream in (*encoders, *decoders):
                    stream.reset()
                start = 0
            ranges = raw[:, :, start : start + count]
            baseline, actual = [encoder(ranges) for encoder in encoders]
            torch.testing.assert_close(actual, baseline, rtol=1e-3, atol=1e-3)
            expected, output = [decoder(baseline) for decoder in decoders]
            valid = (output[:, 2] > 0) & (expected[:, 2] > 0)
            error = (output[:, 0] - expected[:, 0]).abs()[valid]
            if not error.numel():
                raise AssertionError("No jointly valid rays for range comparison")
            flips = (output[:, 2] != expected[:, 2]).float().mean().item()
            maximum = error.max().item()
            assert maximum < 0.01, f"Same-latent range error exceeds 1 cm: {maximum}"
            assert flips == 0, f"Validity mask changed on synthetic fixture: {flips}"
            assert all(
                k.shape[2] <= 9
                for stream in (*encoders, *decoders)
                for k, _ in stream.cache.values()
            )
            if index == 0:
                first = tuple(t.clone() for t in (baseline, actual, expected, output))
            if reset:
                assert first is not None
                for value, initial in zip((baseline, actual, expected, output), first):
                    torch.testing.assert_close(value, initial, rtol=0, atol=0)
            rows.append(
                {
                    "sweeps": count,
                    "reset": reset,
                    "latent_max_abs": (actual - baseline).abs().max().item(),
                    "range_max_abs_m": maximum,
                    "range_mae_m": error.mean().item(),
                    "validity_disagreement_fraction": flips,
                }
            )
            start += count
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "version": version,
        "device": torch.cuda.get_device_name(device),
        "reference_tile": 64,
        "candidate_tile": 32,
        "precision": "FP32 IEEE; TF32 disabled",
        "input": "seed-0 synthetic metric ranges, not generated model latents",
        "rows": rows,
    }


def main() -> None:
    """Write strict same-input/same-latent checks; fail on a quality threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--version", choices=("v1p1", "v1p2"), default="v1p2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = compare(args.checkpoint, device=args.device, version=args.version)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Passed {len(result['rows'])} chunks including cache eviction and reset; "
        f"max range error {max(r['range_max_abs_m'] for r in result['rows']):.6f} m"
    )


if __name__ == "__main__":
    main()
