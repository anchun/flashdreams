<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optional LiDAR generation

> Prepared-scene LiDAR rendering uses independently written Apache-2.0 geometry
> and projection implementations. See the [test specification](tests/specs/lidar_geometry_projection.md).

LiDAR is generated alongside RGB from the driven trajectory. **Use RGB-only
for responsive interaction on one GPU**; joint RGB+LiDAR is much slower.
Both modes use the same version 2 scene bundle, which includes `lidar.json`.

**Live default:** Interactive Drive selects the LiDAR clock from the chunk size:
7.5 Hz for `--chunk-frames 1`, which avoids the expensive 1, 1, 2 sweep cycle,
and the original 10 Hz for `--chunk-frames 3`, which produces four evenly spaced
sweeps per 400 ms chunk. Pass `--lidar-fps 7.5` or `--lidar-fps 10` to override
the selection explicitly. This changes scan acquisition intervals and model
timing, not just output publication. Trained-model quality at 7.5 Hz remains
unqualified; startup warns about it. Recorded batch generation is unchanged.

Prepared-scene geometry starts at the format's compatibility spacing of 2 cm.
If the map plus the largest actor frame would exceed the point caps, preparation
doubles the spacing until it fits and logs the effective value. The selected
spacing stays fixed throughout the rollout, retains every source primitive and
actor, and keeps the 60-million-point scene cap. Coarser sampling can change
LiDAR returns; 2 cm is a compatibility default, not a measured speed/quality
optimum. Invalid geometry still fails validation; retries are bounded to reject
inputs that cannot fit even at extremely coarse spacing.

## Download model and codec

Complete the [demo setup](README.md#install). Enable Interactive Drive LiDAR
with `--lidar`. The application downloads **and loads** the matching V1.2 codec
from `lidar_vae` in the same Sim-Auto HF repository/revision as the transformer.
Both assets use normal HF authentication and caching during initialization;
no separate codec download or `LIDAR_CHECKPOINT` variable is needed.
The native codec uses vendored vLLM-Omni PyTorch FlexAttention, without NATTEN.

The V1.2 checkpoint is
[`lidar_vae/diffusion_pytorch_model.safetensors`](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/blob/ccde59bc545540cbbb38389a9c18e2347ba679bf/lidar_vae/diffusion_pytorch_model.safetensors)
in `nvidia/Cosmos3-Nano-Sim-Auto`. The repository required private access when
verified; the QA account needs access and a normal HF login:

```bash
uv run --no-sync hf auth login
```

The default model URL uses `main`. For reproducible QA, pin the model URL; the
codec automatically follows the same revision:

```bash
export COSMOS_INTERACTIVE_CHECKPOINT="https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/resolve/ccde59bc545540cbbb38389a9c18e2347ba679bf/model.safetensors.index.json"
```

These model weights use the model card's OpenMDW-1.1 terms, separately from the
Apache/MIT inference source. At the pinned revision the 596,033,216-byte codec
file has SHA-256 `8fc88321323327fcb885d9ff9642f1317c81a96c1758ee99cac7b98f41e557f3`.
All 295 network tensors and both latent statistics exactly match the earlier
`v1p2-iter_000030000.pt` checkpoint used for GPU validation. No conversion or
training checkout is needed.

A standalone CPU check exercises the same default checkpoint selection:

```bash
uv run --no-sync python - <<'PYCODE'
from cosmos_interactive.checkpoint_load import resolve_checkpoint_source
from cosmos_interactive.lidar import LidarTokenizer
from cosmos3_nano_sim_auto.checkpoint import resolve_lidar_checkpoint
from cosmos3_nano_sim_auto.config import DEFAULT_CHECKPOINT
source = resolve_checkpoint_source(None, default=DEFAULT_CHECKPOINT)
checkpoint = resolve_lidar_checkpoint(source)
LidarTokenizer.from_checkpoint(checkpoint, version="v1p2", device="cpu")
print("LiDAR V1.2 checkpoint loaded:", checkpoint)
PYCODE
```

A local transformer snapshot uses its sibling
`lidar_vae/diffusion_pytorch_model.safetensors` automatically. For other exports
or custom codecs, pass `--lidar-checkpoint /path/to/codec.safetensors`; this also
enables Interactive Drive LiDAR. An explicit missing path fails rather than
selecting different weights. V1.1 requires an explicit codec and
`--lidar-tokenizer-version v1p1`.

For recorded batch generation, `--lidar-sample` enables LiDAR with matching RGB
`--sample` inputs; the codec is selected automatically unless overridden with
`--lidar-checkpoint`. The resolved path is also used by the decode worker.

## Run in a browser

From the repository root:

```bash
uv run --no-sync flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode webrtc --host 0.0.0.0 --port 8080 \
  --backpressure-mode block --presentation-mode continuous -- \
  --camera camera_front_wide_120fov \
  --world-model-seed 0 --start-frame 0 --compile-denoise off \
  --lidar \
  --culled-lidar-renderer \
  --lidar-view point-cloud
```

Range culling is enabled by default with LiDAR; the explicit flag above makes
the setting visible. Use `--no-culled-lidar-renderer` for the baseline renderer.
The default retains the complete source map and the 2 cm
compatibility spacing, subject to the capacity fallback above. Each sweep
projects only points within its possible 100 m return range, padded for sensor
motion. Near occluders are retained before the 5 m training-validity threshold.
The moving-shutter solve converges each point independently; it can differ from
the baseline renderer's global-mean stopping rule. The additional
`--lidar-codec-tf32 --lidar-contiguous-linears` settings remain **Useful opt-in**
options. TF32 trades matmul precision
for speed; contiguous codec projections retain FP32 and change layout/kernel
dispatch. Use
`--no-culled-lidar-renderer --no-lidar-codec-tf32 --no-lidar-contiguous-linears`
for the previous baseline configuration.

The Slurm launcher's `LIDAR=1` preset already selects these three options. With
four GPUs per node, the 32-GPU launch is:

```bash
sbatch -A <account> -p <partition> --nodes 8 \
  --export ALL,MODE=drive,LIDAR=1,NPROC=4 \
  integrations_v2/cosmos3_nano_sim_auto/probes/launch.sbatch
```

At 10 Hz, up to four ranks render a full chunk's four sweeps, preferably on ranks without RGB
cameras, while camera ranks prepare RGB controls. One owner gathers and encodes
the LiDAR scans. Joint inference consumes completed controls for the current
supplied poses; how much preparation is hidden depends on the hardware and
collective timing. Validate overlap on the target 32-GPU mesh before treating
rendering as absent from the critical path.

The earlier 10 Hz single-GPU validation on RTX PRO 6000 Blackwell, Torch 2.12.1+cu130, used the
complete residential scene, one front-wide camera plus LiDAR, HF V1.2 codec,
seed 0, start frame 0, neutral coasting and `--compile-denoise off`. Each profile
ran in a fresh process with warm checkpoint/compiler caches for 17 chunks;
the first four were excluded, leaving 13 measured 12-frame chunks. Nsight
profiling was collected separately from these synchronized timing runs:

| Configuration | LiDAR render median | Total chunk median / p90 |
| --- | --- | --- |
| Baseline | 1.444 s | 4.396 / 4.537 s |
| Local culling, FP32 codec | 0.287 s | 3.267 / 3.347 s |
| Local culling, TF32 and contiguous codec projections | 0.281 s | 3.055 / 3.124 s |

All three produced 193-frame RGB+LiDAR videos. On six matched stationary/moving
source poses, local culling matched the full-map per-point solver's output maps
exactly. A separate same-latent codec comparison measured TF32/contiguous versus
baseline: worst chunk p99 range difference 2.06 cm, maximum difference 28.42 cm,
and 77 validity flips among 4,838,400 cells (0.0016%). These are one-hardware
control/codec checks and a coasting video smoke test; broader field quality and
32-GPU throughput/overlap remain unvalidated.

For seven views, replace the `--camera` value with the seven-camera list in
[README.md](README.md#run-in-a-browser). Open `http://<server-address>:8080/`.
Choose range or intensity at launch with `--lidar-view range` or
`--lidar-view intensity`; HDMAP displays the control scan.
Output timestamps use 30 FPS for RGB and the selected 7.5 or 10 Hz for LiDAR;
inference throughput depends on the hardware.

## Short interactive chunks

Interactive Drive accepts `--chunk-frames 1 --lidar` as an experimental opt-in.
The checkpoint was trained with three RGB latents per chunk, which remains the
default quality setting. This restriction was enforced by the live backend;
the joint rollout already selects LiDAR sweeps by timestamp and the causal codec
accepts variable sweep counts.

At 30 RGB FPS, one RGB latent advances four output frames, so the control step
covers `4 / 30 = 0.1333` seconds of simulated time instead of `12 / 30 = 0.4`.
At the automatic clock, each short chunk has one scan at 7.5 Hz, while each
full three-latent chunk has four scans at 10 Hz. At 7.5 Hz, scan endpoints are
RGB frames 4, 8, 12, ... and each scan spans the trailing four RGB intervals
(133.3 ms). At 10 Hz, scan endpoints occur every three RGB intervals and the
four scans in a full chunk end at 100, 200, 300 and 400 ms. Actor selection and
the rolling-shutter model use the same clock, without changing scene metadata.

With `--chunk-frames 1 --lidar-fps 10`, after the instantaneous opening scan,
the sweep counts per short chunk cycle through **1, 1, 2**. This is why the
automatic short-chunk default is 7.5 Hz:

| Output RGB frame indices | Completed LiDAR sweep times (seconds) |
| --- | --- |
| 0 | 0.0 (opening) |
| 1-4 | 0.1 |
| 5-8 | 0.2 |
| 9-12 | 0.3, 0.4 |

Each scan uses its trailing driven pose interval, including retained poses from
the preceding chunk when needed. Presentation holds the most recent completed
sweep at each RGB timestamp; it does not display a future scan or interpolate a
new scan. The 7.5 Hz option changes generation and geometry consistently; discarding scans
after 10 Hz inference would not remove its compute cost. For `chunk_frames=3`
with 10 Hz, the first generated chunk covers RGB frames 1-12 and completes
scans at 0.1, 0.2, 0.3 and 0.4 seconds.

For four GPUs and three cameras:

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 --no-python \
  flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode websocket --host 0.0.0.0 --port 60089 -- \
  --chunk-frames 1 --denoise-steps 3 --compile-denoise off \
  --camera camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov \
  --lidar
```

Validation status: **Deferred** for production quality. Latency measurements
are recorded below; 7.5 Hz is the requested latency-oriented live default, not
a quality-qualified replacement for the trained sensor clock.
CPU regressions cover the joint model with tiny random weights, temporal
metadata and bounded history, live trailing scan intervals, independent
generated/control presentation, distributed variable-size scan transfer, and
the native codec's streaming cache and reset. They do not establish trained
checkpoint quality. Four-GPU CUDA/NCCL coasting smoke runs completed for both
clocks; broader target meshes and interactive driving quality remain unvalidated.

Before deploying, compare `--chunk-frames 1` against `3` with the same scene,
cameras, checkpoint, denoise steps and controlled driving inputs. Check RGB and
LiDAR consistency during turns/braking and after restart or scene changes.
Measure steady-state median/p90 chunk time and input-to-visible latency after
warmup, using the existing stage timings and profiling tools. Shorter simulated
steps do **not** guarantee a threefold wall-clock improvement: codec, rendering,
communication and per-chunk overhead still apply. More autoregressive steps and
fewer denoise steps can change quality. Start with compilation off; the 10 Hz
short path alternates between one- and two-sweep shapes.

This opt-in applies to Interactive Drive only. The recorded batch application's
three-latent LiDAR validation is unchanged.

### Measuring one-sweep versus two-sweep chunks

Use the existing runtime stats sink to record every chunk, not the console log:
fast console chunks can be suppressed. A 67-step run includes the opening
frame, six additional warmup chunks and 60 measured chunks (40 one-sweep and
20 two-sweep). Keep the checkpoint revision, scene, seed, camera count, denoise
steps and parallel mesh fixed when comparing runs.

```bash
uv run --no-sync torchrun --standalone --nproc_per_node=4 --no-python \
  flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode null --total-model-steps 67 --timeout 1800 \
  --stats-path artifacts/lidar-10hz-stats.json -- \
  --chunk-frames 1 --denoise-steps 3 --compile-denoise off \
  --world-model-seed 0 \
  --camera camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov \
  --lidar --lidar-fps 10

uv run --no-sync python \
  integrations_v2/cosmos3_nano_sim_auto/probes/lidar_chunk_stats.py \
  artifacts/lidar-10hz-stats.json --warmup 7 \
  --output artifacts/lidar-10hz-summary.json
```

The summary reports count, median and inclusive p90 for all chunks and each
sweep-count group, plus two-sweep minus one-sweep timing differences. It uses
only the first result channel to avoid double-counting presentation outputs,
and fails if either group is missing. For a 7.5 Hz run, change the input/output
paths, use `--lidar-fps 7.5`, and add `--allow-single-sweep` to the analyzer;
its two-minus-one difference is then empty. `total_ms` is the backend's CUDA-event
span; `backend_wall_ms` includes host work from backend entry through output
assembly, but excludes the subsequent app/UI/browser/network path. Stage
measurements overlap across ranks and streams: do not sum them to attribute
latency. `joint_denoise_ms` also includes both modalities, not just LiDAR.

Null output removes browser transport and user input. This measures neutral
coasting backend performance, not closed-loop input-to-visible latency. Collect
a separate short Nsight Systems trace to inspect actual overlap and stalls;
do not use profiler-instrumented chunk times as the headline benchmark. A
consistent two-sweep slowdown is evidence for testing a lower-frequency mode,
not proof that changing the sensor clock preserves trained-model quality.

### Four-L20 short-chunk measurements

Measured on 2026-11-02 with four NVIDIA L20 GPUs (46,068 MiB each), driver
570.86.15, PyTorch 2.11.0+cu128, CUDA 12.8 and cuDNN 91900. Base commit:
`cc3d245dc0159d7ff2fd4dfc01147da55172b816`, plus the live-clock/timing changes
described here. Both fresh-process runs used the command above with the local
cached model/codec snapshot `c3fa1378e93710242501a6288d7c9e7c9084a580`,
residential scene, 832x480 per view, front-wide/cross-right/cross-left cameras,
seed 0, start frame 0, three denoise steps, BF16 model and FP32 codec,
TP=2/CP=2, range culling on, TF32/contiguous codec projections off,
compilation off and neutral coasting with the normal UI composition enabled.
The checkpoint and compiler caches were warm. Each run had 67 steps; the
opening and next six steps were discarded, leaving 60 measured chunks.

| Clock/group | Samples | Backend CUDA span median / p90 (ms) | Backend wall median / p90 (ms) |
| --- | ---: | ---: | ---: |
| 10 Hz, all | 60 | 1614.5 / 2340.8 | 1616.0 / 2341.9 |
| 10 Hz, one sweep | 40 | 1595.2 / 1625.3 | 1596.5 / 1626.4 |
| 10 Hz, two sweeps | 20 | 2337.5 / 2368.1 | 2339.3 / 2369.3 |
| 7.5 Hz, one sweep | 60 | 1574.2 / 1607.8 | 1575.5 / 1608.9 |

Two-sweep chunks added 742.3 ms to median backend CUDA time (+46.5%). Their
joint denoise median was 881.8 ms versus 622.4 ms for one sweep; owner codec
encode/decode medians were 359.9/357.8 ms versus 196.6/196.7 ms.
Geometry rendering medians were effectively unchanged (163.4 versus 164.6 ms)
because scans were split across ranks. These stages overlap: their differences
must not be added as independent critical-path costs.

At 7.5 Hz, overall backend CUDA p90 decreased by 31.3% versus the 10 Hz run.
This is a backend latency result, **not** a browser closed-loop measurement or
proof of real-time performance: a 133 ms simulated step still took about
1.6 seconds of wall time. The short Nsight attempt stalled before the first
chunk and was stopped; no timeline-based attribution is claimed. Null-output
runs do not provide visual quality evidence or cover turns, braking, restarts
and scene changes. No speed or quality claim is made for default chunk=3.
An additional seven-step four-GPU smoke run with no chunk/frequency overrides
confirmed the new defaults: an opening scan, then twelve RGB frames and three
LiDAR scans per step, followed by coordinated NCCL shutdown.

The codec accepts local `.pt` and `.safetensors` checkpoints. Use `.pt` files
only from trusted sources because loading their metadata requires unpickling.

### FP32 codec attention tiles on L20

Validation status: **Recommended default** for the measured four-L20 workload.
The default `--lidar-attention-block-size 32` reduces the spatial attention tile
from 64x64 to 32x32. This retains FP32 IEEE dot products (including P @ V), the
same neighborhood mask, weights, sensor clock, denoise schedule and nine-sweep
cache. It does not enable TF32, drop scans or change resolution. The 64-token
sparse-mask metadata is unchanged; FlexAttention subdivides it into smaller
compute tiles. Per-codec selection is included in both weight and compiled-runner
cache keys, so a restart or another codec does not silently inherit the setting.
Use `--lidar-attention-block-size 64` to restore the baseline.

A separate short Nsight Systems capture on the LiDAR-owner rank succeeded in
this optimization pass. In one captured encode range, nine sparse-attention
kernels consumed about 134 ms of a 189 ms codec interval. Contiguous FP32
projections were not the bottleneck. Smaller tiles substantially reduce this
cost on L20; no equivalent speed claim is made for other GPUs.

The following matched fresh-process runs used physical GPUs 4,5,6,7, the same
stack, pinned model/codec, scene, resolution, camera order and seed as the
four-L20 measurements above, but **two denoise steps** and **7.5 Hz**.
Both used TP2/CP2, FP32 codec, BF16 model, culling on, compilation off,
TF32/contiguous projections off, neutral coasting and normal UI composition
with null transport and synchronized stats. The candidate ran before a repeated
reference run; both contained 67 steps, discarding the opening plus six warmup
steps, leaving 60 samples each. Other users' work occupied GPUs 0-3; those GPUs
were not used by these experiments, but shared CPU/I/O contention is possible.

| Metric | Tile 64 median / p90 (ms) | Tile 32 median / p90 (ms) |
| --- | ---: | ---: |
| Backend CUDA span | 1366.4 / 1403.7 | 1115.4 / 1143.9 |
| Backend wall | 1367.7 / 1404.8 | 1116.7 / 1145.4 |
| `model_ms` | 1130.5 / 1162.9 | 994.4 / 1028.5 |
| Owner LiDAR encode | 196.6 / 197.1 | 77.3 / 77.6 |
| Owner LiDAR decode | 196.6 / 197.2 | 77.2 / 77.5 |
| Joint denoise | 408.9 / 411.1 | 408.4 / 411.0 |

Median backend span improved **18.4%**, p90 **18.5%**. The first backend chunk
took 7.44 s with tile 32 versus 4.74 s with tile 64 on these warm-cache runs;
first-use specialization and startup costs are not steady-state gains.
`model_ms` includes LiDAR control preparation but excludes generated-LiDAR
decode; it is not the transformer's pure denoise time. Stage measurements
overlap and must not be added together. Geometry-render variation also
contributes to total-time differences. These are backend measurements, not
browser input-to-visible latency or real-time simulation.

For comparison, 37-step screening runs (30 measured after warmup) gave backend
median/p90 of 1358.4/1372.5 ms for the initial reference, 1360.3/1375.5 ms with
`--lidar-contiguous-linears`, and 1392.5/1422.7 ms with TP1/CP4 via
`COSMOS3_NANO_SIM_AUTO_TENSOR_PARALLEL=1`. Both alternatives were **Rejected**
for this workload; keep the default TP2/CP2 mesh and projection layout.
The single-codec tile-16 experiment was slower than tile 32 and is not exposed.

To reproduce the backend measurement, use the short-chunk measurement command
above with `CUDA_VISIBLE_DEVICES=4,5,6,7`, `--denoise-steps 2`,
`--lidar-fps 7.5` and the default attention tile 32. Repeat with tile 64,
distinct stats paths and `--allow-single-sweep` in the analyzer. Pin the local
checkpoint/scene and use the same synchronization flags for both runs.
For interactive use:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --no-sync torchrun \
  --standalone --nproc_per_node=4 --no-python \
  flashdreams-run-v2 interactive-drive-cosmos3-nano-sim-auto \
  --mode websocket --host 0.0.0.0 --port 60089 -- \
  --chunk-frames 1 --denoise-steps 2 \
  --camera camera_front_wide_120fov,camera_cross_right_120fov,camera_cross_left_120fov \
  --lidar --lidar-attention-block-size 32
```

Changing the tiling changes floating-point reduction order, not precision;
outputs are not bitwise identical. The checked-in probe compares the real
V1.2 codec on identical synthetic metric ranges and decodes **identical**
reference latents through both implementations:

```bash
CUDA_VISIBLE_DEVICES=7 uv run --no-sync python \
  integrations_v2/cosmos3_nano_sim_auto/probes/lidar_codec_tiles.py \
  /path/to/lidar_vae/diffusion_pytorch_model.safetensors \
  --output artifacts/lidar-codec-tile-quality.json
```

The probe asserts encoder closeness (`rtol=atol=1e-3`), less than 1 cm maximum
same-latent range error, identical validity masks, bounded caches and exact
per-path reset replay. It passed 25 chunks across all-one-sweep and mixed
one/two-sweep schedules, including eviction beyond nine cached sweeps and reset.
Observed maximum latent difference was 2.82e-5, maximum jointly valid range
error 2.98 mm, worst-chunk range MAE 0.022 mm and validity disagreement zero.
This fixture is not generated-model latent coverage or a driving-quality
qualification; boundary validity decisions and long autoregressive trajectories
can diverge. Keep the option experimental until turns, braking, scene changes
and generated-LiDAR quality have been evaluated.

A separate bounded 20-step run of the WebSocket command above completed with
tile 32, returned HTTP 200 for its client page and shut down NCCL cleanly.
Browser rendering and interactive control response were not validated.
