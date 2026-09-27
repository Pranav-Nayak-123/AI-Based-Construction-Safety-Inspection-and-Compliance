# Construction Safety 2.5D Twin

College project: drop a fixed-camera construction-site video, run offline processing, and get a side-by-side MP4 — annotated source on the left, an approximate 2.5D site twin on the right — with R1–R5 hazard cards, evidence frames, and a run report.

This is a heuristic triage tool, not a certified safety system and not legal advice. Distances are relative risk bands, not metres. The twin does not reconstruct hidden geometry or BIM.

**Status:** the offline pipeline runs end to end on real footage — decode → YOLO26 detection → ByteTrack → R1–R5 rule engine → side-by-side MP4 + [run bundle](docs/contracts/run-bundle.md). Still to land: construction fine-tune of the detector, the PPE classifier (R1/R2), and the relative plane (R4/R5 distances). Until then those rules report `unsupported` / `inconclusive` with the reason, never a fake verdict.

Full specification: [`docs/plan/construction-safety-2.5d-twin-plan-v7.md`](docs/plan/construction-safety-2.5d-twin-plan-v7.md)

## Constraints

- Team of four, 12 weeks; developed on Apple M1 Pro (16 GB) and Windows + GTX 1660 Ti (6 GB)
- Input: prerecorded fixed-camera video
- Output: 1920×1080 side-by-side H.264 MP4
- Twin is **2.5D**, not a surveyed reconstruction
- Gemini is optional and cached; demo must run offline
- LLM fine-tuning is a graded experiment, not the rule authority

## Hazard rules

| Id | Meaning |
|---|---|
| R1 | Apparent missing helmet |
| R2 | Apparent missing hi-vis near machinery or traffic |
| R3 | Restricted-zone intrusion |
| R4 | Approximate machinery proximity |
| R5 | Possible missing fall protection near an elevated/open edge |

## Layout

```text
app/           desktop entrypoints
jobs/          background pipeline jobs
pipeline/      intake, vision, scene, mapping, rules, render
twin/          2.5D renderer
shared/        config, coordinates, schemas
llm/           optional fine-tuning experiment
evaluation/    gates and reports
tools/         dataset fetch, CCTV intake, profilers
config/        runtime defaults
data/          local datasets (gitignored)
docs/plan/     v7 specification
```

## Setup

Requires Python 3.11.9, [uv](https://docs.astral.sh/uv/) and ffmpeg on `PATH`.

```bash
uv sync --group dev                    # rules, schemas, tests (no GPU stack)
uv sync --extra vision --group dev     # + torch, ultralytics for the real pipeline
uv run safety-tools fetch-models yolo26l.pt yolo26n-pose.pt
uv run pytest tests
```

The `vision` extra pulls CUDA 12.8 PyTorch on Windows and the default wheels on macOS/Linux; `device: auto` in `config/defaults.yaml` picks CUDA, then Apple MPS, then CPU.

## Running the pipeline

```bash
uv run safety-twin process path/to/clip.mp4                       # → output/<run_id>/
uv run safety-twin process clip.mp4 --site config/sites/yard.json  # with zones/edges (R3, R5)
uv run safety-twin process clip.mp4 --synthetic                    # week-1 placeholder, no models
uv run safety-twin app                                             # local desk UI (synthetic)
```

Each run writes a validated bundle — `run_manifest.json`, `incidents.json`, `tracks_summary.json`, `scene.json`, evidence images, `safety_twin.mp4` (1920×1080) and `safety_twin_720p.mp4` — described in [docs/contracts/run-bundle.md](docs/contracts/run-bundle.md). The manifest records stage timings and a warning for every capability the run lacked.

**Zones and edges (R3/R5)** are drawn once per fixed camera:

```bash
uv run python tools/annotate_site.py clip.mp4 --camera yard-east --time 10
```

Zones only apply to shots whose view still matches the frame they were drawn on.

**Performance** (GTX 1660 Ti, i5-10300H, 173 s 1440p clip): 4m15s end to end with `yolo26l`. Encoding uses NVENC or Intel Quick Sync when available (`video_codec: auto`). Detector trade-offs: `uv run python tools/benchmark_stage1.py clip.mp4 --models yolo26m.pt yolo26l.pt yolo26x.pt`, results in `evaluation/results/`.

Demo clip assignments are frozen in `evaluation/demo_inventory.yaml`.

Or: `./run.sh status`

Copy `.env.example` to `.env` if you later enable Gemini scene bootstrap. The pipeline must still work with that key unset.

## Data

Datasets and working videos are local-only (about 30 GB on the development machine). Do not commit them. See [`data/README.md`](data/README.md).

```bash
uv run python tools/fetch_sard.py --list complementary
uv run python tools/fetch_sard.py complementary 9.mp4 9.txt   # resumable
uv run python tools/cctv.py vet data/source/your_clip.mp4
```

## License

AGPL-3.0-or-later. Ultralytics / YOLO components used for academic open-source work require AGPL.
