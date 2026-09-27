# Run bundle contract (v1)

The offline pipeline writes one directory per processed clip. The cloud service publishes it
(`POST /api/runs/publish`), the web UI renders it, L1 adjudicates its crops and the L3 agent
queries it. Schemas live in [`shared/schemas/bundle.py`](../../shared/schemas/bundle.py);
`pipeline.output.bundle.validate_bundle(run_dir)` checks a directory against this contract.

```text
output/<run_id>/
  run_manifest.json      RunManifest       provenance, models, R1–R5 coverage, timings, warnings
  incidents.json         IncidentsFile     confirmed incidents with catalogue-driven text
  tracks_summary.json    TracksSummaryFile per-track timelines (presence, PPE, zones, machinery bands)
  scene.json             SceneCache        restricted zones, edges, traffic zones used by R2/R3/R5
  evidence/<id>.jpg      full frame at the evidence moment, subject highlighted
  evidence/<id>_crop.jpg original-resolution person crop (L1 vision input)
  safety_twin.mp4        1920×1080 H.264 side-by-side
  safety_twin_720p.mp4   web proxy for the 720p player
  report.html            offline report
```

Run `uv run safety-twin process <clip> --skip-video` to get a schema-complete bundle from the
synthetic pipeline today; the real pipeline writes the same files.

## Conventions

- All times are **video seconds** from the start of the clip.
- Distances are **relative worker-height units (WH)** with an interval, never metres.
- Track IDs are `shot:<shot_id>/track:<n>`; the UI label is `n` (`display_label`).
- Incident IDs are deterministic: re-processing the same clip in the same run gives the same ID.

## Incidents (`incidents.json`)

Each `IncidentRecord` is a *confirmed* alert (condition persisted past its debounce).

| Field | Meaning |
|---|---|
| `rule_id`, `severity`, `basis`, `references`, `action_code` | From `regulations/catalogue.v1.yaml` only — the LLM layers may reword, never change them |
| `first_seen_s` / `confirmed_at_s` / `resolved_at_s` | Condition start / debounce met / condition ended |
| `resolution_reason` | `condition_cleared`, `evidence_lost`, `shot_boundary`, `end_of_clip` |
| `observation_text`, `action_text` | Deterministic template text (the FR-7 fallback) |
| `distance`, `relative_band` | R4/R5 only: WH interval and band at the closest approach |
| `helmet`, `vest` | Smoothed PPE state at the evidence frame |
| `adjudication_candidate` | `true` → queue `evidence_crop_path` for L1 (PPE unknown or within 0.10 of the floor) |

## Rule coverage (`run_manifest.json → rule_coverage`)

Always exactly R1–R5, in order, even when a rule could not run. Each entry has a summary
`status`, a `reason_code`, `incident_count`, and `status_counts` (subject-frames per status),
so "mostly inconclusive" is visible even when the summary says `evaluated_clear`.

| Status | Meaning |
|---|---|
| `evaluated_alert` | At least one confirmed incident |
| `evaluated_clear` | Assessed; condition not observed (or never persisted) |
| `inconclusive` | Evidence insufficient — e.g. `inconclusive_distance`, `operating_state_unknown` |
| `not_applicable` | Nothing to assess — e.g. `no_machinery_tracks`, `no_supported_restricted_zone` |
| `unsupported_for_feed` | The feed cannot support the rule — e.g. `geometry_unstable_for_feed` |

## Rule semantics worth knowing

- Timing is on video timestamps: R1/R2 debounce 1.5 s, R3–R5 1.0 s; a gap ≤ 0.25 s pauses the
  debounce, longer resets it; 2.0 s of clear resolves; re-triggering within 30 s reopens the
  same incident instead of creating a duplicate.
- **Inconclusive only when it matters:** a rule reports `inconclusive` only if the unknown could
  change the outcome. A worker whose whole distance interval is beyond the near band is clear
  even when the machine's operating state is unknown; a stationary machine never alerts.
- `unknown` PPE never counts as a negative, and a `no_helmet`/`no_vest` below the 0.70 floor is
  `inconclusive`, never `clear`.
