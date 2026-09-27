# PPE classifier — cross-domain audit on SARD CCTV footage

In-domain metrics are in `ppe_effb0_v1.json` / `ppe_effb0_v2.json` (Ultralytics
Construction-PPE test split at original and CCTV-degraded scales). They pass every v7 §51
PPE gate down to 64 px. This note records what happened on real site footage, which is the
number that matters.

**Set-up.** SARD complementary clip 9 (fixed CCTV, 2560×1440, evaluation-only licence). Its
annotation file indexes a longer recording than the published clip, so no ground truth
could be aligned; the audit is visual, by the developer, on 48 upright tracked workers
(≥ 60 px, sampled every 3 s) plus every R1 alert the pipeline raised. Not a statistic —
a failure-mode inventory.

| Finding | v1 | v2 (promoted) | Mitigation in the pipeline |
|---|---|---|---|
| Yellow helmets called `no_helmet` | 2 / 48 | 0 / 48 | head-band hue augmentation (v2) |
| Red hard hat called `no_helmet` (one worker) | yes | yes (0.86) | YOLOE hard-hat detection vetoes `no_helmet` |
| Crouching / bent-over workers called `no_helmet` | 2 of 3 R1 alerts | — | upright gate: box aspect < 1.6 → `unknown` |
| Woven straw sun hats called `helmet` | several | several | none yet — errs toward a missed R1, not a false one |

**First end-to-end run (v1, no mitigations):** 3 R1 incidents, all false positives (two red
helmets, one bent-over worker in a yellow helmet). This is why the upright gate and the
hard-hat veto exist, and why every R1/R2 incident carries `adjudication_candidate` and an
original-resolution crop for the L1 vision check.

**Known limitation.** The training set has no straw or cloth sun hats; they read as
helmets. Closing this needs labelled site data (not SARD, which is evaluation-only).
