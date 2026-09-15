# Recovery RL adaptation-stage pilot: seed 3701, 20k interactions

## Protocol

- Source checkpoint: `outputs/pretrain/recoveryrl/safe.pt`
- Source SHA-256: `8e7fa233037def741b0936e54094a9b53464a351a319b9d0252ded2413a5a145`
- Shared warmup: 10,000 collection interactions and 1,000 reward-critic updates
- Compared adaptation prefix: 20,000 interactions
- Full trace: `outputs/onrobot/recoveryrl/training.jsonl` (first 20k adaptation rows)
- Task-only trace: `outputs/ablations/onrobot/recoveryrl/task_only_pilot20k_seed3701/training.jsonl`
- The two warmup traces are exactly equal row by row.
- Both runs retain frozen-source digest
  `31caf2efac71dbaf5011a33d6e70584684fa164ac74ff61fb60be15c48c31f36`.

## Raw comparison

| Arm | Failures | Failures / 10k | Completed episodes | Episode failure fraction | Mean reward | Mean risk | Recovery rate | Task updates | Replay size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Recovery | 11 | 5.5 | 45 | 24.44% | 0.001694 | 0.181418 | 16.51% | 16,698 | 26,698 |
| Task-only | 14 | 7.0 | 48 | 29.17% | 0.007775 | 0.192339 | 0% | 19,986 | 30,000 |
| Full minus Task-only | -3 | -1.5 | -3 | -4.72 pp | -0.006081 | -0.010921 | +16.51 pp | -3,288 | -3,302 |

The observed interaction-level failure reduction is 21.43% (`(14-11)/14`).

## Fixed 5k windows

| Adaptation window | Full failures | Task-only failures | Full reward | Task-only reward | Full Recovery rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–5k | 0 | 9 | -0.009296 | -0.003503 | 14.28% |
| 5–10k | 8 | 5 | -0.000211 | 0.009603 | 20.80% |
| 10–15k | 3 | 0 | 0.006322 | 0.012174 | 16.68% |
| 15–20k | 0 | 0 | 0.009960 | 0.012825 | 14.28% |

Task-only failures occur at interactions 495, 767, 833, 898, 973, 1038,
1121, 1249, 2019, 7939, 8197, 8505, 8615, and 9303. Full has no failure in
the first 5k; its failures occur at 6965, 7080, 7459, 7603, 7818, 8093, 8801,
9091, 10766, 11098, and 11408.

## Interpretation

This pilot supports a limited mechanism claim: Recovery shields the initially
unadapted task actor from early failures and reduces observed failures over the
first 20k interactions. It does not support a final efficacy claim because it
uses one training seed, the trajectories cease to be paired after the first
intervention, and both arms reach zero failures in the final 5k window.

The safety benefit has a learning-efficiency cost. Full excludes 3,302
intervention transitions from replay and performs 3,288 fewer task updates,
while Task-only obtains higher reward throughout adaptation. This is consistent
with Recovery trading early protection for slower task-policy adaptation.

## Next experiment

Run the full 200k protocol for at least five paired seeds. Treat adaptation-stage
failure count/rate as the primary endpoint, and report reward plus update count
as costs. Then evaluate each final checkpoint under paired impulse-strength
sweeps. Do not combine `pretrain/recovery` until this baseline effect is
replicated.
