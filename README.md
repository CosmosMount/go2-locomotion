# Go2 locomotion and recovery

This repository contains five deliberately small tasks behind one dispatcher:

From the repository root, use the OSA interpreter for every task:

```bash
/home/xyz/micromamba/envs/osa/bin/python run.py pretrain-locomotion --device cuda:0 --headless --output-dir outputs/pretrain/locomotion
/home/xyz/micromamba/envs/osa/bin/python run.py pretrain-recovery --device cuda:0 --headless --output-dir outputs/pretrain/recovery
/home/xyz/micromamba/envs/osa/bin/python run.py pretrain-recoveryrl --device cuda:0 --headless --output-dir outputs/pretrain/recoveryrl
/home/xyz/micromamba/envs/osa/bin/python run.py cem-cpg --device cpu --headless --output-dir outputs/onrobot/cem-cpg
/home/xyz/micromamba/envs/osa/bin/python run.py recovery-rl --device cpu --headless --checkpoint outputs/pretrain/recoveryrl/safe.pt --output-dir outputs/onrobot/recoveryrl
```

An explicit `--output-dir` is cleared and recreated before the task writes new
results. For safety, explicit output directories must be below this repository's
`outputs/` directory; omitting the flag creates a temporary directory instead.

Recovery RL source pretraining is split into three implementation components:
`pretrain/recoveryrl/task.py` owns Task SAC updates,
`pretrain/recoveryrl/safety.py` owns Safety-Q updates, and
`pretrain/recoveryrl/recovery.py` owns dependency freezing and offline MF
recovery updates. The single `pretrain-recoveryrl` command intentionally keeps
Task SAC and Safety-Q paired on the same source interaction stream, then runs
MF recovery only after both source learners and their normalizers are frozen.

No source checkpoint, trained policy, CEM best vector, log, or generated result
is shipped in this repository. A later CEM result can be evaluated explicitly
with `cem-cpg --evaluate-only --checkpoint <run>/best.json`; there is no bundled
default policy.

`onrobot` currently means synchronous target-domain MuJoCo simulation. It does
not contain Unitree SDK/DDS integration and is not a hardware validation claim.

Every backend emits the `go2_raw_46d` raw observation. PPO, Recovery RL and
CPG consume explicit projections of that ABI; commands and controller-internal
state are not hidden inside the raw sensor vector. Each task has exactly one
tracked `config.yaml`; checkpoints and generated results are intentionally not
tracked.

Isaac Lab uses `assets/robots/go2/usd/go2.usd`, while MuJoCo uses the single
mechanical definition `assets/robots/go2/mjcf/go2.xml`. These cannot be one file
because USD and MJCF are different simulator formats. Both MuJoCo tasks now
share that one MJCF; all terrain and scene assets live under `assets/`, and
the shared model exposes the two source-compatible IMU locations explicitly.

## Source provenance

The implementation was distilled from the following working trees on
2026-09-14. They all contained local changes, so the commit identifies the base
while the imported behavior follows the inspected working-tree files.

| Destination | Source | Base commit | Dirty when inspected |
| --- | --- | --- | --- |
| `pretrain/locomotion` | `../go2-rl` | `24591e29b9b9f3ba02ff615ea12653b57c37ac93` | yes |
| `pretrain/recovery` | `../FR-Net` | `85c872d9bef0a74716d8d3fb1e89606c8a42101e` | yes |
| `pretrain/recoveryrl`, `onrobot/recoveryrl` | `../recovery-rl/go2_recovery` | `faf65cd73dd10bbfefb3b8cbfce98b73c6851906` | yes |
| `onrobot/cem_cpg` | `../go2-rl-proprio-stair-feasibility` | `24591e29b9b9f3ba02ff615ea12653b57c37ac93` | yes |

The shared `rl` submodule is pinned by the superproject to
`9a85f348dcf31aca487bb844c319a012790edf20`.
