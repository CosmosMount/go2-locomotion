# Recovery RL ablations

This directory separates two questions that cannot be answered by one training
trace:

1. Does Recovery change the data and outcome of online adaptation?
2. For a frozen checkpoint, does executing Recovery actions reduce failures
   under the same target-domain perturbations?

## 1. Paired training arms

Run both arms from the same source checkpoint and seed. Each command uses the
unchanged 10k collection, 1k critic warmup, and 200k adaptation protocol.
The default target is 1.2 m/s because the transferred actor already runs at
approximately 1.0 m/s in the target MuJoCo environment. This target affects
the reward; it is not an input command in the 46D policy observation.

```bash
/home/xyz/micromamba/envs/osa/bin/python -m ablations.onrobot.recoveryrl.train \
  --arm full \
  --checkpoint outputs/pretrain/recoveryrl/safe.pt \
  --output-dir outputs/ablations/onrobot/recoveryrl/full

/home/xyz/micromamba/envs/osa/bin/python -m ablations.onrobot.recoveryrl.train \
  --arm task-only \
  --checkpoint outputs/pretrain/recoveryrl/safe.pt \
  --output-dir outputs/ablations/onrobot/recoveryrl/task_only
```

`full` is the production intervention rule. `task-only` forces every proposed
task action to execute, so failure transitions enter replay and every safe task
transition can update the learner. The latter is the causal no-Recovery arm,
not merely an evaluation-time switch.

Use `--target-velocity 1.1` on both arms for a more conservative +0.1 m/s
adaptation. Values at or below the 1.0 m/s source reference are rejected so a
run cannot silently become a deceleration experiment again.

Summarize learning curves in fixed interaction windows:

```bash
/home/xyz/micromamba/envs/osa/bin/python -m ablations.onrobot.recoveryrl.summarize \
  --run full=outputs/ablations/onrobot/recoveryrl/full \
  --run task_only=outputs/ablations/onrobot/recoveryrl/task_only \
  --output outputs/ablations/onrobot/recoveryrl/training_summary.json
```

Repeat both arms with at least five paired training seeds before making a paper
claim. A single seed only diagnoses mechanism behavior.

For a quick mechanism pilot, keep the production warmup but truncate adaptation
to the first 20k interactions. Compare it only with the first 20k interactions
of the full run; do not report it as a final safety result:

```bash
/home/xyz/micromamba/envs/osa/bin/python -m ablations.onrobot.recoveryrl.train \
  --arm task-only \
  --checkpoint outputs/pretrain/recoveryrl/safe.pt \
  --adaptation-interactions 20000 \
  --output-dir outputs/ablations/onrobot/recoveryrl/task_only_pilot20k_seed3701
```

## 2. Frozen-policy perturbation test

Evaluate Recovery on and off for each checkpoint. The evaluator applies paired
backward, lateral, roll-rate, and pitch-rate impulses after 100 control steps.

```bash
/home/xyz/micromamba/envs/osa/bin/python -m ablations.onrobot.recoveryrl.evaluate \
  --checkpoint source=outputs/pretrain/recoveryrl/safe.pt \
  --checkpoint full=outputs/ablations/onrobot/recoveryrl/full/final.pt \
  --checkpoint task_only=outputs/ablations/onrobot/recoveryrl/task_only/final.pt \
  --seeds 20 \
  --impulse-scale 0.5 \
  --impulse-scale 1.0 \
  --impulse-scale 1.5 \
  --impulse-scale 2.0 \
  --output-dir outputs/ablations/onrobot/recoveryrl/evaluation
```

Primary metric: failure rate. Secondary metrics: time to failure, Recovery
activation rate/delay, mean reward, and velocity RMSE. Report paired confidence
intervals or a paired bootstrap over scenario/seed rows. `paired_effects.json`
reports both discordant outcomes (`off_failed_on_survived` and the reverse), so
a benefit cannot be hidden by equal aggregate failure rates. Use the impulse
scale sweep as a dose-response curve; do not infer safety from nominal
flat-ground reward alone.
