# Ghostline RL Experiment Log

This log records the learning story for the first-lap RL milestone. It is meant
to stay useful for both engineering and portfolio writing: what changed, what was
trained, what artifacts were produced, and what the model learned or failed to
learn.

## Provenance Rules

From 2026-06-20 onward, every serious training run should keep these artifacts:

- Model weights: `runs/rl/models/*.zip`
- Training manifest: `runs/rl/models/*.manifest.json`
- Eval rollout artifacts: `runs/rl/evals/<experiment_id>/*.json`
- Eval summary JSONL: `runs/rl/evals/<experiment_id>.jsonl`
- Analytics report: `runs/rl/analysis/ranked_runs.md`
- Visual report: `runs/rl/analysis/index.html`

The manifest records algorithm, seed, timesteps, PPO hyperparameters, env config,
reward weights, action adapter, physics fingerprint, and package versions. Rollout
artifacts additionally record the exact reward config, action adapter, physics
identity, replay actions, and cached frames.

Use descriptive experiment ids:

```text
b7_4_ppo_seed3_200k_reward_v2_wallprox
```

Run analytics after every eval:

```bash
python -m momentum_lab.rl.analytics --root runs/rl --top 30 --group-by model,policy --output runs/rl/analysis/ranked_runs.md --visual-dir runs/rl/analysis --visual-limit 30
```

The ranked report includes reward-driver columns (`top +`, `top -`) and a reward
component table per model group. This is the first place to look when a policy has
a good score but still drives badly: it shows which criteria paid for the bad
behavior and which penalties were too small to stop it.

## Current Reward Versions

### `reward_v1`

Original smoke reward.

- `progress_scale`: 2.0
- `checkpoint_bonus`: 5.0
- `finish_bonus`: 25.0
- `time_penalty`: -0.01
- `wall_hit_penalty`: -2.0
- `wall_scrape_penalty_per_second`: -0.5
- No target-speed or heading-alignment shaping.
- No wall-proximity shaping.

### `reward_first_lap_v1` early discrete profile

Used by `b7_4_ppo_seed1_100k`.

- Action adapter: `discrete` (12 actions, includes neutral/brake actions)
- `progress_scale`: 14.0
- `checkpoint_bonus`: 25.0
- `finish_bonus`: 200.0
- `time_penalty`: -0.002
- `wall_hit_penalty`: -5.0
- `wall_scrape_penalty_per_second`: -1.0
- `target_speed_scale`: 0.04
- `heading_alignment_scale`: 0.002

### `reward_first_lap_v1` drive-discrete profile

Used by `b7_4_ppo_seed2_200k`.

- Action adapter: `drive_discrete` (5 throttle-only driving actions)
- `progress_scale`: 20.0
- `checkpoint_bonus`: 50.0
- `finish_bonus`: 500.0
- `time_penalty`: -0.01
- `wall_hit_penalty`: -5.0
- `wall_scrape_penalty_per_second`: -1.0
- `target_speed_scale`: 0.12
- `heading_alignment_scale`: 0.004

### `reward_first_lap_v2` (superseded)

Same no-racing-line target-gate shaping as the drive-discrete v1 profile, with
heavier wall-risk feedback. Diagnosed as counterproductive (see v3 below) and
replaced as the `first_lap_reward()` default.

- Action adapter: `drive_discrete`
- `progress_scale`: 20.0
- `checkpoint_bonus`: 50.0
- `finish_bonus`: 500.0
- `time_penalty`: -0.01
- `wall_hit_penalty`: -12.0
- `wall_scrape_penalty_per_second`: -3.0
- `wall_proximity_penalty`: -0.035
- `wall_proximity_threshold`: 55.0 px shell clearance
- `target_speed_scale`: 0.12
- `heading_alignment_scale`: 0.004

### `reward_first_lap_v3` (current default)

Progress-dominant redesign. The diagnosis that motivated it: scoring all earlier
rollouts under one config showed v2's *continuous* wall terms inverted the run
ranking. A timid agent that crawled to checkpoint 1 and parked in open space
out-scored agents that drove 73-81% of the way around but brushed walls, because
`wall_scrape` (-3/s) and `wall_proximity` (-0.035/step, accumulated every step)
punished the racing line, which necessarily hugs walls. `heading_alignment` also
penalized drifting through corners — the intended fast technique.

v3 makes continuous progress the star (a full stage ≈ a checkpoint), keeps a
discrete impact penalty, and otherwise lets physics make walls costly (a hit kills
speed, which already costs progress reward).

- Action adapter: `drive_discrete`
- `progress_scale`: 140.0 (a full stage of progress now rivals a checkpoint bonus)
- `checkpoint_bonus`: 25.0
- `finish_bonus`: 500.0
- `time_penalty`: -0.002 (was -0.01; the flat tax dwarfed progress and equally
  punished every non-finisher)
- `wall_hit_penalty`: -8.0 (discrete impact only)
- `wall_scrape_penalty_per_second`: -0.5 (light)
- `wall_proximity_penalty`: 0.0 (off — it taxed the racing line)
- `wall_proximity_threshold`: 0.0
- `target_speed_scale`: 0.15
- `heading_alignment_scale`: 0.0 (off — it penalized cornering drift)

Validation trick (no compute): `analytics._reward_component_totals` replays a saved
rollout's action stream and recomputes reward under any config, so a candidate
reward can be ranked against existing rollouts before training. Under v3 the old
rollouts rerank to match human judgment (far-driving > crawl-and-park).

### `reward_time_attack_v1` (stage-2 curriculum)

Used to *fine-tune* a model that already finishes under v3 (warm start), to make
the lap faster. Keeps v3's finish/progress structure so the agent does not trade
the lap away, but raises time pressure. The per-step time penalty only became a
useful signal once the agent reliably finishes: a faster lap is fewer steps, so it
accrues less penalty, turning the flat tax into a real speed gradient.

- Action adapter: `drive_discrete`
- `progress_scale`: 140.0
- `checkpoint_bonus`: 25.0
- `finish_bonus`: 500.0
- `time_penalty`: -0.05 (was -0.002 in v3)
- `wall_hit_penalty`: -8.0
- `wall_scrape_penalty_per_second`: -0.5
- `wall_proximity_penalty`: 0.0
- `target_speed_scale`: 0.25 (was 0.15 in v3)
- `heading_alignment_scale`: 0.0

Warm-start is via `train_ppo(..., init_model_path=<base.zip>)`, which loads the
saved policy and continues training under the new env/reward instead of starting
from a random init. The manifest records `init_model_path` for provenance.

### `reward_time_attack_v2` (superseded)

Fixes a reward/lap-time misalignment found in v1. Diagnosis: the 5.15s finisher
(`b7_6`, 310 steps, 2 wall hits) scored *lower* (795) than a 6.50s clean lap
(`b7_7`, 391 steps, 0 hits, 804), because under v1 avoiding two -8 wall hits (+16)
beat the 81-step time saving (only -4.05 at -0.05/step). The reward preferred a
clean-but-slow line over a faster line that brushed a wall. v2 rebalances so lap
time dominates: the discrete wall penalty is softened (enough to discourage real
crashes but not to outweigh a faster line) and time/speed pressure is raised.

- Action adapter: `drive_discrete`
- `progress_scale`: 140.0
- `checkpoint_bonus`: 25.0
- `finish_bonus`: 500.0
- `time_penalty`: -0.08 (was -0.05 in v1)
- `wall_hit_penalty`: -3.0 (was -8.0 in v1)
- `wall_scrape_penalty_per_second`: -0.5
- `wall_proximity_penalty`: 0.0
- `target_speed_scale`: 0.35 (was 0.25 in v1)
- `heading_alignment_scale`: 0.0

Re-scoring validation (no training; `analytics`-style action-stream replay under
v2): the v1 inversion is gone — `b7_6` 5.15s = **808.48** > `b7_8` 5.72s = **806.26**
> `b7_7` 6.50s = **805.03**. v2 reward is now monotonic in lap time *at equal wall
hits*.

Residual tiebreak (intended, but it capped the fine-tune search): a wall hit is
still worth ~0.6s under v2 (`3 / 0.08 = 37.5` steps), so the reward prefers a
cleaner line that is up to ~0.6s slower. The seed-5 fine-tune's 5.20s / 1-hit line
re-scores to **810.95**, *above* `b7_6`'s 5.15s / 2-hit line (808.48). So v2 does
not actually rank the 5.15s line top — it pulls policies toward ~5.2s clean lines,
which is why no v2 fine-tune beat 5.15s (see runs below).

### `reward_time_attack_v3` (superseded — direct lap-time objective)

v2 fixed the *sign* of the reward/lap-time relationship but two weaknesses kept it
from breaking 5.15s: (1) lap time was only signalled through the per-step
`time_penalty`, a weak/noisy proxy; (2) the -3 wall penalty still valued a hit at
~0.6s, so the reward preferred a cleaner-but-slower line and never ranked the 5.15s
line top. v3 optimizes the real objective directly.

New mechanism — a finish-gated lap-time bonus (`rewards.py`): when the lap closes,
the agent earns `finish_time_bonus_scale` per control step the lap came in under
`finish_time_reference_steps`. This is a strong, direct gradient on total lap time,
not a per-step proxy. `finish_time_reference_steps == 0` disables it (all earlier
versions). New `RewardConfig` fields `finish_time_bonus_scale` /
`finish_time_reference_steps`; new `RewardBreakdown.finish_time` component (shown as
a `finish time` column in the analytics report).

- Action adapter: `drive_discrete`
- `progress_scale`: 140.0
- `checkpoint_bonus`: 25.0
- `finish_bonus`: 500.0
- `finish_time_bonus_scale`: 2.0 (per control step under the reference)
- `finish_time_reference_steps`: 450.0 (≈7.5s, the first valid lap)
- `time_penalty`: -0.08
- `wall_hit_penalty`: -1.5 (softened again; a brush ≈ 0.19s of bonus, so it no
  longer caps lap-time tradeoffs — crashes are discouraged mainly via physics,
  since an impact kills speed → more steps → smaller finish-time bonus)
- `wall_scrape_penalty_per_second`: -0.5
- `target_speed_scale`: 0.35
- `heading_alignment_scale`: 0.0

Re-scoring validation (action-stream replay under v3): now ranks **strictly by lap
time**, and the 5.15s/2-hit line beats the cleaner 5.20s/1-hit line (which v2
preferred) — `b7_6` 5.15s = **1093.48** > seed5 5.20s = **1088.45** > `b7_8` 5.72s =
**1023.26** > `b7_7` 6.50s = **925.03** (finish-time bonus +282 / +276 / +214 / +120
respectively). v3 finally ranks the genuine fastest lap top, so a warm-started
fine-tune's gradient points at *faster*, not *cleaner-slower*.

### `reward_time_attack_v4` (current default — racing efficiency)

v4 responds to the top-run SVG/trace diagnosis: after boost 2, the fastest v3 runs
went too far outside into the right wall before checkpoint 2, losing speed and
taking a suboptimal arc. Rather than making all wall proximity scary again, v4
adds a finish-gated efficiency bonus that normalizes average speed and path length.
Lap time remains the main objective through the direct finish-time bonus.

- Action adapter: `drive_discrete`
- `progress_scale`: 140.0
- `checkpoint_bonus`: 25.0
- `finish_bonus`: 500.0
- `finish_time_bonus_scale`: 2.0
- `finish_time_reference_steps`: 450.0
- `time_penalty`: -0.08
- `wall_hit_penalty`: -6.0
- `wall_scrape_penalty_per_second`: -2.0
- `target_speed_scale`: 0.35
- `heading_alignment_scale`: 0.0
- `avg_speed_bonus_scale`: 400.0
- `avg_speed_reference`: 520.0 px/s
- `path_efficiency_bonus_scale`: 700.0
- `path_distance_reference`: 2500.0 px

Implementation detail: `World.path_distance` is a descriptive, snapshotted metric
accumulated from actual movement for reward/replay accounting only. It does not
feed back into physics.

## Historical Runs

These early runs predate automatic training manifests. Their reward/action
settings are taken from the saved rollout artifacts; PPO hyperparameters are
assumed to be the training CLI defaults unless separately noted:

- `learning_rate`: 3e-4
- `n_steps`: 1024
- `batch_size`: 256
- `gamma`: 0.995
- `ent_coef`: 0.01

| Experiment | Timesteps | Reward | Action Adapter | Deterministic Eval | Best Result | What It Taught Us |
| --- | ---: | --- | --- | --- | --- | --- |
| `b7_4_ppo_seed0_50k` | 50k | `reward_v1` | `discrete` | 3 evals, seeds 10000-10002 | 0/4 checkpoints, stalls near `(544,638)` | Sparse/light reward was not enough; the policy found a repeatable lower-wall failure. |
| `b7_4_ppo_seed1_100k` | 100k | `reward_first_lap_v1` early profile | `discrete` | 3 evals, seeds 10000-10002 | 1/4 checkpoints, stalls near `(926,548)` | Denser target progress got the car through checkpoint 1, but the broad action set still allowed unhelpful stopping/stalling behavior. |
| `b7_4_ppo_seed2_200k` | 200k | `reward_first_lap_v1` drive-discrete profile | `drive_discrete` | 3 evals, seeds 10000-10002 | 1/4 checkpoints, 0.81 progress toward next gate, stalls on right wall near `(1174,293)` | Removing no-op/brake-only actions helped exploration and got much farther, but direct checkpoint seeking drove the car into wall-risk zones. |
| `b7_4_ppo_seed3_200k_reward_v2_wallprox` | 200k | `reward_first_lap_v2` | `drive_discrete` | 3 evals, seeds 10000-10002 | 1/4 checkpoints, 0.73 progress toward next gate, stalls on right wall near `(1178,381)` | Wall-proximity shaping changed the line and produced much more drift time (`1.78s` vs `0.27s`), but it still learned a right-wall attractor rather than a clean checkpoint-2 approach. |

The identical eval rows per model are expected: evaluation used deterministic PPO
actions from the same fixed spawn. The seed was recorded, but the environment had
no randomized reset factor for it to affect.

## reward_first_lap_v3 Runs (first valid lap)

Same PPO defaults, `drive_discrete`, 200k timesteps, 3 deterministic evals each.

| Experiment | Best Result | What It Taught Us |
| --- | --- | --- |
| `b7_5_ppo_seed4_200k_reward_v3` | **4/4 checkpoints — valid lap, ~7.5s, reward 772**, 3 wall brushes, finishes near `(367,501)` | Removing the wall-line tax let the policy commit to the racing line and complete the lap. First valid lap of the project. |
| `b7_5_ppo_seed5_200k_reward_v3` | 1/4 checkpoints, 0.76 progress, stalls right corner near `(1179,357)`, reward 85.8 | The right-hand corner after cp1 is still a local optimum for some seeds, but it now scores a sensible positive instead of a large negative. |
| `b7_5_ppo_seed6_200k_reward_v3` | 1/4 checkpoints, 0.72 progress, stalls right corner near `(1179,373)`, reward 81.2 | Same corner; the marginally-shorter run scores marginally lower (81.2 < 85.8), so reward is now monotonic in distance covered. |

## reward_time_attack_v1 Runs (stage-2: make the lap faster)

Warm-started from the seed-4 v3 finisher; same PPO defaults, `drive_discrete`.

| Experiment | Base | Result | What It Taught Us |
| --- | --- | --- | --- |
| `b7_6_ppo_seed4_finetune_timeattack_150k` | `b7_5_ppo_seed4_200k_reward_v3` | **valid 4/4 lap, 5.15s (down from 7.50s), 2 wall hits** | Warm start + a stronger time/speed reward improved lap time ~31% without losing the lap. Curriculum (finish, then optimize time) beats re-rolling the seed lottery for a known-good base. |
| `b7_7_ppo_seed4_finetune_timeattack_r2_150k` | `b7_6_ppo_seed4_finetune_timeattack_150k` | valid 4/4 lap, 6.50s, **0 wall hits** | A second fine-tune round did **not** keep improving lap time — it regressed to 6.50s but drove a perfectly clean line. Continued fine-tuning is not monotonic; it can trade speed for safety (avoiding -8 wall hits). Keep the run with the genuine best metric (5.15s), which the ranked report does automatically. |

## reward_time_attack_v2 Runs (fix the reward/lap-time misalignment)

Warm-started from the seed-4 v3 finisher's time-attack descendant
(`b7_6_ppo_seed4_finetune_timeattack_150k`, the 5.15s best); same PPO defaults,
`drive_discrete`, 150k timesteps, `--reward time_attack` (now v2).

| Experiment | Variant | Result | What It Taught Us |
| --- | --- | --- | --- |
| `b7_8_ppo_seed4_finetune_timeattack_v2_150k` | seed 4, LR 3e-4 | valid 4/4, **5.72s**, 2 hits | Fine-tuning the 5.15s base under the (correct) v2 reward *regressed* it to 5.72s. The reward gradient points at 5.15s, but the optimizer relaxed off that sharp optimum. |
| `b7_9_ppo_seed4_finetune_timeattack_v2_lr1e4_150k` | seed 4, LR 1e-4 | valid 4/4, **5.72s**, 2 hits — byte-identical to b7_8 | A gentler LR converged to the *exact same* deterministic eval (5.717s / 806.26). LR is not the lever; deterministic arg-max eval also hides sub-threshold weight differences. |
| `b7_10_ppo_seed5_finetune_timeattack_v2_150k` | seed 5, LR 3e-4 | valid 4/4, **5.20s**, 1 hit, reward 810.95 | A different *training* seed found a different line: ~tied on time but cleaner. Exploration (seed), not LR, moves the policy. Re-scores *above* the 5.15s base under v2 (cleaner). |
| `b7_10_ppo_seed7_finetune_timeattack_v2_150k` | seed 7, LR 3e-4 | valid 4/4, **5.22s**, 2 hits, reward 810.12 | Another seed, another ~5.2s line. The v2 search clusters around ~5.2s clean-ish laps and does not break under 5.15s. |

Outcome: the v2 reward fix is correct and validated (re-scoring removes the v1
inversion), but **no v2 fine-tune beat the 5.15s best** (`b7_6`, produced earlier
under v1). The genuine fastest valid lap is still 5.15s; the ranked report keeps it
on top. Two compounding reasons: (a) the 5.15s policy sits on a sharp optimum that
continued training relaxes, and (b) v2's -3 wall penalty still values cleanliness
at ~0.6s/hit, so the reward itself does not rank the 5.15s / 2-hit line first — it
prefers ~5.2s / fewer-hit lines. Beating 5.15s needs the reward to weight raw lap
time over wall cleanliness more aggressively (lower `wall_hit_penalty` toward ~0
and/or more time pressure), at the cost of tolerating wall brushes — or a genuinely
better racing line, which may be near this policy/track's speed ceiling.

## reward_time_attack_v3 Runs (direct lap-time objective)

Warm-started from the 5.15s base (`b7_6_ppo_seed4_finetune_timeattack_150k`); same
PPO defaults, `drive_discrete`, 150k timesteps, `--reward time_attack` (now v3 with
the finish-time bonus).

| Experiment | Variant | Result | What It Taught Us |
| --- | --- | --- | --- |
| `b7_11_ppo_seed7_finetune_timeattack_v3_150k` | seed 7 | **valid 4/4, 4.767s (287 steps), 1 hit, drift 0.55s** — NEW BEST | The direct lap-time bonus broke the ~5.2s wall v2 was stuck at: 7.4% faster than the old 5.15s. Notably the *fastest* run also *drifts the most* (0.55s) — confirming the corner handbrake-turn is the fast technique, not waste. |
| `b7_11_ppo_seed5_finetune_timeattack_v3_150k` | seed 5 | valid 4/4, **5.05s**, 1 hit | Also beats 5.15s. A second seed confirms v3's gradient genuinely pulls toward faster laps, not just one lucky seed. |
| `b7_11_ppo_seed4_finetune_timeattack_v3_150k` | seed 4 | valid 4/4, 6.05s, 2 hits | Seed 4 regressed again — the same seed that struggled under v2. Seed (exploration) remains a real variable even with an aligned reward; keep the best across seeds. |

Outcome: **goal met — a valid lap under 5.15s (4.767s).** Re-scoring confirms v3 ranks
strictly by lap time and puts the 4.77s run top (1141). The win came from changing
the *objective* (a direct finish-time bonus), not from more seeds at a proxy reward.
New best model: `b7_11_ppo_seed7_finetune_timeattack_v3_150k.zip` (run id 116236 in
the analytics report).

## b7_12/b7_13 Runs (warm-start from the 4.77s base + keep-best)

Warm-started from the new 4.767s base
(`b7_11_ppo_seed7_finetune_timeattack_v3_150k`), same PPO defaults,
`drive_discrete`, 150k timesteps, `--reward time_attack`.

First pass: plain final-weight fine-tunes with `--no-report`, then one report regen.
All three stayed valid, which is a strong reliability improvement over fresh seeds,
but none beat the parent. The pattern is now clear: PPO can relax from a sharp fast
line into a cleaner/slower line by the end of training.

| Experiment | Variant | Result | What It Taught Us |
| --- | --- | --- | --- |
| `b7_12_ppo_seed8_finetune_timeattack_v3_150k` | seed 8, final weights | valid 4/4, **5.20s**, 0 hits | Reliable finisher, but slower/cleaner than the parent. |
| `b7_12_ppo_seed9_finetune_timeattack_v3_150k` | seed 9, final weights | valid 4/4, **5.017s**, 0 hits, drift 0.78s | Good lap and high drift, but still slower than 4.767s. |
| `b7_12_ppo_seed10_finetune_timeattack_v3_150k` | seed 10, final weights | valid 4/4, **4.917s**, 0 hits | Best plain b7_12 final policy, but still slower than the parent. |

Second pass: added opt-in checkpoint selection to `momentum_lab.rl.train`.
`--keep-best` periodically runs deterministic eval during training, selects the
best checkpoint by valid lap time (falling back to progress/reward for non-finishers),
records the chosen policy in the training manifest's `selection` block, and writes
the selected checkpoint to `--model-path`. This is deliberately RL-layer-only:
`core/` stays pure and policies still drive the same `Action` stream through
`GhostlineEnv`.

| Experiment | Variant | Result | What It Taught Us |
| --- | --- | --- | --- |
| `b7_13_ppo_seed11_finetune_timeattack_v3_150k_keepbest` | seed 11, `--keep-best --best-eval-freq 5000` | **valid 4/4, 4.750s (286 steps), 1 hit, run id 790117** - NEW BEST | Keep-best found and preserved a genuine 1-step improvement over the 4.767s parent. This directly fixes the "final weights can regress" failure mode. |
| `b7_13_ppo_seed12_finetune_timeattack_v3_150k_keepbest` | seed 12, same | valid 4/4, **4.767s**, 1 hit | Selected the parent-equivalent behavior instead of emitting a slower final policy. That is the reliability win: one run now reliably returns at least the best checkpoint it saw. |

Outcome: **new best is 4.750s** via
`runs/rl/models/b7_13_ppo_seed11_finetune_timeattack_v3_150k_keepbest.zip`
(run id 790117). The improvement is small (one control step), but the bigger change
is operational: serious fine-tunes should use `--keep-best` so the selected model is
the best deterministic checkpoint, not whatever PPO happens to be at the final
update. Verification: focused RL tests passed (15 passed), and
`python -m pytest -q` passed (107 passed).

## reward_time_attack_v4 Runs (racing efficiency: avg speed + path length)

Motivation: the top v3/v3-keepbest policies all exited boost 2 too far outside on
the right-hand wall before checkpoint 2. The old best dropped from roughly 600 px/s
to roughly 215 px/s at the wall, then crossed checkpoint 2 much slower than it
should. Blunt wall punishment alone was risky because older experiments showed that
heavy continuous wall terms make the agent timid and slow.

v4 keeps the direct finish-time bonus as the main objective, then adds a small
finish-gated racing-efficiency bonus:

- `avg_speed`: normalized around 520 px/s, scale 400.
- `path_efficiency`: normalized around 2500 px path distance, scale 700.
- `wall_hit_penalty`: -6.0.
- `wall_scrape_penalty_per_second`: -2.0.

This rewards the sweet spot: carry speed, but do not take a long outside arc just
to keep momentum. Implementation note: `World.path_distance` is a new descriptive
metric accumulated from actual movement and snapshotted/replayed for reward
accounting. It never feeds physics.

Warm-started from the 4.750s keep-best model, `drive_discrete`, 150k timesteps,
`--keep-best --best-eval-freq 5000`.

| Experiment | Variant | Result | What It Taught Us |
| --- | --- | --- | --- |
| `b7_14_ppo_seed13_finetune_timeattack_v4_150k_keepbest` | seed 13 from 4.750s base | valid 4/4, **4.700s**, 0 hits, path 2509 px, run id 985976 | First v4 run immediately beat 4.750s. It still skimmed the wall but carried far more speed through checkpoint 2. |
| `b7_15_ppo_seed14_finetune_timeattack_v4_150k_keepbest` | seed 14 from 4.700s v4 base | **valid 4/4, 4.567s (275 steps), 0 hits, path 2445.8 px, run id 743189** - NEW BEST | The normalized efficiency idea worked: faster lap, shorter path, no hard wall hit. It still runs very close to the outer wall after boost 2, so a future predictive wall-risk term may be useful if this line plateaus. |

Outcome: **new best is 4.567s** via
`runs/rl/models/b7_15_ppo_seed14_finetune_timeattack_v4_150k_keepbest.zip`
(run id 743189). Verification: focused RL/replay tests passed (30 passed), and
`python -m pytest -q` passed (108 passed).

## Analytics: Stable Run Ids + Content Dedup (2026-06-20)

Two report-readability fixes landed with the v3 work:

- **Stable run id.** Each run gets a 6-digit id hashed from its behavioral signature
  (cached trajectory, or outcome metrics for summary-only rows), excluding the no-op
  eval seed. The same rollout always gets the same id (e.g. `run 116236`), so a human
  can quote a run across re-runs; it appears in the ranked table (`run` column), the
  HTML index, and the per-run SVG filename/title.
- **Content dedup.** `rank_runs` now collapses behaviorally-identical rollouts by
  run id (not file path), so the pre-fix deterministic-eval triplicates show as one
  row with an `n` = collapsed-count column instead of N identical rows. Combined with
  the B7.4b eval fix (deterministic eval now writes a single episode), the leaderboard
  is one row per distinct policy behavior.

## Default Results Presentation

Every `python -m momentum_lab.rl.train` run now regenerates the run portfolio under
`runs/rl/analysis/` as part of the flow (disable with `--no-report`):

- `ranked_runs.md` — ranked best-N table (default top 20, single global ranking).
- `top_runs.svg` — overlay of the top trajectories.
- `run_NNN_*.svg` — per-run trajectory SVG (rank = file number).
- `index.html` — links the table + overview + each per-run SVG.

`generate_training_reports()` exposes this for direct calls. Train CLI flags:
`--reward {first_lap,time_attack}`, `--init-model-path` (warm start), `--keep-best`,
`--best-eval-freq`, `--best-model-path`, `--no-report`, `--report-top`,
`--report-visual-limit`. When reporting results to a human, surface the ranked table
and the clickable per-run SVG links so the run can actually be watched, not just
scored.

### Interactive viewer + B7.6b extensions

The interactive replay/story viewer is `python -m momentum_lab.rl.visualizer`
(read-only over `runs/rl`; writes `runs/rl/analysis/replay_viewer/index.html`).
Common forms:

```bash
# Best-N runs + curated story, human run 750982 as the reference line:
python -m momentum_lab.rl.visualizer --top 24 --story progression --reference 750982
# Just one family/run; select by model substring, run id, reward version, etc.:
python -m momentum_lab.rl.visualizer --model b7_15 --reference 750982
```

When continuing the curriculum, two B7.6b options extend what the viewer covers:

- **Trace recorder (intra-run "learning over steps").** Add
  `--save-trace-checkpoints` (off by default; one eval rollout to disk per interval)
  with `--trace-eval-freq` to any train run. It records a deterministic-eval rollout
  at step 0 and every `--trace-eval-freq` steps as a standard artifact tagged
  `policy=ppo_trace`, model `<stem>@<step>`, under `evals/<stem>_trace/`
  (override with `--trace-output-dir` / `--trace-summary-path`). Composes with
  `--keep-best`. Watch the timeline by selecting those artifacts, e.g.:

  ```bash
  python -m momentum_lab.rl.train --reward time_attack \
    --init-model-path runs/rl/models/<base>.zip --keep-best \
    --save-trace-checkpoints --trace-eval-freq 10000 \
    --model-path runs/rl/models/<new>.zip
  python -m momentum_lab.rl.visualizer --model <new-stem> --policy ppo_trace
  ```

- **On-demand re-evaluation.** For a saved model with no rollout artifact yet,
  `--reevaluate <model.zip>` loads it, runs one deterministic eval, writes a standard
  artifact + summary under `<root>/evals/<stem>`, and includes it in the view. The
  env config is rebuilt from the model's manifest (or, if absent, the policy's action
  space — older models used the larger `discrete` adapter). Needs stable-baselines3.

  ```bash
  python -m momentum_lab.rl.visualizer --reevaluate runs/rl/models/<model>.zip
  ```

## Current Interpretation

The reward redesign achieved the first valid lap (seed 4), made run scores
comparable and human-ordered, and a stage-2 time-attack fine-tune then cut the lap
from 7.50s to 5.15s on the same policy lineage. The direct v3 finish-time bonus then
cut it to 4.767s, keep-best checkpoint selection nudged it to 4.750s, and the v4
racing-efficiency reward cut it again to 4.567s with zero wall hits. The remaining
robustness gap is mostly fresh-seed learning: only some fresh policies escape the
right-hand corner after checkpoint 1. The practical answer for time attack is still
to grow the good base, now with `--keep-best` enabled.

This is a good portfolio story: deterministic simulation, action-stream replay,
reward-versioned artifacts, analytics, diagnosis (re-scoring rollouts to expose an
inverted ranking), then a targeted reward fix that both unblocked learning and
corrected the ranking.

Update (2026-06-20, v2): a second misalignment was found and fixed at the reward
level — `reward_time_attack_v1` ranked a clean 6.50s lap above the 5.15s best
because a -8 wall hit outweighed the time saving. `reward_time_attack_v2` (wall hit
-3, time -0.08, target speed 0.35) makes reward monotonic in lap time at equal wall
hits (re-scoring: 5.15s > 5.72s > 6.50s). But four v2 fine-tunes (LR sweep + seed
sweep) all landed at 5.20–5.72s, none under 5.15s: the 5.15s policy is a sharp
optimum continued training relaxes, and v2's residual -3 wall penalty still prefers
~5.2s clean lines over the 5.15s / 2-hit line.

Update (2026-06-20, v3 — goal met): `reward_time_attack_v3` adds a direct
finish-time bonus (`+2.0`/step under a 450-step reference) and softens the wall
penalty to -1.5. This optimizes the real objective (total lap time) instead of the
weak per-step proxy, and it makes the reward rank the genuine fastest lap top.
Warm-started v3 fine-tunes from the 5.15s base produced **4.767s (seed 7)** and
5.05s (seed 5), both beating 5.15s — a 7.4% improvement. New best:
`b7_11_ppo_seed7_finetune_timeattack_v3_150k` (run id 116236). The fastest run also
drifts the most (0.55s), confirming corner drift is the fast technique.

Update (2026-06-20, keep-best): warm-starting from the 4.767s base produced three
valid but slower final-weight b7_12 policies (5.20s, 5.017s, 4.917s), confirming
that final PPO weights can relax away from the fast line. `--keep-best` fixes the
selection problem by saving the best deterministic checkpoint during training. Seed
11 found a new best **4.750s** (`b7_13_ppo_seed11_finetune_timeattack_v3_150k_keepbest`,
run id 790117); seed 12 selected the parent-equivalent 4.767s behavior instead of
regressing.

Update (2026-06-20, v4): normalized average-speed + path-efficiency finish bonuses
were added after inspecting the top SVGs/traces and finding the boost-2 outside-wall
speed dump. Warm-started v4 keep-best runs produced **4.700s** and then **4.567s**
(`b7_15_ppo_seed14_finetune_timeattack_v4_150k_keepbest`, run id 743189), with zero
wall hits and a shorter 2445.8 px path.

Update (2026-06-20, human benchmark): the player's current personal best replay
was imported into the RL analysis portfolio as `Michi` / `human_keyboard`. It is a
valid **4.033s** lap (243 control steps), 0 wall hits, 2
boosts, 0.37s drift, and 2214 px path distance. The regenerated leaderboard ranks
it first as run id 750982, with the per-run SVG at
`runs/rl/analysis/run_001_id750982_Michi_human_keyboard.svg`.
This is now the visible target: the best model (`b7_15`, 4.567s) is about 0.53s
behind the human lap.

Update (2026-06-20, trace comparison): added a read-only trace comparison path to
`momentum_lab.rl.analytics`:

```bash
python -m momentum_lab.rl.analytics --root runs/rl --compare-run 750982 743189 \
  --compare-output runs/rl/analysis/michi_vs_b7_15_trace.md --no-print
```

The report compares checkpoint-sector splits, per-sector path distance,
avg/min/max speed, boost entry/exit speed, drift time, wall-contact windows, and
links the existing trajectory SVGs. Current findings from
`runs/rl/analysis/michi_vs_b7_15_trace.md`:

- Michi is 0.533s faster overall (4.033s vs 4.567s), 32 control steps shorter,
  and 231.6 px shorter on path distance.
- Sector gaps for b7_15: start->CP1 +0.017s, CP1->CP2 +0.117s,
  CP2->CP3 +0.217s, CP3->finish +0.183s.
- The path excess is concentrated after the second checkpoint:
  CP1->CP2 +55.5 px, CP2->CP3 +87.5 px, CP3->finish +98.7 px.
- Boost 2 is the clearest local failure: b7_15 enters it faster than Michi
  (656 px/s vs 551 px/s) but exits much slower (474 px/s vs 588 px/s), a
  -114 px/s exit-speed gap. That supports the line/boost-corner timing diagnosis
  over generic "more acceleration" tuning.
- Both runs have 0 wall-hit summaries and no wall-contact frame windows, so the
  remaining gap is not hard collision avoidance. It is a longer outside arc and
  post-boost speed bleed.

## Next Experiment

Sub-5.15s is **done (4.567s)**. The next target is closing the gap to the 4.033s
human benchmark. Next directions, in order:

1. **Push lap time further from the best v4 base.** Warm-start from the 4.567s v4
   keep-best model (`b7_15` seed 14), keep `--keep-best` enabled, and sweep seeds /
   shorter horizons. Keep the human run in the SVG overlay so improvements are
   judged against the real target, not just the previous model.
2. **Consider a v5 selection/reward lever only if more v4 keep-best plateaus.** The
   trace comparison points at sector path efficiency and boost-2 exit speed, not a
   wall-hit problem. Candidate v5 ideas: finish-gated sector path efficiency,
   finish-gated minimum speed after boost 2, or a keep-best selection tiebreak on
   lap time first, then shorter path / stronger boost-2 exit. Avoid blunt wall
   proximity unless a new trace shows actual contact.
3. **Fresh-seed robustness.** The cp1->cp2 right-hand corner is still open for
   policies that do not warm-start from a finisher.
4. **Teach gate-approach geometry more explicitly**, still without the authored racing
   line. Candidate observation/reward additions:
- signed distance to the active gate plane,
- lateral offset along the active gate segment,
- whether the car is on the correct side of the directed gate,
- alignment with the gate forward normal,
- a small reward for crossing the gate in the correct direction with nonzero speed.

Portfolio framing: the project has moved from "can the agent seek checkpoints?"
to "can the agent learn the geometry of passing through a directed gate without
being handed the racing line?"

## 2026-06-21: v5/v6 push to 4.100s, the 246-tick plateau, and the lookahead pivot

Continuing the warm-start chain past 4.567s:

- v4 keep-best reroll: `b7_15` 4.567s -> `b7_17` 4.367s -> `b7_19` 4.300s
  (diminishing: 0.20s then 0.067s).
- `reward_time_attack_v5` (stronger inside-line path efficiency + a small per-step
  `drift_penalty_per_second`, designed off the `michi_vs_b7_19` sector trace and
  validated with the `_reward_component_totals` replay trick before training):
  4.300s -> 4.217s (`b7_23`) -> **4.100s** (`b7_27`, run `048082`, 0 walls). The
  `michi_vs_b7_27` trace is nearly flat vs the human; residual is CP2->CP3 apex
  speed (543 vs 603).

The 4.100s plateau (the "no improvement after N runs" note):

- After `b7_27`, three more attacks were tried and **every one re-converged to
  exactly 4.100s (246 control ticks)**: a 4th v5 reroll (seeds 27/28/29), a
  higher-entropy/longer exploration probe (ent-coef 0.03, 200k, seeds 30/31/32),
  and `reward_time_attack_v6` (path 1100 / ref 2250 + target_speed 0.50, seeds
  33/34/35). That is ~10 saved models all at 246 ticks. Conclusion: 4.100s is a
  **structural information ceiling**, not a search or reward-shaping problem -- the
  observation only exposes the *next* gate, so the policy cannot plan the apex/exit
  for the corner after it. The human is 4.033s (242 ticks); the whole remaining gap
  is 4 control ticks.
- Leaderboard hygiene: the redundant 4.100s eval artifacts (`b7_25`, `b7_28`..`b7_36`)
  were removed so the ranked report/viewer keep a single 4.100s representative
  (`b7_27`). Their model weights are retained under `runs/rl/models/`. This entry is
  the record that those runs happened and produced no improvement.

Pivot (B7.7): `reward_time_attack_v6` confirmed the limit is informational, so the
lever is richer observations, not reward. Added an opt-in `ObservationConfig.
include_lookahead` (default off; old models keep their dim) exposing the target
gate's forward-normal orientation plus the next-next gate's relative
position/bearing/distance and a `has_next2` flag. Because the observation dimension
changes (41 -> 49), warm-start from `b7_27` cannot transfer, so this needs a fresh
`first_lap -> time_attack` curriculum at the new obs dim (the seed lottery returns).

Lookahead generation results (`b8_*`, obs dim 49):

- **first_lap (fresh):** seeds 40-45 / 200k keep-best -> **6/6 finished a valid lap**
  (`b8_1`..`b8_6`, 5.78s-6.63s). Contrast the old obs dim, where only ~1/3 of fresh
  seeds finished. The richer observation makes learning to complete a lap much more
  reliable -- the first concrete evidence the lookahead features are usable. Fastest
  base `b8_3` (seed42, 5.78s); cleanest `b8_5` (seed44, 0 walls).
- **time_attack v6 (warm-start from finishers):** from `b8_3`/`b8_5`, seeds 50-53 /
  200k -> 4.85s-5.42s (`b8_7`..`b8_10`). Young policies (~400k cumulative steps);
  comparable to where the old chain sat after its first time-attack stage (5.15s).
- **time_attack v6 continuation** (warm-start from `b8_7`/`b8_8`, seeds 60-62 /
  200k) -> 4.633s/4.65s/4.75s, all 0 walls (`b8_11`/`b8_13`/`b8_12`). One sweep cut
  the lookahead best 4.85s -> **4.633s** (`b8_13`, seed62, run `925330`), ~0.2s/sweep,
  matching the old chain's early maturing rate.

State at handoff (session ~325k tokens): overall best lap is still the non-lookahead
`b7_27` 4.100s (run `048082`); the lookahead chain is younger (best `b8_13` 4.633s,
0 walls) and improving ~0.2s/sweep. Open question for the next session: does the
matured lookahead chain break the 4.100s/246-tick ceiling and reach the human's
4.033s? Continue warm-start sweeps from `b8_13` under `reward_time_attack_v6`, then
re-trace vs the human at CP2->CP3 once it nears 4.1s.

## 2026-06-21 (later): gamma 0.997 breaks the ceiling -- lookahead chain ties the human

Continuing the lookahead chain from `b8_13` (4.633s) under v6:

- **Plain reroll stalled.** Warm-starting more seeds from the current best gave only
  4.633 -> 4.600 (`b8_15`), ~0.03s/sweep, with seeds clustering within a tick of the
  parent. The reroll lever was exhausting at ~4.60s -- 0.5s short of even b7's old
  4.100 ceiling. Diagnosis (sector trace + action analysis): the gap was excess path
  (a wide CP2->CP3 / CP3->finish line, +343px), and the action space was *not* the
  limit -- the human's winning lap is full-throttle, bang-bang steering, zero
  brake/lift, i.e. exactly the 5-action `drive_discrete` set. So the human is an
  existence proof that 242 ticks is reachable in the model's own action+observation
  space; the problem was the *line*, a credit-assignment issue.
- **gamma fixed it.** Over a ~245-step lap gamma 0.995 discounts the finish to ~0.30,
  so the policy barely felt how an early-corner line sets up a later apex. Raising
  `gamma` 0.995 -> **0.997** re-accelerated the chain to ~4 ticks/sweep and steadily
  tightened the line (path 2557 -> ~2180). A gamma 0.999 probe was no better (best
  seed 4.433 vs 0.997's 4.383) and was dropped.
- **Chain under gamma 0.997 (warm-start keep-best, v6, lookahead):** `b8_17` 4.467 ->
  `b8_22` 4.383 -> `b8_25` 4.317 -> `b8_30` 4.250 -> `b8_32` 4.183 (path 2227, line
  now matches the human) -> `b8_36` 4.050 (244 ticks, **broke b7's 246-tick wall**) ->
  **`b8_37`/`b8_39` 4.033s (242 ticks, 0 walls, run `233101`) -- ties the human
  exactly, on a tighter line (path 2178 vs 2214).**
- **Sector trace `michi_vs_b8_37`:** flat except CP2->CP3. The policy gains a tick at
  CP1->CP2 (tighter line, -50px) and loses it at CP2->CP3 (apex 548 vs the human's
  603). CP3->finish is now a tie and the mid-chain over-drift resolved on its own
  (b8_37 drift 0.283s < human 0.367s). So the whole residual is one corner's apex
  speed -- the same corner that capped every version (b7_27 543, b8_32 523, b8_37 548).

Conclusion: b7's 4.100/246-tick ceiling was a *combined* information limit (next gate
only -> fixed by `include_lookahead`) and credit-assignment horizon limit (gamma 0.995
-> fixed by 0.997). Both were broken from scratch (self-discovered warm-start chain,
no human seed). B7.7's acceptance bound (<=242 ticks, 0 walls) is met; the strict beat
(<=241) is open and is purely CP2->CP3 apex speed. Tick quantization note: 242 == 242
is a tie on the same step of the 1/60s lap-time staircase (see `BACKLOG.md` B9 on
sub-tick timing); a strict beat needs a full frame (241).

## 2026-06-21 (final): wall sensors reach 239 ticks and close time attack

The last push added wall-distance sensors on top of the successful lookahead +
gamma-0.997 setup. This was not a physics change and did not hand the policy an
authored racing line; it gave the agent more local boundary awareness for the
corner that kept costing apex speed.

- **first_lap with lookahead + wall sensors:** `b10_1`..`b10_6`, seeds 100-105,
  all reached valid laps. The best fresh finisher was already usable as a
  time-attack warm start.
- **time_attack v6 with lookahead + wall sensors + gamma 0.997:** the chain moved
  from `b10_7` 4.837s through `b10_16` 4.088s and `b10_17` 4.010s, then reached
  `b10_18` 3.9719s / 240 ticks / 0 walls.
- **plateau, then one last crack:** `b10_18` through `b10_27` all reported the
  identical best-eval result: 240 steps, `3.9719111127294244` seconds, 0 wall hits,
  reward 1285.43. The final chain run, **`b10_28`**, then found **239 steps /
  `3.9651` seconds / 0 wall hits** (reward 1281.34) and the chain completed.

Interpretation: the original time-attack arc is complete. The project moved from
early wall-stalls to a deterministic, replayable, 239-tick RL lap, with each
major improvement tied to a diagnosable system change: reward alignment, keep-best
selection, racing-efficiency shaping, lookahead observations, longer credit
assignment, and finally wall sensors. Future work should be presentation,
multi-track generalization, or a new research question rather than another
same-recipe Track 1 continuation.
