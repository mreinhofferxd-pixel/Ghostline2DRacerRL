# Ghostline Portfolio Brief

Use this as the handoff brief for adding Ghostline to the personal website.

## Selected Work Card

Title: Ghostline

Category: Reinforcement learning lab

Years: 2026

Status: Complete study

Short description:

A deterministic 2D racer where I trained an RL driver from wall-stalls to a sub-4 second lap, then built the replay and analytics stack to prove how it learned.

Metric:

3.965s final lap

Metric caption:

from a 7.5s first finish to a clean 239-tick lap

Tags:

- Reinforcement learning
- Deterministic sim
- PPO
- Reward design
- Replay tooling
- Python

CTA:

Case study ->

Suggested href:

`/work/ghostline`

## Case Study Headline

Ghostline: teaching a tiny racer to chase a human lap

## Case Study Intro

Ghostline started as a small top-down time-trial racer, but the real project became the system around it: deterministic physics, action-stream replays, reward-versioned training runs, and tools that could explain why a policy improved or got stuck.

The model did not simply get "more training." It learned through a sequence of engineering bets: finish the lap first, make the reward match the stopwatch, preserve the best checkpoint instead of trusting final weights, add enough track geometry for the policy to plan a corner ahead, then lengthen credit assignment so early lines could pay off at the finish.

By the end, the best policy ran a clean 3.965s lap on Track 1. The interesting part is not just the number. It is that every jump had an artifact trail: replays, leaderboards, sector traces, reward components, and a viewer that shows the policy evolving from stalled attempts into a fast line.

## Story Beats

Use these as sections or replay-viewer captions.

1. Early policies stalled before they could complete the lap. The first lesson was that sparse "go fast" rewards were not enough.
2. The first real breakthrough was a valid 7.5s lap. That proved the sim, action stream, checkpoints, and training loop could all close the loop.
3. Warm-start time attack cut the lap into the low 5s, but exposed a ranking bug: the reward sometimes preferred cleaner slow laps over faster messy ones.
4. A direct finish-time reward made the objective match the stopwatch, and the policy broke under 5s.
5. Keep-best checkpoint selection fixed final-weight regression. The best policy during training mattered more than the last policy after training.
6. Racing-efficiency shaping shortened the line and removed wall hits, pushing the agent toward the human route.
7. The 4.1s plateau showed the observation was missing context. The model could see the next gate, but not enough of the next corner.
8. Lookahead observations made fresh lap learning reliable again.
9. Raising gamma to 0.997 lengthened credit assignment and broke the old 246-tick ceiling, bringing the model to the human benchmark.
10. Wall-sensor training sat at 3.972s for repeated seeds, then the final continuation cracked one more tick and landed at 3.965s.

## Interactive Elements

Development links on this machine:

- Replay story viewer: `file:///C:/Users/reinh/Documents/2DRacerRL/runs/rl/analysis/replay_viewer/index.html`
- Leaderboard/static analysis: `file:///C:/Users/reinh/Documents/2DRacerRL/runs/rl/analysis/index.html`

For the production website, copy the generated analysis assets into the site's public/static folder and link or iframe them from `/work/ghostline`. Do not depend on `file://` paths in production.

Suggested page modules:

- Hero: title, one-sentence premise, final lap metric.
- "The loop": deterministic sim -> action stream -> replay -> reward -> training -> analytics.
- "What changed": a compact timeline using the Story Beats above.
- Embedded replay viewer: default to the progression story and human reference.
- Deduped leaderboard: show unique leaderboard outcomes, keeping equal-time runs only when their rewards differ.
- Closing note: the project ends as a complete time-attack study, not an unfinished training chase.

## Tone

Make it feel like an engineering story, not a model leaderboard flex. The cool part is that the system could explain itself: why the policy stalled, why reward versions changed, why a plateau was information-limited, and why the final result was believable.
