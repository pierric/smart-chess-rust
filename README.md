# Building a Chess AI from scratch

[![Rust](https://github.com/pierric/smart-chess-rust/actions/workflows/rust.yml/badge.svg)](https://github.com/pierric/smart-chess-rust/actions/workflows/rust.yml)

## Build

Python part

```bash
poetry install
```

Rust part

```bash
poetry shell
source .env
cargo build
```

### notes

If Cargo added multiple versions of ndarray and results in some type error, see this
[link](https://github.com/PyO3/rust-numpy?tab=readme-ov-file#dependency-on-ndarray)
for the explanation and solution.

## Scripts

It works by iterating the following steps.

1. Run self-play to gather a few thousands of plays (2K seems very sufficient) with the last model.
2. Sample the plays (adding a few old plays a well if any). Keeping positive, negative, draw cases in a close amount.
3. Train a new model with the sampled plays. Watch the loss reduction on the validation set.
4. Run evaluation against the previous model (both as white and as black).

### self play

```bash
N=500 P=2 ../scripts/run_batch -c path-to-last-pt --cpuct 2.5 --num-steps 150 --rollout-num 180 --temperature-switch 4 --temperature 0
```

- Generate 500 self-play traces in parallel runs (max 2)
- start with temperate 1, and switch to 0 after step 4.
- maximal 150 steps. Force draw after then.
- roll out 180 times in each step.
- cpuct is set to 2.5

### evaluation

To compare two models, it is convenient to let them compete. There is a script to facilitate this task.

```bash
ROLLOUT=20 TEMPERATURE_SWITCH=8 W=<path-to-checkpoint-1> B=<path-to-checkpoint-2> ../scripts/leader-board
```

1. Model often has high chance to fall into the same openings when the Dirichlet noise is turned off and the temperate is 0. Particularly in the early models, such openings may favor either WHITE or BLACK regardless of the performance of the model. Therefore, set the temperature to 1 and switch to 0 after the 8th play is a way to mitigate the issue.
2. It seems effective to use a small number for simulation (20) to see there is a signal of improvement then switch to some bigger number, e.g. 100 to confirm it.
