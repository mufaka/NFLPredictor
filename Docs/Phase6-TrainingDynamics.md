# Phase 6 — Training Dynamics

A reader's guide to what happens inside `python -m nflpredictor.train` and how to read the per-epoch loss curves it emits to `Data/processed/training_loss_curves.parquet`.

This doc and its [companion notebook](../notebooks/phase6_training_dynamics.ipynb) together cover the **diagnostic vocabulary** for training. Picking specific knobs to turn is **Phase 7's** job; this doc teaches you how to look at a loss curve and form a hypothesis worth handing to Phase 7.

---

## 1. The training loop in this project

The training build is a script — `python -m nflpredictor.train` — that runs the full Phase 4 ladder in one process. Trivial rungs (rung 0 mean, rung 1 team_mean) are closed-form and have no training loop. The **learned rungs** (rung 2 linear, rung 3 MLP) share a common loop in `src/nflpredictor/train/train_loop.py`. Per `(combination, fold)` it does roughly this:

1. Seed RNGs from `seed + epoch` so per-epoch shuffles are deterministic per device (TR-TRAIN-03).
2. For each epoch in `1..max_epochs`:
   * **Shuffle** the training games via `torch.randperm(n_train, generator=gen)`. Order is reproducible given the same `seed` and device.
   * **Batch** the shuffled games into chunks of `batch_size` (v1 default: 32). The dataset is small enough (~180 games per S1 train slice; smaller per S3 fold) that an epoch is a handful of batches.
   * For each batch: forward pass through the encoder + model, L1 loss against home/away labels, backward pass, optimizer step.
   * At end of epoch: one forward pass over the full val slice → records `(train_loss, val_loss, val_mae)` for that epoch.

**"Epoch" vs. "batch" in this project's scale.** With ~180 training games and `batch_size = 32`, one epoch is ~6 optimizer steps. That's small. Practical implication: per-epoch loss curves are smoother in spirit than typical deep-learning loss curves (each epoch sees the whole dataset), but per-fold curves on the S3 strategy can be jagged because each fold trains on a much smaller slice (S3 fold k=6 trains on ~85 games — about 3 batches per epoch).

**Sample order is deterministic, not arbitrary.** Phase 4's determinism contract (TR-NF-01) requires that re-running on the same device produce byte-identical outputs. The per-epoch shuffle is therefore seeded as a function of the epoch number, not from a global generator state. Two runs starting from the same checkpoint walk the same sequence of batches.

---

## 2. The loss function

The training loss is **MAE (L1)** — `nn.L1Loss(reduction="mean")` from PyTorch. The model predicts a 2-vector `(pred_home, pred_away)`; the target is the actual `(home_score, away_score)`. The per-batch loss is the mean of `|pred − target|` summed over the two scores and the batch dimension.

Why L1 specifically:

* **It matches the headline metric.** Phase 5's MAE formula and Phase 4's training loss are the same thing modulo per-side averaging (EV-MET-01 ↔ this loss). What you optimize is what you report.
* **It's interpretable.** A train_loss of 7.2 means the model is off by 7.2 points per side on average across the training batches. No squared-error or log-prob translation step.
* **It's robust to outliers.** NFL games include occasional 40-point blowouts. L1 lets those games contribute a bounded gradient instead of a quadratic gradient that would dominate the batch.

The cost is that L1 is not strictly convex everywhere (its gradient is constant in magnitude); for very small residuals the optimizer can oscillate around the minimum rather than settle into it. On a dataset this small, that's a side-show — the bigger constraint is sample size, not loss-surface geometry.

---

## 3. The optimizer and learning rate

The optimizer is **Adam** (`torch.optim.Adam`). Hyperparameters in `Data/raw/training_config.yaml`:

* `lr: 0.001` — fixed learning rate for both `linear` and `mlp` rungs.
* No LR schedule. The LR is constant for the full training run.
* No weight decay configured (Adam's default `weight_decay = 0`).
* No gradient clipping.

Adam's update rule maintains per-parameter running estimates of the first and second moments of the gradient and uses them to scale each parameter's step. Practical implications for reading the curves:

* **The first few epochs are when Adam is most aggressive.** The running averages haven't converged; the LR-scaled steps are large. Expect the train and val losses to fall quickly in epochs 1–5.
* **After the moments stabilize, Adam behaves close to a tuned per-parameter LR.** The remaining trajectory is mostly determined by the loss-surface geometry and the LR magnitude.
* **A fixed LR with no schedule means there's no late-training "fine-tune" phase.** The model trains at the same step size until early stopping fires or `max_epochs` runs out. If the curves are still meaningfully descending when training ends, the LR (or `max_epochs`) is the lever — not the schedule, because there isn't one to adjust.

---

## 4. How "done" is decided today

Phase 4 uses **early stopping on val MAE** (TR-TRAIN-04). Practical mechanics:

* `max_epochs: 200` is the upper bound — training never runs longer.
* After each epoch the trainer recomputes val MAE. The lowest val MAE seen so far is the **running best**.
* If `early_stop_patience: 20` consecutive epochs pass without improvement on the running best, training halts.
* The model used for predictions is the **best-epoch parameters**, not the final-epoch parameters. The training loop snapshots state when the running best updates and restores it before emitting predictions.

This is meaningful for reading curves: the visible end-of-curve val loss is usually *not* the val loss that the prediction parquet was generated from. The relevant epoch is wherever the val loss first hit its minimum — that's the `best_epoch`. The Phase 4 manifest's `training_summaries.<combo>.<slice>.best_epoch` field records it.

**Risks of each tuning direction.** Early stopping with a generous patience traded against `max_epochs`:

* If `early_stop_patience` is too small relative to typical noise in val MAE, the trainer halts during a normal stagnation that would have resumed descent. The fix is a larger patience, not a higher `max_epochs`.
* If `max_epochs` is small enough that training stops *before* early stopping has fired, the best-epoch params may still be on the descending trajectory and you're under-training. Look at the loss curve: does val loss appear to still be falling at the cutoff?
* If `max_epochs` is very large and `early_stop_patience` is also large, you can spend epochs on noise after the model has effectively converged. Wall-clock cost only — the predictions are unchanged because they use `best_epoch`.

**"No early stopping" is not the current mode**, despite the appearance of an `early_stop_patience` field that could in principle be set to a value larger than `max_epochs`. The right reading is: *early stopping is active and conservative*. Disabling it entirely is a knob change for Phase 7, not a setting.

---

## 5. Reading the train↔val gap

The four canonical patterns you'll see when plotting `train_loss` and `val_loss` together (the [companion notebook](../notebooks/phase6_training_dynamics.ipynb) renders one subplot per learned combination):

1. **Both still decreasing at the cutoff** — *still learning*. The model hasn't reached its capacity ceiling; if compute budget permits, more epochs (or a larger `max_epochs`) would likely improve val. On the CUDA machine, this is the easiest diagnosis to confirm: re-run with `max_epochs: 400` and see whether val continues to descend. If `early_stop_patience` ended training, lengthening the patience accomplishes the same thing.

2. **Train low, val high (and val rising)** — *overfit*. The model has memorized the training data and is now harming generalization. Levers: smaller model (`hidden_dim` for the MLP rung), more regularization (`dropout`), shorter training (smaller `early_stop_patience` *or* smaller `max_epochs`), or more data (a future Phase 1 expansion to multiple seasons). On a 200-ish-game dataset this is the easiest failure mode to fall into for rung 3.

3. **Both flat from epoch 1** — *optimization stuck*. The optimizer isn't moving the loss. Almost always an LR issue (too small to escape the init plateau) or a feature-encoding bug (the model is seeing all-zero or all-sentinel inputs). Quick sanity check: does the **trivial-rung baseline beat this combination's val MAE** in `metrics_headline.json`? If yes, the learned rung is broken, not just slow.

4. **Train bottomed, val high and flat** — *underfit with respect to the val distribution*. The model has fit the training data as much as it can but the residual val error reflects irreducible mismatch between train and val. On this project that's often the early-season-vs.-late-season distribution shift (Phase 3 S1 splits at week 12). Levers: more capacity (larger `hidden_dim`, larger `embedding_dim`), richer features (Phase 1 Madden columns we didn't include in v1), or the **S3 expanding-window protocol** which is specifically designed to amortize this shift.

In practice, real curves don't fit exactly one pattern. The useful question is: **which of the four does this combination *most look like*, and what's the lever closest to that diagnosis?**

---

## 6. Which knob to reach for

A small mapping from diagnostic pattern to the field in `Data/raw/training_config.yaml` you'd touch first. Picking a value is **Phase 7**'s problem; this table is about the entry point.

| Diagnostic (from §5)           | First knob                                       | Why                                                                                                                                  |
| ------------------------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| Still learning at cutoff       | `max_epochs` ↑ *or* `early_stop_patience` ↑      | Give training more room. Validate with a re-run; if val still descends, push further.                                                |
| Overfit (val rising)           | `dropout` ↑ *or* `hidden_dim` ↓                  | Reduce effective capacity. Dropout is the gentler intervention; `hidden_dim` is the heavier one.                                     |
| Overfit (specific to MLP rung) | `mlp.early_stop_patience` ↓                      | If the MLP is overshooting its best epoch by a lot, tighten the patience so the snapshot is taken sooner.                            |
| Optimization stuck             | `lr` ↑ (carefully) — then inspect feature inputs | Try doubling the LR; if curves still flat, the issue is upstream — verify Phase 2 features for the affected combination aren't degenerate. |
| Underfit                       | `hidden_dim` ↑ *or* `embedding_dim` ↑            | More capacity. Note the embedding-dim per-vocab keys in `embedding_dims:` — each entry is a separate knob.                           |
| Asymmetric home/away error     | (no training-config lever — Phase 2 issue)       | If the loss is symmetric but the per-side errors aren't, the input encoding has dropped a signal. Look at `column_to_vocab_key`.     |
| S3 worse than S1 on the same combination | `seed` (try 2–3 values)                | Could be small-fold variance; if a re-seed flips the conclusion, the gap was noise. If it doesn't, the gap is real.                  |

The table is intentionally **not exhaustive** — Phase 7 will deepen it. The current goal is to give you a starting point when you open `training_config.yaml`.

---

## What is *not* covered in Phase 6

This doc and the rest of Phase 6 are about **reading what's already there** — manifests, loss curves, breakdowns, plots — and forming hypotheses. The following are deliberately Phase 7's job:

* **Feature attribution.** Which Madden columns or starter slots are pulling the prediction in which direction? Saliency-style analysis (gradients × inputs, permutation importance, etc.).
* **Ablation results.** Systematically removing feature families (turn off weather, turn off officials, turn off the slot-level archetype) and measuring the MAE delta.
* **Per-team error analysis.** Beyond the `by_team.parquet` summary — drilling into specific teams that the model consistently misses on (e.g., is the model bad on teams that frequently change QBs mid-season?).
* **Hyperparameter sweeps.** Picking values for `hidden_dim`, `lr`, `dropout`, etc. The "first knob" table above tells you where to start; Phase 7 owns the sweep protocol.
* **Loss-function alternatives.** Switching from L1 to Huber, MSE, or a calibration-aware objective.

When in doubt: Phase 6 reads. Phase 7 turns knobs.

---

## See also

* [Phase 6 — Pipeline Walkthrough](./Phase6-Walkthrough.md) — the one-game trace from raw to plots.
* [Phase 6 — Reading the Outputs](./Phase6-ReadingTheOutputs.md) — what each Phase 5 artifact says and how to interpret it.
* [Spec-Phase4-BaselineLadder.md](./Spec-Phase4-BaselineLadder.md) — the canonical training contract (TR-TRAIN-*, TR-LC-*).
* [`notebooks/phase6_training_dynamics.ipynb`](../notebooks/phase6_training_dynamics.ipynb) — companion notebook rendering one subplot per learned combination.
