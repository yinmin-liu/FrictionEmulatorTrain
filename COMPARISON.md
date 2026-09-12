# Preprocessing comparison

Run from this directory:

```bash
bash train_sqrt_weighted_standard.sh
bash train_none_uniform_standard.sh
bash compare_training.sh
```

Both training runs first remove rows with vmag < 5e-8, then split the remaining
rows 70/15/15 with seed 42. Validation and test sets are identical and contain
only velocities >= 5e-8. This restores the original full-preprocessing split
and sampling sequence for the same source data. Retrain both workflows if their
reports were generated with the intermediate split-before-filter implementation.

- `train_sqrt_weighted_standard.sh`: after filtering and splitting, select up to 30,000 training rows
  using inverse joint-bin counts in square-root input space, then apply square
  roots and fit training means and standard deviations. Joint sampling uses seed 42.
- `train_none_uniform_standard.sh`: after the same filtering and splitting, select up to 30,000 training rows
  uniformly without replacement (seed 42), then fit means and standard deviations
  directly on those raw inputs and targets. No square root or log is applied.

Each run uses the requested 30,000 rows when its training pool is large enough;
otherwise it uses all available rows. Uniform sampling approximately preserves
the filtered distribution. Validation/test normalization uses the corresponding
model's training statistics, never holdout statistics.

Python evaluation and deployment clamp physical predictions to zero after
undoing normalization; the training loss and linear output layer are unchanged.
Deploy the updated friction_emulator/friction_emulator.py with the new checkpoint.
Text export stores the fitted affine statistics; external text-model consumers
must also clamp physical outputs to zero.

The comparison script reads friction_emulator/training_report_none_uniform_standard.npz and
friction_emulator/training_report_sqrt_weighted_standard.npz and rejects mismatched test inputs or
validation/test targets. Plotting requires matplotlib. The fixed-scaling baseline is `train_none_weighted_fixed.sh`. For a comparison
that changes only the nonlinear transform, use `train_none_weighted_standard.sh`
and `train_sqrt_weighted_standard.sh`.
