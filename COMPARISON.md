# Preprocessing comparison

Run from this directory:

```bash
bash train.sh
bash train_uniform.sh
bash compare_training.sh
```

Both training runs load the same original valid rows and split them 70/15/15
with seed 42 before applying training-only filtering or sampling. Validation
and test sets retain the original distribution, including low velocities.
Both models must be retrained; old full-preprocessing reports use a different split.

- `train.sh`: filter training velocities below 5e-8, select up to 30,000 rows
  using inverse joint-bin counts in square-root input space, then apply square
  roots and fit training means and standard deviations.
- `train_uniform.sh`: no velocity filter, select up to 30,000 training rows
  uniformly without replacement (seed 42), then fit means and standard deviations
  directly on those raw inputs and targets. No square root or log is applied.

Each run uses the requested 30,000 rows when its training pool is large enough;
otherwise it uses all available rows. Uniform sampling approximately preserves
the original distribution. Validation/test normalization uses the corresponding
model's training statistics, never holdout statistics.

Python evaluation and deployment clamp physical predictions to zero after
undoing normalization; the training loss and linear output layer are unchanged.
Deploy the updated friction_emulator/friction_emulator.py with the new checkpoint.
Text export stores the fitted affine statistics; external text-model consumers
must also clamp physical outputs to zero.

The comparison script reads training_report_uniform_standardized.npz and
training_report_full_preprocessing.npz and rejects mismatched test inputs or
validation/test targets. Plotting requires matplotlib. Existing scaling-only
scripts and outputs remain available as a separate experiment.
