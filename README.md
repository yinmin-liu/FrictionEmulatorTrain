# FrictionEmulatorTrain

PyTorch training and evaluation tools for a neural-network emulator of the
Weertman basal friction coefficient. The emulator maps two inputs, `C2` and
`vmag`, to `alpha2`, using a multilayer perceptron with two hidden layers and
ReLU activations (64 neurons per hidden layer by default).

The repository includes training samples, saved PyTorch checkpoints, plotting
utilities, and a Python inference helper.

## Repository contents

| Path | Purpose |
| --- | --- |
| `train.py` | Load samples, preprocess, train across seeds, evaluate, and export models. |
| `preprocessing.py` | Input/output transformations and normalization utilities. |
| `accuracy_reports.py` | Accuracy metrics, report serialization, and plots. |
| `plot_training_results.py` | Generate plots from a saved `.npz` report. |
| `compare_training_cases.py` | Compare two saved training reports. |
| `data/weertman_rank_*.csv` | 40 sample files, indexed from 0 to 39. |
| `friction_emulator/` | Python inference helper and three saved `.pt` checkpoints. |

## Installation

Use Python 3.10 or newer. The Python workflow requires NumPy and PyTorch;
Matplotlib is recommended for plots and required by the comparison script.
Run commands from the repository root.

```bash
git clone https://github.com/yinmin-liu/FrictionEmulatorTrain.git
cd FrictionEmulatorTrain
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch matplotlib
```

This installs current packages, not a pinned manuscript environment. Record the
Python and package versions used for each published experiment. GPU execution
requires a compatible PyTorch installation; CPU execution is supported.

## Data format

Each CSV row contains three numeric columns in this order:

```text
C2,vmag,alpha2
```

`C2` is the squared friction parameter `C`, `vmag` is velocity magnitude, and
`alpha2` is the target friction coefficient. The Weertman
relation is `alpha2 = C^2 * |v|^(1/m - 1)`. The exponent `m` is not
an emulator input, so applying the model to a different exponent requires
checking its training assumptions.

The supplied files have numeric rows without a header; the loader also skips
header rows and invalid or non-finite rows. `--n-ranks 40` reads indices 0–39;
the default reads only index 0. Missing files are reported and skipped.
Dataset units, simulation provenance, the value of `m`, and the source ISSM
revision still need to be documented before the manuscript release.

## Train and plot

For a short execution check using one data file:

```bash
mkdir -p outputs/smoke
python train.py --folder data --n-ranks 1 --device cpu \
  --epochs 2 --n-seeds 1 --train-samples 256 --print-every 1 \
  --checkpoint friction_emulator/smoke_model.pt \
  --model-file friction_emulator/smoke_model.txt \
  --report-data friction_emulator/smoke_report.npz
python plot_training_results.py \
  --report-data friction_emulator/smoke_report.npz --plots-dir outputs/smoke/plots
```

For training with all supplied files:

```bash
mkdir -p outputs/full
python train.py --folder data --n-ranks 40 --device auto \
  --epochs 5000 --n-seeds 5 --transform sqrt --sampling weighted --scaling standard \
  --checkpoint friction_emulator/model_sqrt_weighted_standard.pt \
  --model-file friction_emulator/model_sqrt_weighted_standard.txt \
  --report-data friction_emulator/training_report_sqrt_weighted_standard.npz
python plot_training_results.py \
  --report-data friction_emulator/training_report_sqrt_weighted_standard.npz --plots-dir outputs/full/plots
```

These are usage examples, not verified commands for reproducing a particular
paper or the bundled checkpoints. Create output directories before training.
Use `python train.py --help` for all options.

The current defaults:

- Remove samples with `vmag < 5e-8`; disable with `--disable-vmag-filter`.
- Split samples into 70% training, 15% validation, and the remainder testing,
  with split seed 42.
- Select up to 30,000 training rows with `--sampling weighted` (default) or
  `--sampling uniform`. Weighted sampling uses inverse bin counts in
  sqrt(C2)/sqrt(vmag) space, independently of the network transform.
- Apply `--transform sqrt` (default), `log`, or `none` to inputs and target.
- Apply `--scaling standard` (default) to fit mean/std on the transformed,
  selected training rows, or `--scaling fixed` to use zero offsets and fixed
  divisors `[9.05e6, 2.08e-5]` for inputs and `2.09e11` for the target.
  Scaling follows transformation. Fixed divisors S become sqrt(S) for `sqrt`
  and abs(log(S)) for `log`; offsets stay zero. No dataset statistics are fitted.
  With `none`, the original divisors are used unchanged.
- Train seeds `0` through `--n-seeds - 1` and select by validation RMSE.

Training exports a PyTorch checkpoint (weights and preprocessing metadata), a
text model for an external C++ loader, and an `.npz` report for plotting.
Seeds alone do not guarantee identical results across devices and environments.
The checkpoint does not record every run setting; retain the command, console
log, environment, and exact input data alongside published results.

## Compare experiments

After generating two report files, compare them with:

```bash
python compare_training_cases.py \
  --raw-report friction_emulator/training_report_none_uniform_standard.npz \
  --preprocessed-report friction_emulator/training_report_sqrt_weighted_standard.npz \
  --out-dir outputs/comparison
```

Use `bash train_none_uniform_standard.sh` for the first report and
`bash train_sqrt_weighted_standard.sh` for the second. To isolate the network
transform, compare `train_none_weighted_standard.sh` against the latter.
`train_none_weighted_fixed.sh` retains the fixed-scaling weighted baseline.

Scripts and artifacts use `<transform>_<sampling>_<scaling>` names. Without
explicit output paths, `train.py` derives `model_<name>.pt`, `model_<name>.txt`,
and `training_report_<name>.npz` under `friction_emulator/` from its options.
Filtering is separate and enabled in all four scripts. Changing command-line
options does not rename existing artifacts.

The old `--normalization`, `--uniform-sampling`, and
`--disable-joint-sampling` training options are replaced by the explicit options.
To select the entire training pool, set `--train-samples` to at least its size.
Legacy checkpoints (including `mixed`) remain readable. New checkpoints record
`transform`, `sampling`, and `scaling`, plus the legacy `normalization` field and
saved affine arrays for compatibility. The C++ text format retains its existing
`NORMALIZATION` transform tag and affine arrays.

## Python inference

Pass an explicit checkpoint path to override the helper's package-local
default. For a checkpoint generated by the training example:

```python
from friction_emulator.friction_emulator import init_model, predict_alpha2_np

init_model("friction_emulator/model_sqrt_weighted_standard.pt", device="cpu")
alpha2 = predict_alpha2_np([[8.154506659837895e6, 2.775038552821494e-7]])
print(alpha2)
```

Inputs have shape `(N, 2)` in the original data units; predictions have shape
`(N, 1)`. The helper reads preprocessing parameters from the checkpoint.
Only load trusted checkpoints: this helper uses `torch.load` with
`weights_only=False`.

## License

This project is licensed under the [MIT License](LICENSE).
