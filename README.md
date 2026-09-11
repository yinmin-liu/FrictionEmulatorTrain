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
  --epochs 5000 --n-seeds 5 --normalization sqrt \
  --checkpoint friction_emulator/models_full_preprocessing.pt \
  --model-file friction_emulator/models_full_preprocessing.txt \
  --report-data friction_emulator/training_report_full_preprocessing.npz
python plot_training_results.py \
  --report-data friction_emulator/training_report_full_preprocessing.npz --plots-dir outputs/full/plots
```

These are usage examples, not verified commands for reproducing a particular
paper or the bundled checkpoints. Create output directories before training.
Use `python train.py --help` for all options.

The current defaults:

- Remove samples with `vmag < 5e-8`; disable with `--disable-vmag-filter`.
- Split samples into 70% training, 15% validation, and the remainder testing,
  with split seed 42.
- Select up to the requested training sample count (default 30,000) using joint
  sampling in square-root input space; disable with `--disable-joint-sampling`.
- Apply square-root preprocessing. Alternatives are `raw` (fixed scaling only),
  `log`, and `mixed`.
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
  --raw-report friction_emulator/training_report_uniform_standardized.npz \
  --preprocessed-report friction_emulator/training_report_full_preprocessing.npz \
  --out-dir outputs/comparison
```

The first report must be generated separately, for example using
`--normalization raw`. Keep the remaining settings the same for a controlled
comparison; the labels alone do not determine how a model was trained.

## Python inference

Pass an explicit checkpoint path to override the helper's machine-specific
default. For a checkpoint generated by the training example:

```python
from friction_emulator.friction_emulator import init_model, predict_alpha2_np

init_model("friction_emulator/models_full_preprocessing.pt", device="cpu")
alpha2 = predict_alpha2_np([[8.154506659837895e6, 2.775038552821494e-7]])
print(alpha2)
```

Inputs have shape `(N, 2)` in the original data units; predictions have shape
`(N, 1)`. The helper reads preprocessing parameters from the checkpoint.
Only load trusted checkpoints: this helper uses `torch.load` with
`weights_only=False`.

## License

This project is licensed under the [MIT License](LICENSE).
