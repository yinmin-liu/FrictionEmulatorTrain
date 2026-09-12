#!/bin/bash
python compare_training_cases.py \
  --raw-report ./friction_emulator/training_report_none_uniform_standard.npz \
  --preprocessed-report ./friction_emulator/training_report_sqrt_weighted_standard.npz \
  --raw-label "uniform sampling + standardization" \
  --out-dir ./plots_preprocessing_comparison
