#!/bin/bash
python compare_training_cases.py \
  --raw-report ./training_report_uniform_standardized.npz \
  --preprocessed-report ./training_report_full_preprocessing.npz \
  --raw-label "uniform sampling + standardization" \
  --out-dir ./plots_preprocessing_comparison
