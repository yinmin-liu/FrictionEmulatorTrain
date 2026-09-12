#!/bin/bash
torchrun --nproc_per_node=1 train.py \
	--h1 32 --h2 32 --n-ranks 40 --epochs 7000 \
	--lr 1e-3 --lr-scheduler plateau --lr-patience 4 --lr-factor 0.5 --lr-min 1e-6 \
	--batch-size 16384 --n-seeds 1 --print-every 50 \
   --folder ./data \
   --model-file ./friction_emulator/model_none_uniform_standard.txt \
   --checkpoint ./friction_emulator/model_none_uniform_standard.pt \
	--device auto \
	--transform none --sampling uniform --scaling standard \
	--min-vmag 5e-8 \
	--train-samples 30000 --joint-sampling-seed 42 \
	--slow-vmag-threshold 1e-6 \
	--fast-vmag-threshold 1e-5 \
	--report-data ./friction_emulator/training_report_none_uniform_standard.npz
