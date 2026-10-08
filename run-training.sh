#!/bin/bash
# Train a few models on the same data config, then compare them with evaluate.py.
python train.py --model dense-ae --epochs 20 --data-config config/baseline.yml --output output/baseline/
python train.py --model tiny-ae --epochs 20 --data-config config/baseline.yml --output output/tiny/
python train.py --model pca --data-config config/baseline.yml --output output/pca/
