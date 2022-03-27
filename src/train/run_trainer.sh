#!/bin/bash
for i in {0..4}
do
  echo "BASH launching TRAIN $i"
  PYTHONPATH=$(pwd) python3 src/train/trainer.py --fold_index=$i
done