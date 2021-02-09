#!/bin/bash

TRAIN_AUDIO="/data/input/train/"
TRAIN_GT="/data/gt/train/"
OUTPUT_DIR="/tmp/"

python /code/read_and_get_mel.py $TRAIN_AUDIO $TRAIN_GT
python /code/folds.py
python /code/train_model.py $OUTPUT_DIR