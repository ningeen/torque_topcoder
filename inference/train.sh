#!/bin/bash

TRAIN_AUDIO="/data/input/train/"
TRAIN_GT="/data/gt/train/"
OUTPUT_DIR="/tmp/"

python /code/train.py $TRAIN_AUDIO $TRAIN_GT $OUTPUT_DIR