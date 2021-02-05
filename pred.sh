#!/bin/bash

MODEL_DIR="/tmp/"
INPUT_AUDIO="/data/input/pred/"
OUTPUT_DIR="/data/output/pred/"

python /code/predict_test.py $MODEL_DIR $INPUT_AUDIO $OUTPUT_DIR
