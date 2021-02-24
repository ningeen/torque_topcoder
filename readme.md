# 6th place solution for ["Predict the torque"](https://www.topcoder.com/challenges/07596fc0-961b-471b-aca9-0932501ef594) Topcoder challenge
## Problem Description

In this problem, you will be given an audio file containing the sound made while bolting steel material on a construction site. Your code should read the audio file and predict the torque value of it.  
**Metric**: RMSE.  
**Inference**: Only CPU and must not take any longer than 10 minutes.  

## Solution
To run training:
```
docker run --rm -v {AUDIO_DIR}:/data/input/train -v {CSV_DATA_DIR}:/data/gt/train -v {MODEL_WEIGHTS_DIR}:/tmp/ {IMAGE_NAME} /code/train.sh
```
To run prediction:
```
docker run --rm -v {AUDIO_AND_CSV_DATA_DIR}:/data/input/pred -v {OUTPUT_RESULTS_DIR}:/data/output/pred -v {MODEL_WEIGHTS_DIR}:/tmp/ {IMAGE_NAME} /code/pred.sh
```
