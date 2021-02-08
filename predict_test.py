import time
import sys
import os
import torch
import numpy as np
import pandas as pd
from pytorch_model import get_prediction, TorqueModel
from read_and_get_mel import get_data, CONFIG

def search_file(fname):
    for dirpath, dirnames, filenames in os.walk("../"):
        for filename in filenames:
            if filename == fname:
                return os.path.join(dirpath, filename)
    return fname

if len(sys.argv) != 4:
    MODEL_DIR = "./"
    INPUT_AUDIO = "../../pred/"
    OUTPUT_DIR = "./"
else:
    MODEL_DIR = sys.argv[1]
    INPUT_AUDIO = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]

start = time.time()

pretrained_weights_dir = '/code/weights/'
INPUT_CSV = 'input_for_pred.csv'
DATA_FNAME = 'data.p'
device = torch.device("cpu")

csv_path = os.path.join(INPUT_AUDIO, INPUT_CSV)
data, mel_logs, _ = get_data(
    csv_path=csv_path,
    wav_dir=INPUT_AUDIO,
    data_path=DATA_FNAME,
    to_save=False,
    is_train=False,
)

print(f"Data loaded in {time.time() - start}s")
start = time.time()

num_folds = 10
weights = [0.9, 0.9, 1.0]
y_preds = np.zeros((3, len(mel_logs), 1))

# origin_2ch
model = TorqueModel(
    CONFIG['model_params']['out_features_conv'],
    14,
    CONFIG['model_params']['mid_features'],
    n_channels=2
)
for i in range(num_folds):
    fname = f'work_origin_2ch_fold{i}.pt'
    pretrained_path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(pretrained_path):
        pretrained_path = search_file(fname)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    prediction = get_prediction(data, mel_logs, model, device, n_feat=14, n_channels=2)
    y_preds[0] = prediction
print("First model done")

# origin_1ch_9f
model = TorqueModel(
    CONFIG['model_params']['out_features_conv'],
    9,
    CONFIG['model_params']['mid_features'],
    n_channels=1
)
for i in range(num_folds):
    fname = f'work_origin_1ch_9f_fold{i}.pt'
    pretrained_path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(pretrained_path):
        pretrained_path = search_file(fname)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    prediction = get_prediction(data, mel_logs, model, device, n_feat=9, n_channels=1)
    y_preds[1] = prediction
print("Second model done")

# part_new_augs
for i in range(num_folds):
    fname = f'work_part_new_augs_fold{i}.pt'
    pretrained_path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(pretrained_path):
        pretrained_path = search_file(fname)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    prediction = get_prediction(data, mel_logs, model, device, n_feat=9, n_channels=1)
    y_preds[2] = prediction
print("Third model done")

print(f"Model predicted in {time.time() - start}s")
start = time.time()

df = pd.read_csv(csv_path)
df = df[['filename']]
df['result'] = np.average(y_preds, weights=weights, axis=0)
df.to_csv(os.path.join(OUTPUT_DIR, 'result.csv'), index=False)

print(f"Results saved in {time.time() - start}s")
