import time
import sys
import os
import yaml
import torch
import numpy as np
import pandas as pd
from pytorch_model import get_prediction, TorqueModel
from read_and_get_mel import DataMelLoader, CONFIG

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

# with open("code/proj_config.yaml", 'r') as stream:
#     config_long = yaml.safe_load(stream)

# with open("proj_config_nmel.yaml", 'r') as stream:
#     config_wide = yaml.safe_load(stream)

config_long = CONFIG

csv_path = os.path.join(INPUT_AUDIO, INPUT_CSV)
data, mel_logs, _ = DataMelLoader(config_long).get_data(
    csv_path=csv_path,
    wav_dir=INPUT_AUDIO,
    data_path=DATA_FNAME, to_save=False, is_train=False)
# _, mel_logs_wide, _ = DataMelLoader(config_wide).get_data(to_save=False, is_train=False)

print(f"Data loaded in {time.time() - start}s")
start = time.time()

num_folds = 10
model_names = ['bootstrap', 'bootstrap_2ch']
# y_preds = np.zeros((len(mel_logs), 1))
weights = [1, 1]
y_preds = np.zeros((len(model_names), len(mel_logs), 1))

model = TorqueModel(
    config_long['model_params']['out_features_conv'],
    config_long['model_params']['out_features_dence'],
    config_long['model_params']['mid_features'],
    n_channels=1
)
for i in range(num_folds):
    fname = f'work_bootstrap_fold{i}.pt'
    pretrained_path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(pretrained_path):
        pretrained_path = search_file(fname)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    prediction = get_prediction(data, mel_logs, model, device, n_feat=14, n_channels=1)
    y_preds[0] += prediction
print("First model done")

model = TorqueModel(
    config_long['model_params']['out_features_conv'],
    config_long['model_params']['out_features_dence'],
    config_long['model_params']['mid_features'],
    n_channels=2
)
for i in range(num_folds):
    fname = f'work_bootstrap_2ch_fold{i}.pt'
    pretrained_path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(pretrained_path):
        pretrained_path = search_file(fname)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    prediction = get_prediction(data, mel_logs, model, device, n_feat=14, n_channels=2)
    y_preds[1] += prediction
print("Second model done")
#
# for i in range(num_folds):
#     fname = f'work_bootstrap_nmel_glen_153_fold{i}.pt'
#     pretrained_path = os.path.join(MODEL_DIR, fname)
#     if not os.path.isfile(pretrained_path):
#         pretrained_path = search_file(fname)
#     model.load_state_dict(torch.load(pretrained_path, map_location=device))
#     prediction = get_prediction(data, mel_logs_wide, model, device, n_feat=14, n_channels=1)
#     y_preds[2] += prediction
# print("Third model done")

print(f"Model predicted in {time.time() - start}s")
start = time.time()

df = pd.read_csv(csv_path)
df = df[['filename']]
df['result'] = np.average(y_preds, weights=weights, axis=0) / num_folds
df.to_csv(os.path.join(OUTPUT_DIR, 'result.csv'), index=False)

print(f"Results saved in {time.time() - start}s")
