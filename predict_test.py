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

model = TorqueModel(
    CONFIG['model_params']['out_features_conv'],
    CONFIG['model_params']['out_features_dence'],
    CONFIG['model_params']['mid_features'],
    device=device
)

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
y_pred = np.zeros((len(mel_logs), 1))
for i in range(num_folds):
    fname = f'work_add_table_fold{i}.pt'
    pretrained_path = os.path.join(MODEL_DIR, fname)
    if not os.path.isfile(pretrained_path):
        pretrained_path = search_file(fname)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    prediction = get_prediction(data, mel_logs, model, device)
    y_pred += prediction

print(f"Model predicted in {time.time() - start}s")
start = time.time()

df = pd.read_csv(csv_path)
df = df[['filename']]
df['result'] = y_pred / num_folds
df.to_csv(os.path.join(OUTPUT_DIR, 'result.csv'), index=False)

print(f"Results saved in {time.time() - start}s")
