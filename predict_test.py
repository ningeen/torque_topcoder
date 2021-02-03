import sys
import os
import torch
import pandas as pd
from pytorch_model import get_mobilenet_model, get_prediction
from read_and_get_mel import get_data

if len(sys.argv) != 4:
    MODEL_DIR = "./"
    INPUT_AUDIO = "../../pred/"
    OUTPUT_DIR = "./"
else:
    MODEL_DIR = sys.argv[1]
    INPUT_AUDIO = sys.argv[2]
    OUTPUT_DIR = sys.argv[3]

INPUT_CSV = 'input_for_pred.csv'
DATA_FNAME = 'data.p'
device = torch.device("cpu")
model = get_mobilenet_model(
    pretrained_path=os.path.join(MODEL_DIR, 'weights.pt'),
    device=device
)

csv_path = os.path.join(INPUT_AUDIO, INPUT_CSV)
data, mel_logs, target = get_data(
    csv_path=csv_path,
    wav_dir=INPUT_AUDIO,
    data_path=DATA_FNAME,
    to_save=False
)

prediction = get_prediction(data, mel_logs, target, model, device)

df = pd.read_csv(csv_path)
df = df['filename']
df['result'] = prediction
df.to_csv(os.path.join(OUTPUT_DIR, 'result.csv'), index=False)
