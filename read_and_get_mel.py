import os
import yaml
import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew


CONFIG_PATH = "proj_config.yaml"
with open(CONFIG_PATH, 'r') as stream:
    CONFIG = yaml.safe_load(stream)


def read_file(path):
    """Read wav file"""
    wav, _ = librosa.core.load(path, sr=CONFIG['sampling_rate'])
    return wav


def wav_to_mel(wav):
    """Calculate melspectrogram"""
    melspec = librosa.feature.melspectrogram(
        wav,
        sr=CONFIG['sampling_rate'],
        n_fft=CONFIG['mel']['n_fft'],
        hop_length=CONFIG['mel']['hop_length'],
        n_mels=CONFIG['mel']['n_mels']
    )
    logmel = librosa.core.power_to_db(melspec)
    return logmel


def normalize(x):
    return (x - CONFIG['min_val']) / (CONFIG['max_val'] - CONFIG['min_val'])


def read_wav_files(folder, filenames):
    """Read all wav files from directory"""
    features = []

    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
        path = os.path.join(folder, filename)
        wav = read_file(path)
        features.append(wav)
    return features

def get_mel(wav_files):
    features = []
    for wav in tqdm(wav_files):
        mel = wav_to_mel(wav)
        mel = normalize(mel)
        features.append(mel.astype(np.float32))
    return features

def get_data(csv_path, wav_dir, data_path, to_save=True):
    df = pd.read_csv(csv_path)  # .head(10)
    df = df[~df['filename'].isin(CONFIG['ignore_files'])]

    data = []
    dummy_cols = ['device_id', 'junction_type', 'is_flange']
    for col in dummy_cols:
        data.append(pd.get_dummies(df[col], prefix=f"dummy_{col}"))
    data = pd.concat(data, axis=1)

    wav_files = read_wav_files(wav_dir, df['filename'])
    mel_logs = get_mel(wav_files)

    sr = CONFIG['sampling_rate']
    data['sound_lengths'] = [len(wav) / sr for wav in wav_files]
    data['sound_mean'] = [wav.mean() for wav in wav_files]
    data['sound_min'] = [wav.min() for wav in wav_files]
    data['sound_max'] = [wav.max() for wav in wav_files]
    data['sound_std'] = [wav.std() for wav in wav_files]
    data['sound_skew'] = [skew(wav) for wav in wav_files]

    target = None
    if 'tightening_result_torque' in df.columns:
        target = df['tightening_result_torque']

    data = data.to_numpy().astype(np.float32)
    target = target.to_numpy().astype(np.float32)

    if to_save:
        with open(data_path, 'wb') as f:
            pickle.dump((data, mel_logs, target), f)

    return data, mel_logs, target


if __name__ == "__main__":
    get_data(CONFIG['csv_path'], CONFIG['wav_dir'], CONFIG['data_path'])