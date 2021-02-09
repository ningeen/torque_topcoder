import os
import yaml
import pickle
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew


CONFIG_PATH = "proj_config.yaml"
# CONFIG_PATH = "proj_config_nmel.yaml"
try:
    with open(CONFIG_PATH, 'r') as stream:
        CONFIG = yaml.safe_load(stream)
except FileNotFoundError as e:
    with open(os.path.join('/code', CONFIG_PATH), 'r') as stream:
        CONFIG = yaml.safe_load(stream)

class DataMelLoader:
    def __init__(self, config):
        self.config = config

    def read_file(self, path):
        """Read wav file"""
        wav, _ = librosa.core.load(path, sr=self.config['sampling_rate'])
        return wav


    def wav_to_mel(self, wav):
        """Calculate melspectrogram"""
        melspec = librosa.feature.melspectrogram(
            wav,
            sr=self.config['sampling_rate'],
            n_fft=self.config['mel']['n_fft'],
            hop_length=self.config['mel']['hop_length'],
            n_mels=self.config['mel']['n_mels']
        )
        logmel = librosa.core.power_to_db(melspec)
        return logmel


    def normalize(self, x):
        return (x - self.config['min_val']) / (self.config['max_val'] - self.config['min_val'])


    def read_wav_files(self, folder, filenames):
        """Read all wav files from directory"""
        features = []

        for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
            path = os.path.join(folder, filename)
            wav = self.read_file(path)
            features.append(wav)
        return features

    def get_mel(self, wav_files):
        features = []
        for wav in tqdm(wav_files):
            mel = self.wav_to_mel(wav)
            mel = self.normalize(mel)
            features.append(mel.astype(np.float32))
        return features

    def get_data(self, csv_path, wav_dir, data_path, to_save=True, is_train=True):
        # csv_path, wav_dir, data_path = self.config['csv_path'], self.config['wav_dir'], self.config['data_path']
        df = pd.read_csv(csv_path)
        if is_train:
            df = df[~df['filename'].isin(self.config['ignore_files'])]

        data = []
        dummy_cols = ['device_id', 'junction_type', 'is_flange']
        for col in dummy_cols:
            data.append(pd.get_dummies(df[col], prefix=f"dummy_{col}"))
        data = pd.concat(data, axis=1)

        wav_files = self.read_wav_files(wav_dir, df['filename'])
        mel_logs = self.get_mel(wav_files)

        sr = self.config['sampling_rate']
        data['sound_lengths'] = [len(wav) / sr for wav in wav_files]
        data['sound_mean'] = [wav.mean() for wav in wav_files]
        data['sound_min'] = [wav.min() for wav in wav_files]
        data['sound_max'] = [wav.max() for wav in wav_files]
        data['sound_std'] = [wav.std() for wav in wav_files]
        data['sound_skew'] = [skew(wav) for wav in wav_files]

        target = None
        if 'tightening_result_torque' in df.columns:
            target = df['tightening_result_torque'].to_numpy().astype(np.float32)

        data = data.to_numpy().astype(np.float32)

        if to_save:
            with open(data_path, 'wb') as f:
                pickle.dump((data, mel_logs, target), f)

        return data, mel_logs, target


if __name__ == "__main__":
    DataMelLoader(CONFIG).get_data()
