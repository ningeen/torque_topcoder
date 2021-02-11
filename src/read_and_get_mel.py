import logging
import os
import sys
import pickle

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataMelLoader:
    """Data loader class"""

    def __init__(self, config):
        self.config = config

    def read_file(self, path):
        """Read wav file"""
        wav, _ = librosa.core.load(path, sr=self.config['sampling_rate'])
        return wav

    def wav_to_mel(self, wav):
        """Calculate mel spectrogram"""
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
        """Normalize input to [0, 1]"""
        return (x - self.config['min_val']) / (self.config['max_val'] - self.config['min_val'])

    def read_wav_files(self, folder, filenames):
        """Read all wav files from directory"""
        features = []

        for i, filename in tqdm(enumerate(filenames), total=len(filenames)):
            path = os.path.join(folder, filename)
            wav = self.read_file(path)
            features.append(wav)
        logger.info("Loaded %d audio files", len(features))
        return features

    def get_mel(self, wav_files):
        """Generate mel spectrogram from waveform"""
        features = []
        for wav in tqdm(wav_files):
            mel = self.wav_to_mel(wav)
            mel = self.normalize(mel)
            features.append(mel.astype(np.float32))
        logger.info("Files converted to mel spectrogram")
        return features

    def make_features(self, df, wav_files):
        """Generate features from df and audio"""
        data = []
        dummy_cols = ['device_id', 'junction_type', 'is_flange']
        for col in dummy_cols:
            data.append(pd.get_dummies(df[col], prefix=f"dummy_{col}"))
        data = pd.concat(data, axis=1)

        sr = self.config['sampling_rate']
        data['sound_lengths'] = [len(wav) / sr for wav in wav_files]
        data['sound_mean'] = [wav.mean() for wav in wav_files]
        data['sound_min'] = [wav.min() for wav in wav_files]
        data['sound_max'] = [wav.max() for wav in wav_files]
        data['sound_std'] = [wav.std() for wav in wav_files]
        data['sound_skew'] = [skew(wav) for wav in wav_files]
        logger.debug("Generated %d features", data.shape[1])
        return data

    def get_data(self, csv_path=None, wav_dir=None, data_path=None, to_save=False, is_train=True):
        """Load, preprocess and return data"""
        csv_path = self.config['csv_path'] if csv_path is None else csv_path
        wav_dir = self.config['wav_dir'] if wav_dir is None else wav_dir
        data_path = self.config['data_path'] if data_path is None else data_path

        df = pd.read_csv(csv_path)
        target = None

        if is_train:
            df = df[~df['filename'].isin(self.config['ignore_files'])]  # remove files with no torque sound
            target = df['tightening_result_torque'].to_numpy().astype(np.float32)

        wav_files = self.read_wav_files(wav_dir, df['filename'])
        mel_logs = self.get_mel(wav_files)
        data = self.make_features(df, wav_files)

        data = data.to_numpy().astype(np.float32)

        if to_save:
            with open(data_path, 'wb') as f:
                pickle.dump((data, mel_logs, target), f)
            logger.info("Data saved in %s", data_path)
        logger.info("Data successfully loaded")
        return data, mel_logs, target


if __name__ == "__main__":
    from config import CONFIG
    try:
        TRAIN_AUDIO = sys.argv[1]
        TRAIN_GT = sys.argv[2]
        CONFIG['wav_dir'] = TRAIN_AUDIO
        CONFIG['csv_path'] = os.path.join(TRAIN_GT, 'training.csv')
    except:
        logger.info("Using default paths")

    DataMelLoader(CONFIG).get_data(to_save=True)
