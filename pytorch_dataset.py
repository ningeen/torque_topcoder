import os
import random
import time
import logging
import logging.config
import pickle
import yaml
import librosa
import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from read_and_get_mel import CONFIG

from mobilenetv3 import mobilenetv3_small


def uniform_len(mel, input_len):
    mel_len = mel.shape[-1]
    if mel_len > input_len:
        diff = mel_len - input_len
        start = np.random.randint(diff)
        end = start + input_len
        mel = mel[:, start: end]
    elif mel_len < input_len:
        diff = input_len - mel_len
        offset = np.random.randint(diff)
        offset_right = diff - offset
        mel = np.pad(
            mel,
            ((0, 0), (offset, offset_right)),
            "symmetric",  # constant
        )
    return mel


class TorqueDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, mel_logs, labels=None, transform=None):
        """Init Dataset"""
        self.mel_logs = mel_logs
        self.data = data
        self.labels = labels
        self.transform = transform
        self.input_len = CONFIG['mel']['mel_len']

    def __len__(self):
        """Length"""
        return len(self.mel_logs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        table_data = self.data[[index]]

        label = None
        if self.labels is not None:
            label = self.labels[[index]]

        mel_data = uniform_len(self.mel_logs[index], self.input_len)
        if self.transform:
            mel_data = self.transform(mel_data)

        mel_data = np.expand_dims(mel_data, axis=0)
        return mel_data, label