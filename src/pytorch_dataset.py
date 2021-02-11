import logging

import numpy as np
from torch.utils.data import Dataset

from config import CONFIG

logger = logging.getLogger(__name__)


class TorqueDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, mel_logs, labels=None, transform=None, n_feat=14, n_channels=1):
        """Init Dataset"""
        self.mel_logs = mel_logs
        self.data = data
        self.labels = labels
        self.transform = transform
        self.input_len = CONFIG['mel']['mel_len']
        self.mode = 'test' if self.labels is None else 'train'
        self.n_feat = n_feat
        self.n_channels = n_channels
        logger.debug(
            "Dataset initialized with length: %d. Labels: %s, transform: %s, n_feat: %d, n_channels: %d",
            len(mel_logs),
            "yes" if labels is not None else "no",
            "yes" if transform is not None else "no",
            n_feat, n_channels
        )

    def __len__(self):
        """Length"""
        return len(self.mel_logs)

    @staticmethod
    def add_frequency_encoding(x):
        """Adds layer with -1 to 1 values for keeping time info"""
        d, h, w = x.shape
        vertical = np.linspace(-1, 1, h).reshape((1, -1, 1))
        vertical = np.repeat(vertical, w, axis=2)
        x = np.concatenate([x, vertical], axis=0)
        return x.astype(np.float32)

    @staticmethod
    def uniform_len(mel, input_len):
        """Uniforms length from custom to input_len"""
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

    def __getitem__(self, index):
        """Generates one sample of data"""
        table_data = self.data[index][:self.n_feat]
        mel_data = self.uniform_len(self.mel_logs[index], self.input_len)
        label = None

        if self.mode == 'train':
            label = self.labels[[index]]
            if self.transform:
                mel_data = self.transform(mel_data)

        mel_data = np.expand_dims(mel_data, axis=0)
        if self.n_channels == 2:
            mel_data = self.add_frequency_encoding(mel_data)

        if label is None:
            return mel_data, table_data
        return mel_data, table_data, label
