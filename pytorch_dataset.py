import numpy as np
from torch.utils.data import Dataset

from read_and_get_mel import CONFIG


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
        self.mode = 'test' if self.labels is None else 'train'
        self.n_feat = CONFIG['model_params']['out_features_dence']

    def __len__(self):
        """Length"""
        return len(self.mel_logs)

    def add_frequency_encoding(self, x):
        d, h, w = x.shape
        vertical = np.linspace(-1, 1, h).reshape((1, -1, 1))
        vertical = np.repeat(vertical, w, axis=2)
        x = np.concatenate([x, vertical], axis=0)
        return x.astype(np.float32)

    def __getitem__(self, index):
        """Generates one sample of data"""
        table_data = self.data[index][:self.n_feat]

        label = None
        if self.mode == 'train':
            label = self.labels[[index]]

        mel_data = uniform_len(self.mel_logs[index], self.input_len)
        if self.transform and self.mode == 'train':
            mel_data = self.transform(mel_data)

        mel_data = np.expand_dims(mel_data, axis=0)
        if CONFIG['channels'] == 2:
            mel_data = self.add_frequency_encoding(mel_data)
        if label is None:
            return mel_data, table_data
        return mel_data, table_data, label