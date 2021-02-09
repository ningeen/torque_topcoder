import random

import numpy as np


def spec_augment(spec, num_mask=2, freq_masking_max_percentage=0.05, time_masking_max_percentage=0.1):
    """Spectrogram augmentation https://paperswithcode.com/paper/specaugment-a-simple-data-augmentation-method"""
    spec = spec.copy()
    for i in range(num_mask):
        num_freqs, num_frames = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        time_percentage = random.uniform(0.0, time_masking_max_percentage)

        num_freqs_to_mask = int(freq_percentage * num_freqs)
        num_frames_to_mask = int(time_percentage * num_frames)

        t0 = int(
            np.random.uniform(low=0.0, high=num_frames - num_frames_to_mask))
        f0 = int(np.random.uniform(low=0.0, high=num_freqs - num_freqs_to_mask))

        spec[:, t0:t0 + num_frames_to_mask] = 0
        spec[f0:f0 + num_freqs_to_mask, :] = 0
    return spec
