import logging
import logging.config
import os
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from config import CONFIG
from pytorch_dataset import TorqueDataset
from pytorch_model import TorqueModel
from read_and_get_mel import DataMelLoader

LOG_PATH = '/code/logging.conf.yml'
with open(LOG_PATH) as config_fin:
    logging.config.dictConfig(yaml.safe_load(config_fin))
logger = logging.getLogger(__name__)


def get_prediction_args():
    """Parse arguments from command line"""
    parser = ArgumentParser(
        prog="Training model",
    )
    parser.add_argument(
        "model_dir",
        help="Directory with saved model weights",
        default=None
    )
    parser.add_argument(
        "input_audio",
        help="Path to audio *.wav and *.csv files directory for prediction",
        default=None
    )
    parser.add_argument(
        "output_dir",
        help="Path where to save result",
        default=None
    )
    args = parser.parse_args()
    logger.debug("Arguments parsed.")
    return args


def get_prediction(data, mel_logs, model, device, n_feat, n_channels=1):
    """Get prediction from model on given data"""
    test_dataset = TorqueDataset(data, mel_logs, n_feat=n_feat, n_channels=n_channels)
    loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['loader_params']['batch_size'],
        num_workers=CONFIG['loader_params']['num_workers'],
        shuffle=False
    )

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.grad = None

    y_pred = []
    for local_batch, local_data in loader:
        local_batch, local_data = local_batch.to(device), local_data.to(device)
        outputs = model(local_batch, local_data)
        y_pred.append(outputs.data.detach().numpy())
    return np.concatenate(y_pred)


def search_file(fname):
    """Search model weights"""
    for dirpath, dirnames, filenames in os.walk("../"):
        for filename in filenames:
            if filename == fname:
                result_path = os.path.join(dirpath, filename)
                logger.debug("Weights found in %s", result_path)
                return result_path
    return fname


def load_files(csv_path, input_audio, data_path):
    """Load given data"""
    start = time.time()
    data, mel_logs, _ = DataMelLoader(CONFIG).get_data(
        csv_path=csv_path, wav_dir=input_audio, data_path=data_path, to_save=False, is_train=False
    )
    logger.info("Files loaded in %.1f s", time.time() - start)
    return data, mel_logs


def predict_test(data, mel_logs, model_name, model_dir, n_feat, n_channels, num_folds, device):
    """Get average prediction from different folds models"""
    start = time.time()
    y_pred = np.zeros((len(mel_logs), 1))

    model = TorqueModel(
        CONFIG['model_params']['out_features_conv'],
        CONFIG['model_params']['out_features_dense'],
        CONFIG['model_params']['mid_features'],
        n_channels=n_channels
    )
    for i in range(num_folds):
        fname = f'work_{model_name}_fold{i}.pt'
        pretrained_path = os.path.join(model_dir, fname)
        if not os.path.isfile(pretrained_path):
            logger.debug("Weights not found in %s", pretrained_path)
            pretrained_path = search_file(fname)
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        prediction = get_prediction(data, mel_logs, model, device, n_feat=n_feat, n_channels=n_channels)
        y_pred += prediction
    logger.info("%s model done in %.1f s", model_name, time.time() - start)
    return y_pred / num_folds


def make_prediction(data, mel_logs, model_dir, device):
    """Get average prediction from all models"""
    num_folds = CONFIG['fold_params']['n_splits']
    model_names = CONFIG['test']['model_names']
    weights = CONFIG['test']['average_weights']
    n_feats = CONFIG['test']['n_feats']
    n_channels = CONFIG['test']['n_channels']
    y_preds = np.zeros((len(model_names), len(mel_logs), 1))

    for n_model in range(len(model_names)):
        y_preds[n_model] = predict_test(
            data, mel_logs, model_names[n_model], model_dir,
            n_feats[n_model], n_channels[n_model], num_folds, device
        )
    result = np.average(y_preds, weights=weights, axis=0)
    return result


def save_result(csv_path, output_dir, result):
    """Save prediction to csv"""
    df = pd.read_csv(csv_path)
    df = df[['filename']]
    df['result'] = result
    df.to_csv(os.path.join(output_dir, 'result.csv'), index=False)
    logger.info("Predictions were successfully saved.")


def main():
    """Main script"""
    args = get_prediction_args()

    data_path = os.path.basename(CONFIG['data_path'])
    device = torch.device(CONFIG['test']['device'])
    csv_path = os.path.join(args.input_audio, CONFIG['test_input_path'])

    data, mel_logs = load_files(csv_path, args.input_audio, data_path)
    result = make_prediction(data, mel_logs, args.model_dir, device)
    save_result(csv_path, args.output_dir, result)


if __name__ == '__main__':
    main()
