import logging
import logging.config
from argparse import ArgumentParser

import yaml

LOG_PATH = '/code/logging.conf.yml'
with open(LOG_PATH) as config_fin:
    logging.config.dictConfig(yaml.safe_load(config_fin))
logger = logging.getLogger(__name__)


def get_train_args():
    """Parse arguments from command line"""
    parser = ArgumentParser(
        prog="Training model",
    )
    parser.add_argument(
        "train_audio",
        help="Path to audio *.wav files directory for training",
        default=None
    )
    parser.add_argument(
        "train_gt",
        help="Path to *.csv file with data for training",
        default=None
    )
    parser.add_argument(
        "output_dir",
        help="Path where to save files/weights for prediction phase",
        default=None
    )
    args = parser.parse_args()
    logger.debug("Arguments parsed.")
    return args


def main():
    """Run training"""
    from read_and_get_mel import main as read_input
    from folds import get_folds
    from train_model import main as train_models

    args = get_train_args()
    read_input(args.train_audio, args.train_gt)
    get_folds()
    train_models(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
