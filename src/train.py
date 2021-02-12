# python /code/read_and_get_mel.py $TRAIN_AUDIO $TRAIN_GT
# python /code/folds.py
# python /code/train_model.py $OUTPUT_DIR
import sys
import os
import logging
import logging.config
import yaml

LOG_PATH = '/code/logging.conf.yml'
with open(LOG_PATH) as config_fin:
    logging.config.dictConfig(yaml.safe_load(config_fin))
logger = logging.getLogger(__name__)


def main():
    from read_and_get_mel import main as read_input
    from folds import get_folds
    from train_model import main as train_models

    try:
        train_audio = sys.argv[1]
        train_gt = sys.argv[2]
        output_dir = sys.argv[3]
    except:
        train_audio = None
        train_gt = None
        output_dir = None
        logger.info("Using default paths")

    read_input(train_audio, train_gt)
    get_folds()
    train_models(output_dir=output_dir)


if __name__ == '__main__':
    main()
