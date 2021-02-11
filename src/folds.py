import logging
import pickle

from sklearn.model_selection import KFold

from config import CONFIG

logger = logging.getLogger(__name__)


def get_folds():
    """Generate and save folds"""
    try:
        with open(CONFIG['data_path'], 'rb') as f:
            (data, mel_logs, target) = pickle.load(f)
    except FileNotFoundError:
        logger.error("No files found for split")
        return

    folds = KFold(**CONFIG['fold_params'])
    splits = list(folds.split(data))
    with open(CONFIG['folds_path'], 'wb') as f:
        pickle.dump(splits, f)
    logger.info("Folds saved in %s", CONFIG['folds_path'])


def load_folds():
    """Load saved folds"""
    try:
        with open(CONFIG['folds_path'], 'rb') as f:
            splits = pickle.load(f)
    except FileNotFoundError:
        logger.error("Folds not found in %s", CONFIG['folds_path'])
        return
    logger.info("Folds loaded from %s", CONFIG['folds_path'])
    return splits


if __name__ == '__main__':
    get_folds()
