import pickle

from sklearn.model_selection import KFold

from config import CONFIG


def get_folds():
    """Generate and save folds"""
    try:
        with open(CONFIG['data_path'], 'rb') as f:
            (data, mel_logs, target) = pickle.load(f)
    except FileNotFoundError:
        print("No files found for split")
        return

    folds = KFold(**CONFIG['fold_params'])
    splits = list(folds.split(data))
    with open(CONFIG['folds_path'], 'wb') as f:
        pickle.dump(splits, f)


def load_folds():
    """Load saved folds"""
    with open(CONFIG['folds_path'], 'rb') as f:
        splits = pickle.load(f)
    return splits


if __name__ == '__main__':
    get_folds()
