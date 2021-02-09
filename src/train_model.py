import os
import sys
import pickle
import random
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader

from augmentations import spec_augment
from cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts
from folds import load_folds
from pytorch_dataset import TorqueDataset
from pytorch_model import TorqueModel
from config import CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def seed_everything(seed=1234):
    """Fix random seeds"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def process_epoch(model, criterion, optimizer, loader):
    """Calc one epoch"""
    losses = []
    y_true = []
    y_pred = []
    with torch.set_grad_enabled(model.training):
        for local_batch, local_data, local_labels in loader:
            local_batch, local_data, local_labels = \
                local_batch.to(DEVICE), local_data.to(DEVICE), local_labels.to(DEVICE)

            for param in model.parameters():
                param.grad = None
            outputs = model(local_batch, local_data)

            loss = criterion(outputs, local_labels)
            if model.training:
                loss.backward()
                optimizer.step()

            losses.append(loss)
            y_true.append(local_labels.detach().cpu().numpy())
            y_pred.append(outputs.data.detach().cpu().numpy())
    loss_train = np.array(losses).astype(np.float32).mean()
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    rmse_train = mean_squared_error(y_true, y_pred, squared=False)
    return loss_train, rmse_train, y_true, y_pred


def train_model(model, criterion, optimizer, scheduler, train_loader, test_loader, n_fold):
    """Training loop"""
    logs = {'loss_train': [], 'loss_val': [], 'mse_train': [], 'mse_val': []}
    best_true = None
    best_pred = None
    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
        scheduler.step()

        # Training
        model.train()
        loss_train, mse_train, _, _ = \
            process_epoch(model, criterion, optimizer, train_loader)
        logs['loss_train'].append(loss_train)
        logs['mse_train'].append(mse_train)

        # Validation
        model.eval()
        loss_val, mse_val, y_true, y_pred = \
            process_epoch(model, criterion, optimizer, test_loader)
        logs['loss_val'].append(loss_val)
        logs['mse_val'].append(mse_val)
        print(
            f"Epoch #{epoch + 1}. "
            f"Time: {(time.time() - start_time):.1f}s. "
            f"Train loss: {loss_train:.3f}, train rmse: {mse_train:.5f}. "
            f"Val loss: {loss_val:.3f}, val rmse: {mse_val:.5f}"
        )
        if mse_val <= np.min(logs['mse_val']):
            if CONFIG['save_model']:
                weights_filename = f"work_{CONFIG['experiment_name']}_fold{n_fold}.pt"
                weights_path = os.path.join(CONFIG['weights_dir'], weights_filename)
                torch.save(model.state_dict(), weights_path)
            best_true = y_true
            best_pred = y_pred
    return best_true, best_pred


def run_training(all_data=None):
    """Run training"""
    if all_data is None:
        with open(CONFIG['data_path'], 'rb') as f:
            (data, mel_logs, target) = pickle.load(f)
    else:
        (data, mel_logs, target) = all_data

    splits = load_folds()

    total_rmse = list()
    for n_fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Start #{n_fold + 1} fold")
        train_dataset = TorqueDataset(
            data[train_idx],
            [mel_logs[i] for i in train_idx],
            target[train_idx],
            transform=spec_augment
        )
        val_dataset = TorqueDataset(
            data[val_idx],
            [mel_logs[i] for i in val_idx],
            target[val_idx]
        )
        train_loader = DataLoader(train_dataset, **CONFIG['loader_params'])
        val_loader = DataLoader(val_dataset, **CONFIG['loader_params'])

        model = TorqueModel(**CONFIG['model_params'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), CONFIG['lr'])

        scheduler = CosineAnnealingWarmupRestarts(optimizer, **CONFIG['scheduler_params'])

        best_true, best_pred = train_model(
            model, criterion, optimizer, scheduler, train_loader, val_loader, n_fold
        )

        rmse = mean_squared_error(best_true, best_pred, squared=False)
        print(f"Training done. Best rmse: {rmse}")
        total_rmse.append(rmse)
    print(f"Total rmse: {np.mean(total_rmse)}")


if __name__ == "__main__":
    OUTPUT_DIR = sys.argv[1]
    CONFIG['weights_dir'] = OUTPUT_DIR

    seed_everything()
    run_training()