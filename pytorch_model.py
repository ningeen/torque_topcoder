import os
import pickle
import random
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader

from mobilenetv3 import mobilenetv3_small
from pytorch_dataset import TorqueDataset
from read_and_get_mel import CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=1234):
    """Fix random seeds"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'Saving weights to {filename}\n')
    torch.save(model.state_dict(), filename)

def load_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'Loading weights from {filename}\n')
    model.load_state_dict(torch.load(filename))
    return model


# class WarmupCosineAnnealingLR(CosineAnnealingLR):
#     def __init__(self, optimizer, warmup_prop, t_max, eta_min=0, last_epoch=-1):
#         super().__init__(optimizer, t_max, eta_min, last_epoch)
#         self.warmup_prop = warmup_prop
#
#     def get_lr(self):
#         if self.last_epoch < self.T_max * self.warmup_prop:
#             return [
#                 self.base_lrs * self.last_epoch / self.T_max
#                 for base_lr, group
#                 in zip(self.base_lrs, self.optimizer.param_groups)
#             ]
#         else:
#             return super(CosineAnnealingLR, self).step()


def get_model(pretrained_mn3_path="", pretrained_path=""):
    """Load MobilenetV3 model with specified in and out channels"""
    model = mobilenetv3_small().to(DEVICE)
    if pretrained_mn3_path and not pretrained_path:
        model.load_state_dict(torch.load(pretrained_mn3_path))

    model.features[0][0].weight.data = torch.sum(
        model.features[0][0].weight.data, dim=1, keepdim=True
    )
    model.features[0][0].in_channels = 1

    model.classifier[-1].weight.data = torch.sum(
        model.classifier[-1].weight.data, dim=0, keepdim=True
    )

    model.classifier[-1].bias.data = torch.sum(
        model.classifier[-1].bias.data, dim=0, keepdim=True
    )
    model.classifier[-1].out_features = 1

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    return model


def process_epoch(model, criterion, optimizer, loader):
    """Calc one epoch"""
    losses = []
    y_true = []
    y_pred = []
    with torch.set_grad_enabled(model.training):
        for local_batch, local_labels in loader:
            local_batch, local_labels = \
                local_batch.to(DEVICE), local_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(local_batch)

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


def train_model(model, criterion, optimizer, train_loader, test_loader, n_fold):
    """Training loop"""
    logs = {'loss_train': [], 'loss_val': [], 'mse_train': [], 'mse_val': []}
    best_true = None
    best_pred = None
    for epoch in range(CONFIG['num_epochs']):
        start_time = time.time()
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
            f"Train loss: {loss_train:.3f}, train mse: {mse_train:.5f}. "
            f"Val loss: {loss_val:.3f}, val mse: {mse_val:.5f}"
        )
        if mse_val <= np.min(logs['mse_val']):
            if CONFIG['save_model']:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        CONFIG['model_dir'],
                        f"work_{CONFIG['experiment_name']}_fold{n_fold}.pt"
                    )
                )
            best_true = y_true
            best_pred = y_pred
    return best_true, best_pred


def run_training():
    with open(CONFIG['data_path'], 'rb') as f:
        (data, mel_logs, target) = pickle.load(f)

    folds = KFold(
        n_splits=CONFIG['n_folds'],
        shuffle=True,
        random_state=CONFIG['fold_seed']
    )
    splits = list(folds.split(mel_logs))

    for n_fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Start #{n_fold + 1} fold")
        train_dataset = TorqueDataset(
            data[train_idx],
            [mel_logs[i] for i in train_idx],
            target[train_idx]
        )
        val_dataset = TorqueDataset(
            data[val_idx],
            [mel_logs[i] for i in val_idx],
            target[val_idx]
        )
        train_loader = DataLoader(train_dataset, **CONFIG['loader_params'])
        val_loader = DataLoader(val_dataset, **CONFIG['loader_params'])

        model = get_model(CONFIG['pretrained_path'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), CONFIG['lr'])
        best_true, best_pred = \
            train_model(model, criterion, optimizer, train_loader, val_loader, n_fold)

        rmse = mean_squared_error(best_true, best_pred, squared=False)
        print(f"Training done. Best rmse: {rmse}")
        if n_fold == 2:
            break


if __name__ == "__main__":
    seed_everything()
    run_training()
