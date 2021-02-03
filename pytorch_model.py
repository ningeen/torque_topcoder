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
import torch.nn.functional as F

from mobilenetv3 import mobilenetv3_small, mobilenetv3_large
from cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts
from pytorch_dataset import TorqueDataset
from read_and_get_mel import CONFIG

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

def save_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'Saving weights to {filename}\n')
    torch.save(model.state_dict(), filename)

def load_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'Loading weights from {filename}\n')
    model.load_state_dict(torch.load(filename))
    return model


def spec_augment(spec: np.ndarray, num_mask=2, freq_masking_max_percentage=0.05,
                 time_masking_max_percentage=0.1):
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


def get_mobilenet_model(pretrained_mn3_path="", pretrained_path="", device=DEVICE):
    """Load MobilenetV3 model with specified in and out channels"""
    # model = mobilenetv3_small().to(DEVICE)
    model = mobilenetv3_large().to(device)
    if pretrained_mn3_path and not pretrained_path:
        model.load_state_dict(torch.load(pretrained_mn3_path))

    model.features[0][0].weight.data = torch.sum(
        model.features[0][0].weight.data, dim=1, keepdim=True
    )
    model.features[0][0].in_channels = 1

    # model.classifier[-1].weight.data = torch.sum(
    #     model.classifier[-1].weight.data, dim=0, keepdim=True
    # )
    #
    # model.classifier[-1].bias.data = torch.sum(
    #     model.classifier[-1].bias.data, dim=0, keepdim=True
    # )
    # model.classifier[-1].out_features = out_features

    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path))
    return model


class TorqueModel(nn.Module):
    def __init__(self, out_features_conv, out_features_dence, mid_features, pretrained_mn3_path="", pretrained_path=""):
        super(TorqueModel, self).__init__()
        self.mnet = get_mobilenet_model(pretrained_mn3_path, pretrained_path)
        self.fc1 = nn.Linear(out_features_conv + out_features_dence, mid_features)
        self.fc2 = nn.Linear(mid_features, mid_features)
        self.fc3 = nn.Linear(mid_features, 1)

    def forward(self, image, data):
        x1 = self.mnet(image)
        x2 = data
        # print(type(x1), type(x2))
        # print(x1.shape, x2.shape)
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def process_epoch(model, criterion, optimizer, loader):
    """Calc one epoch"""
    losses = []
    y_true = []
    y_pred = []
    with torch.set_grad_enabled(model.training):
        for local_batch, local_data, local_labels in loader:
            local_batch, local_data, local_labels = \
                local_batch.to(DEVICE), local_data.to(DEVICE), local_labels.to(DEVICE)

            # optimizer.zero_grad()
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


def run_training(all_data=None):
    if all_data is None:
        with open(CONFIG['data_path'], 'rb') as f:
            (data, mel_logs, target) = pickle.load(f)
    else:
        (data, mel_logs, target) = all_data

    folds = KFold(
        n_splits=CONFIG['n_folds'],
        shuffle=True,
        random_state=CONFIG['fold_seed']
    )
    splits = list(folds.split(mel_logs))

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

        model = TorqueModel(
            CONFIG['model_params']['out_features_conv'],
            CONFIG['model_params']['out_features_dence'],
            CONFIG['model_params']['mid_features'],
            CONFIG['pretrained_path']
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), CONFIG['lr'])

        CONFIG['scheduler_params']['max_lr'] *= CONFIG['lr']
        CONFIG['scheduler_params']['min_lr'] *= CONFIG['lr']
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  **CONFIG['scheduler_params'])

        best_true, best_pred = \
            train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, n_fold)

        rmse = mean_squared_error(best_true, best_pred, squared=False)
        print(f"Training done. Best rmse: {rmse}")
        total_rmse.append(rmse)
    print(f"Total rmse: {np.mean(total_rmse)}")


def get_prediction(data, mel_logs, target, model, device):
    test_dataset = TorqueDataset(data, mel_logs, target)
    loader = DataLoader(test_dataset, **CONFIG['loader_params'])

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.grad = None

    y_pred = []
    for local_batch, local_data, local_labels in loader:
        local_batch, local_data, local_labels = \
            local_batch.to(device), local_data.to(device), local_labels.to(
                device)
        outputs = model(local_batch, local_data)
        y_pred.append(outputs.data.detach().numpy())
    return np.concatenate(y_pred)


if __name__ == "__main__":
    seed_everything()
    run_training()
