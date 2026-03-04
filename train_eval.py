import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, preds, targets

def compute_metrics(preds, targets):
    n_tissues = preds.shape[1]
    pearson_list = []
    rmse_list = []
    for i in range(n_tissues):
        if np.std(preds[:, i]) > 0 and np.std(targets[:, i]) > 0:
            r, _ = pearsonr(preds[:, i], targets[:, i])
        else:
            r = np.nan
        rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        pearson_list.append(r)
        rmse_list.append(rmse)
    return np.nanmean(pearson_list), np.mean(rmse_list), pearson_list, rmse_list

def plot_results(preds, targets, tissue_names, save_path='results.png'):
    n_tissues = len(tissue_names)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for i in range(min(n_tissues, 16)):
        ax = axes[i]
        ax.scatter(targets[:, i], preds[:, i], alpha=0.3, s=5)
        ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        ax.set_title(tissue_names[i][:20])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()