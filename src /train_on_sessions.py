"""
train_on_sessions.py — Huấn luyện GestureLSTM trên bộ dữ liệu session tự thu thập.

Dùng:
    cd "src "
    python train_on_sessions.py                    # gamma=2.0 (mặc định)
    python train_on_sessions.py --gamma 0.0        # CE thuần (baseline)
    python train_on_sessions.py --data ../my_data  # thư mục session khác

Weights được lưu tại:
    ../src/weights/best_lstm_sessions_gamma{gamma}.pth
    ../src/weights/norm_stats_sessions_gamma{gamma}.npz
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from model          import GestureLSTM
from losses         import GestureLoss
from dataset        import sEMGDataset
from session_loader import load_and_split_session_csv, CLASS_NAMES, NUM_CLASSES

# ── Đường dẫn ─────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / 'src' / 'weights'
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────
WINDOW     = 40
STEP       = 20
BATCH      = 64
EPOCHS     = 60
LR         = 1e-3
HIDDEN     = 128
WARMUP_EP  = 5
PATIENCE   = 12
INPUT_SIZE = 1


def normalize(X_train, X_val, X_test):
    mu  = X_train.mean()
    std = X_train.std() + 1e-8
    return (X_train - mu) / std, (X_val - mu) / std, (X_test - mu) / std, mu, std


def lr_lambda(ep):
    if ep < WARMUP_EP:
        return (ep + 1) / WARMUP_EP
    p = (ep - WARMUP_EP) / max(EPOCHS - WARMUP_EP, 1)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))


def run_epoch(model, loader, criterion, optimizer=None, device='cpu'):
    model.train() if optimizer else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if optimizer else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(xb)
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(xb)
    return total_loss / total, correct / total


def train_model(gamma=2.0, data_dir=None, verbose=True):
    """
    Huấn luyện GestureLSTM với focal loss gamma cho trước.

    Trả về:
        history (dict): train_loss, val_loss, train_acc, val_acc theo epoch
        best_val_loss (float)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    suffix = f'_gamma{gamma}'

    if verbose:
        print(f"\n{'='*60}")
        print(f"  TRAIN SESSION — gamma={gamma}")
        print(f"{'='*60}")
        print(f"  Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_split_session_csv(
        data_dir, WINDOW, STEP)
    X_tr, X_va, X_te, mu, std = normalize(X_tr, X_va, X_te)
    np.savez(WEIGHTS_DIR / f'norm_stats_sessions{suffix}.npz',
             mean=np.array([mu]), std=np.array([std]))

    tr_ld = DataLoader(sEMGDataset(X_tr, y_tr), BATCH, shuffle=True,  num_workers=0)
    va_ld = DataLoader(sEMGDataset(X_va, y_va), BATCH, shuffle=False, num_workers=0)

    # ── Class weights ─────────────────────────────────────────────────
    cw = np.clip(
        compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_tr),
        0.2, 10.0)
    if verbose:
        print(f"  Class weights: { {CLASS_NAMES[i]: round(w,3) for i,w in enumerate(cw)} }")

    # ── Model ─────────────────────────────────────────────────────────
    model     = GestureLSTM(INPUT_SIZE, HIDDEN, 2, NUM_CLASSES,
                             use_attention=True).to(device)
    criterion = GestureLoss(use_focal_loss=True, class_weights=cw.tolist(),
                             gamma=gamma, label_smoothing=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if verbose:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: {params:,} | Epochs: {EPOCHS} | Patience: {PATIENCE}\n")

    # ── Training loop ─────────────────────────────────────────────────
    best_val, no_imp = float('inf'), 0
    weights_path = WEIGHTS_DIR / f'best_lstm_sessions{suffix}.pth'
    history = {k: [] for k in ['train_loss', 'val_loss', 'train_acc', 'val_acc']}

    for ep in range(EPOCHS):
        tr_l, tr_a = run_epoch(model, tr_ld, criterion, optimizer, device)
        va_l, va_a = run_epoch(model, va_ld, criterion, device=device)
        scheduler.step()

        for k, v in zip(['train_loss','val_loss','train_acc','val_acc'],
                        [tr_l, va_l, tr_a, va_a]):
            history[k].append(v)

        tag = ''
        if va_l < best_val:
            best_val = va_l; no_imp = 0
            torch.save(model.state_dict(), weights_path)
            tag = ' ★'
        else:
            no_imp += 1

        if verbose:
            print(f"  Ep {ep+1:3d}/{EPOCHS}  "
                  f"Loss {tr_l:.4f}/{va_l:.4f}  "
                  f"Acc {tr_a*100:.1f}%/{va_a*100:.1f}%  "
                  f"LR {optimizer.param_groups[0]['lr']:.5f}{tag}")

        if no_imp >= PATIENCE:
            if verbose:
                print(f"\n  Early stopping tại epoch {ep+1}")
            break

    if verbose:
        print(f"\n  Weights: {weights_path}")
    return history, best_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GestureLSTM on session data')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Focal Loss gamma (default: 2.0)')
    parser.add_argument('--data', type=str, default=None,
                        help='Thư mục chứa session_*.csv')
    args = parser.parse_args()
    train_model(gamma=args.gamma, data_dir=args.data)
