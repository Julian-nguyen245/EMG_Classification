"""
Train + Evaluate GestureLSTM trên 3 dummy sessions.

Pipeline:
  1. Load 3 CSV (session_1/2/3 trong dummy_sessions/)
  2. Label encode: rest=0, fist=1, flex=2, pinch=3
  3. Sliding Window W=40 (400ms), S=20 (200ms, overlap 50%)
  4. Split theo repetition: rep 1-14 → train | rep 15-17 → val | rep 18-20 → test
  5. Z-score normalization (fit trên train)
  6. GestureLSTM + Attention (input=1, hidden=64, classes=4)
  7. Focal Loss + class weights + label smoothing
  8. Warmup 5ep + Cosine Decay, 60 epochs, early stopping
  9. Confusion matrix + classification report
"""

import os, math, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
import seaborn as sns
warnings.filterwarnings('ignore')

# ── Cấu hình ──────────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "dummy_sessions"
OUTPUT_DIR  = Path(__file__).parent / "outputs_dummy"
OUTPUT_DIR.mkdir(exist_ok=True)

FS          = 100
WINDOW      = 40        # 400ms
STEP        = 20        # 200ms, overlap 50%
BATCH       = 64
EPOCHS      = 60
LR          = 1e-3
HIDDEN      = 64
PATIENCE    = 12
WARMUP      = 5

LABEL_MAP   = {'rest': 0, 'fist': 1, 'flex': 2, 'pinch': 3}
CLASS_NAMES = ['rest', 'fist', 'flex', 'pinch']

# Repetition split (NinaPro convention)
TRAIN_REPS  = set(range(1, 15))   # 1-14
VAL_REPS    = set(range(15, 18))  # 15-17
TEST_REPS   = set(range(18, 21))  # 18-20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


# ── Model ─────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.w = nn.Linear(h, 1, bias=False)

    def forward(self, x):            # x: (B, T, H)
        a = torch.softmax(self.w(x), dim=1)   # (B, T, 1)
        return (x * a).sum(dim=1)             # (B, H)


class GestureLSTM(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=2, n_class=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.attn = TemporalAttention(hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(self.norm(self.attn(out))))


# ── Dataset ───────────────────────────────────────────────────────────────────
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):  return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Data loading & windowing ──────────────────────────────────────────────────
def load_sessions(data_dir: Path) -> pd.DataFrame:
    frames = []
    for p in sorted(data_dir.glob("session_*.csv")):
        df = pd.read_csv(p)
        df['session'] = p.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def make_windows(df: pd.DataFrame, rep_set: set):
    """Sliding window trên các rep thuộc rep_set (rep=0 = rest transition → bỏ)."""
    mask = df['repetition'].isin(rep_set)
    sub  = df[mask].reset_index(drop=True)

    sig  = sub['filtered_emg'].values.astype(np.float32).reshape(-1, 1)
    lbl  = sub['label'].map(LABEL_MAP).values.astype(np.int64)

    X, y = [], []
    for i in range(0, len(sig) - WINDOW + 1, STEP):
        X.append(sig[i:i + WINDOW])
        y.append(lbl[i + WINDOW - 1])      # label của mẫu cuối window
    return np.array(X), np.array(y)


# ── Loss ──────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.gamma  = gamma
        self.smooth = label_smoothing
        self.register_buffer('weight', weight)

    def forward(self, logits, targets):
        w = self.weight.to(logits.device)
        ce = F.cross_entropy(logits, targets, weight=w,
                             reduction='none', label_smoothing=self.smooth)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── LR Schedule ──────────────────────────────────────────────────────────────
def lr_lambda(ep):
    if ep < WARMUP:
        return (ep + 1) / WARMUP
    p = (ep - WARMUP) / max(EPOCHS - WARMUP, 1)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))


# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot_loss += loss.item() * len(xb)
            correct  += (out.argmax(1) == yb).sum().item()
            total    += len(xb)
    return tot_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load data
    print("Loading sessions...")
    df = load_sessions(DATA_DIR)
    print(f"  Total rows: {len(df):,} | Sessions: {df['session'].nunique()}")

    X_tr, y_tr = make_windows(df, TRAIN_REPS)
    X_va, y_va = make_windows(df, VAL_REPS)
    X_te, y_te = make_windows(df, TEST_REPS)
    print(f"  Windows → Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")

    # 2. Z-score normalization (fit on train)
    mu  = X_tr.mean()
    std = X_tr.std() + 1e-8
    X_tr = (X_tr - mu) / std
    X_va = (X_va - mu) / std
    X_te = (X_te - mu) / std

    # 3. Class weights
    classes    = np.unique(y_tr)
    cw         = compute_class_weight('balanced', classes=classes, y=y_tr)
    cw_tensor  = torch.tensor(cw, dtype=torch.float32)
    print(f"\nClass weights: { {CLASS_NAMES[c]: round(w,3) for c, w in zip(classes, cw)} }")

    # 4. DataLoaders
    tr_ld = DataLoader(EMGDataset(X_tr, y_tr), BATCH, shuffle=True)
    va_ld = DataLoader(EMGDataset(X_va, y_va), BATCH)
    te_ld = DataLoader(EMGDataset(X_te, y_te), BATCH)

    # 5. Model
    model     = GestureLSTM(input_size=1, hidden=HIDDEN, layers=2, n_class=4).to(device)
    params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = FocalLoss(cw_tensor, gamma=1.5, label_smoothing=0.05).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"\nModel: GestureLSTM (1ch→{HIDDEN}→4) | Params: {params:,}")

    # 6. Training
    print(f"\nTraining {EPOCHS} epochs (warmup={WARMUP}, patience={PATIENCE})...\n")
    best_val, no_imp = float('inf'), 0
    hist = {k: [] for k in ['tr_loss','va_loss','tr_acc','va_acc']}

    for ep in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, tr_ld, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, va_ld, criterion)
        scheduler.step()

        hist['tr_loss'].append(tr_loss)
        hist['va_loss'].append(va_loss)
        hist['tr_acc'].append(tr_acc)
        hist['va_acc'].append(va_acc)

        tag = ''
        if va_loss < best_val:
            best_val = va_loss
            no_imp   = 0
            torch.save(model.state_dict(), OUTPUT_DIR / 'best_model.pth')
            tag = ' ★'
        else:
            no_imp += 1

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Ep {ep+1:3d}/{EPOCHS}  "
              f"Loss {tr_loss:.4f}/{va_loss:.4f}  "
              f"Acc {tr_acc*100:.1f}%/{va_acc*100:.1f}%  "
              f"LR {lr_now:.5f}{tag}")

        if no_imp >= PATIENCE:
            print(f"\nEarly stopping at epoch {ep+1}")
            break

    # 7. Load best & evaluate on test
    model.load_state_dict(torch.load(OUTPUT_DIR / 'best_model.pth',
                                     map_location=device))
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            all_pred.extend(pred)
            all_true.extend(yb.numpy())

    acc  = accuracy_score(all_true, all_pred)
    f1m  = f1_score(all_true, all_pred, average='macro', zero_division=0)
    print(f"\n{'='*55}")
    print(f"TEST SET RESULTS")
    print(f"{'='*55}")
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"F1 Macro  : {f1m*100:.2f}%")
    print(f"\n{classification_report(all_true, all_pred, target_names=CLASS_NAMES, zero_division=0)}")

    # 8. Plots
    _plot_loss(hist)
    _plot_confusion(all_true, all_pred)
    _plot_signal_preview()
    print(f"\nPlots saved to: {OUTPUT_DIR}")


# ── Visualization ─────────────────────────────────────────────────────────────
def _plot_loss(hist):
    ep = range(1, len(hist['tr_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Training History — Dummy Sessions', fontsize=12, fontweight='bold')

    axes[0].plot(ep, hist['tr_loss'], label='Train', color='#f0883e')
    axes[0].plot(ep, hist['va_loss'], label='Val',   color='#58a6ff', ls='--')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Focal Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [v*100 for v in hist['tr_acc']], label='Train', color='#3fb950')
    axes[1].plot(ep, [v*100 for v in hist['va_acc']], label='Val',   color='#d29922', ls='--')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_confusion(true, pred):
    cm = confusion_matrix(true, pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Confusion Matrix — Test Set', fontsize=12, fontweight='bold')

    for ax, data, fmt, title in [
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f',  'Normalized (recall per class)'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5)
        ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def _plot_signal_preview():
    """Plot 30s tín hiệu từ session_1 để kiểm tra độ thực tế."""
    path = Path(__file__).parent / 'dummy_sessions' / 'session_1.csv'
    if not path.exists():
        return
    df  = pd.read_csv(path)
    sub = df.iloc[:3000]    # 30s
    t   = sub['timestamp_ms'] / 1000

    color_map = {'rest': '#888', 'fist': '#e94560', 'flex': '#00b37e', 'pinch': '#c77dff'}

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle('Signal Preview — Session 1 (first 30s)', fontsize=12, fontweight='bold')
    fig.subplots_adjust(hspace=0.35)

    for ax, col, title, color in [
        (axes[0], 'raw_emg',      'Raw EMG (mV)',      '#f0883e'),
        (axes[1], 'filtered_emg', 'Filtered EMG (mV)', '#58a6ff'),
    ]:
        ax.plot(t, sub[col], color=color, lw=0.7, alpha=0.85)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.25)

        # Tô màu nền theo nhãn
        prev_lbl, prev_t = sub['label'].iloc[0], t.iloc[0]
        for i in range(1, len(sub)):
            lbl = sub['label'].iloc[i]
            if lbl != prev_lbl or i == len(sub) - 1:
                ax.axvspan(prev_t, t.iloc[i], alpha=0.08,
                           color=color_map.get(prev_lbl, '#888'), lw=0)
                prev_lbl, prev_t = lbl, t.iloc[i]

    axes[1].set_xlabel('Time (s)')

    # Legend nhãn
    from matplotlib.patches import Patch
    handles = [Patch(color=v, label=k, alpha=0.4) for k, v in color_map.items()]
    axes[0].legend(handles=handles, loc='upper right', fontsize=8, ncol=4)

    plt.savefig(OUTPUT_DIR / 'signal_preview.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
