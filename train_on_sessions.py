"""
Train GestureLSTM trực tiếp trên 3 session CSV (dummy hoặc thực tế).
Dùng lại src/model.py, src/losses.py, src/dataset.py, src/data_utils.py.
Weights → src/weights/best_lstm_sessions.pth
Norm    → src/weights/norm_stats_sessions.npz
"""

import os, sys, math
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

SRC = Path(__file__).parent / "src "
sys.path.insert(0, str(SRC))
from model      import GestureLSTM
from losses     import GestureLoss
from dataset    import sEMGDataset
from data_utils import create_sliding_windows

ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "dummy_sessions"
WEIGHTS_DIR  = ROOT / "src" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = WEIGHTS_DIR / "best_lstm_sessions.pth"
NORM_PATH    = WEIGHTS_DIR / "norm_stats_sessions.npz"

WINDOW = 40;  STEP  = 20;  BATCH = 64
EPOCHS = 60;  LR    = 1e-3; HIDDEN = 128
WARMUP = 5;   PATIENCE = 12

LABEL_MAP   = {'rest': 0, 'fist': 1, 'flex': 2, 'pinch': 3}
CLASS_NAMES = ['rest', 'fist', 'flex', 'pinch']
TRAIN_REPS  = set(range(1, 15))
VAL_REPS    = set(range(15, 18))
TEST_REPS   = set(range(18, 21))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_split():
    frames = [pd.read_csv(p) for p in sorted(DATA_DIR.glob("session_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows từ {len(frames)} sessions")
    result = {}
    for name, reps in [('train', TRAIN_REPS), ('val', VAL_REPS), ('test', TEST_REPS)]:
        sub  = df[df['repetition'].isin(reps)].reset_index(drop=True)
        sig  = sub['filtered_emg'].values.astype(np.float32).reshape(-1, 1)
        lbl  = sub['label'].map(LABEL_MAP).values.astype(np.int64)
        X, y = create_sliding_windows(sig, lbl, WINDOW, STEP)
        result[name] = (X, y)
        print(f"  {name:5s}: {len(X):,} windows")
    return result


def lr_lambda(ep):
    if ep < WARMUP: return (ep + 1) / WARMUP
    p = (ep - WARMUP) / max(EPOCHS - WARMUP, 1)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))


def run_epoch(model, loader, criterion, opt=None):
    model.train() if opt else model.eval()
    tot, correct, total = 0., 0, 0
    ctx = torch.enable_grad() if opt else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb); loss = criterion(out, yb)
            if opt:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot     += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total   += len(xb)
    return tot / total, correct / total


def main():
    splits = load_split()
    X_tr, y_tr = splits['train']
    X_va, y_va = splits['val']

    mu = X_tr.mean(); sd = X_tr.std() + 1e-8
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    np.savez(NORM_PATH, mean=np.array([mu]), std=np.array([sd]))

    cw = np.clip(compute_class_weight('balanced',
                  classes=np.arange(4), y=y_tr), 0.2, 10.0)
    print(f"\nClass weights: { {CLASS_NAMES[i]: round(w,3) for i,w in enumerate(cw)} }")

    tr_ld = DataLoader(sEMGDataset(X_tr, y_tr), BATCH, shuffle=True,  num_workers=0)
    va_ld = DataLoader(sEMGDataset(X_va, y_va), BATCH, shuffle=False, num_workers=0)

    model     = GestureLSTM(1, HIDDEN, 2, 4, use_attention=True).to(device)
    criterion = GestureLoss(True, cw.tolist(), gamma=2.0, label_smoothing=0.1).to(device)
    opt       = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {params:,}\n")
    print(f"Training {EPOCHS} epochs (warmup={WARMUP}, patience={PATIENCE})...\n")

    best_val, no_imp = float('inf'), 0
    for ep in range(EPOCHS):
        tr_l, tr_a = run_epoch(model, tr_ld, criterion, opt)
        va_l, va_a = run_epoch(model, va_ld, criterion)
        scheduler.step()
        tag = ''
        if va_l < best_val:
            best_val = va_l; no_imp = 0
            torch.save(model.state_dict(), WEIGHTS_PATH); tag = ' ★'
        else:
            no_imp += 1
        print(f"Ep {ep+1:3d}/{EPOCHS}  Loss {tr_l:.4f}/{va_l:.4f}  "
              f"Acc {tr_a*100:.1f}%/{va_a*100:.1f}%  "
              f"LR {opt.param_groups[0]['lr']:.5f}{tag}")
        if no_imp >= PATIENCE:
            print(f"\nEarly stopping tại epoch {ep+1}"); break

    print(f"\nWeights: {WEIGHTS_PATH}")
    print(f"Norm   : {NORM_PATH}")


if __name__ == '__main__':
    main()
