"""
evaluate.py — Đánh giá GestureLSTM trên hai nguồn dữ liệu:

  mode='ninapro'  : NinaPro DB1, 10 kênh, 13 lớp (mặc định cũ)
  mode='sessions' : CSV session tự thu / dummy, 1 kênh, 4 lớp
                    (rest=0, fist=1, flex=2, pinch=3)

Cách dùng:
  # Trong code:
  from evaluate import evaluate_model
  metrics = evaluate_model()                          # ninapro
  metrics = evaluate_model(mode='sessions')           # sessions

  # Command line:
  python evaluate.py                                  # ninapro
  python evaluate.py --mode sessions                  # sessions
  python evaluate.py --mode sessions --data path/to/sessions/
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score,
                             precision_score, recall_score)

from model      import GestureLSTM
from dataset    import sEMGDataset
from data_utils import create_sliding_windows

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
WEIGHTS_DIR  = ROOT / "src" / "weights"
OUTPUT_DIR   = ROOT / "outputs"
NINAPRO_CSV  = ROOT / "Data (Ninapro)" / "processed" / "ninapro_db1_ready.csv"
SESSIONS_DIR = ROOT / "dummy_sessions"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Session config ────────────────────────────────────────────────────────────
SESSION_LABEL_MAP  = {'rest': 0, 'fist': 1, 'flex': 2, 'pinch': 3}
SESSION_CLASSES    = ['rest', 'fist', 'flex', 'pinch']
SESSION_GESTURE_CLASSES = ['fist', 'flex', 'pinch']   # loại REST khi báo cáo gesture-only
SESSION_TEST_REPS  = set(range(18, 21))                # rep 18-20

# ── Shared helpers ────────────────────────────────────────────────────────────
def _load_model(weights_path: Path, input_size: int, num_classes: int, device):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy weights: {weights_path}\n"
            "Hãy chạy train.py (ninapro) hoặc train_on_sessions.py (sessions) trước."
        )
    model = GestureLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        use_attention=True,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model


def _run_inference(model, loader, device):
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p  = model(xb).argmax(dim=1).cpu().numpy()
            preds.extend(p)
            targets.extend(yb.numpy())
    return np.array(preds), np.array(targets)


def _print_metrics(preds, targets, labels, names, title=""):
    acc   = accuracy_score(targets, preds)
    f1m   = f1_score(targets, preds, labels=labels, average='macro',    zero_division=0)
    f1w   = f1_score(targets, preds, labels=labels, average='weighted', zero_division=0)
    prec  = precision_score(targets, preds, labels=labels, average='macro', zero_division=0)
    rec   = recall_score(targets, preds, labels=labels, average='macro',    zero_division=0)

    sep = "=" * 56
    print(f"\n{sep}")
    print(f" {title}")
    print(sep)
    print(f"  Accuracy         : {acc*100:.2f}%")
    print(f"  F1 (macro)       : {f1m*100:.2f}%")
    print(f"  F1 (weighted)    : {f1w*100:.2f}%")
    print(f"  Precision (macro): {prec*100:.2f}%")
    print(f"  Recall (macro)   : {rec*100:.2f}%")
    print(sep)
    print(classification_report(targets, preds,
                                 labels=labels, target_names=names,
                                 zero_division=0))
    return dict(accuracy=acc, f1_macro=f1m, f1_weighted=f1w,
                precision_macro=prec, recall_macro=rec)


def _save_confusion(preds, targets, labels, names, save_path: Path, title=""):
    cm      = confusion_matrix(targets, preds, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n       = len(names)
    fs      = max(10, n * 1.1)

    fig, axes = plt.subplots(1, 2, figsize=(fs * 2, fs * 0.95))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    for ax, data, fmt, sub_title in [
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalized (recall)'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=names, yticklabels=names,
                    ax=ax, linewidths=0.4, linecolor='gray',
                    annot_kws={'size': max(7, 11 - n // 3)})
        ax.set_title(sub_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True',      fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix: {save_path}")


# ════════════════════════════════════════════════════════════════════════════
# MODE: ninapro
# ════════════════════════════════════════════════════════════════════════════
def _eval_ninapro(weights_suffix: str, save_suffix: str) -> dict:
    from ninapro_loader import load_and_split_ninapro_csv

    WINDOW = 40
    STEP   = 20
    BATCH  = 64

    weights_path = WEIGHTS_DIR / f"best_lstm{weights_suffix}.pth"
    norm_path    = WEIGHTS_DIR / f"norm_stats{weights_suffix}.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode  : NinaPro DB1 (10ch / 13 classes)")

    # Data
    print("Loading NinaPro test split...")
    _, _, (X_test, y_test) = load_and_split_ninapro_csv(
        str(NINAPRO_CSV), WINDOW, STEP
    )
    if norm_path.exists():
        stats  = np.load(norm_path)
        X_test = (X_test - stats['mean']) / stats['std']
        print(f"Normalization applied from: {norm_path.name}")

    loader = DataLoader(sEMGDataset(X_test, y_test), batch_size=BATCH, shuffle=False)

    # Model
    model = _load_model(weights_path, input_size=10, num_classes=13, device=device)
    print(f"Weights: {weights_path.name}")

    # Inference
    print("Running inference...")
    preds, targets = _run_inference(model, loader, device)

    # Filter REST (class 0) — NinaPro convention
    mask    = targets != 0
    preds_g = preds[mask];   targets_g = targets[mask]
    g_lbls  = list(range(1, 13))
    g_names = [f"Cử chỉ {i}" for i in g_lbls]

    print(f"Test samples: {len(targets):,}  →  gestures only: {mask.sum():,}")

    metrics = _print_metrics(preds_g, targets_g, g_lbls, g_names,
                              title="NinaPro — 12 gestures (REST excluded)")

    tag = save_suffix or f"_ninapro{weights_suffix}"
    _save_confusion(preds_g, targets_g, g_lbls, g_names,
                    OUTPUT_DIR / f"confusion_matrix{tag}.png",
                    title=f"NinaPro DB1 — Confusion Matrix  {weights_suffix or ''}")
    return metrics


# ════════════════════════════════════════════════════════════════════════════
# MODE: sessions
# ════════════════════════════════════════════════════════════════════════════
def _load_session_test(data_dir: Path, include_rest: bool = True):
    """Load và window-ize tập test (rep 18-20) từ các session CSV."""
    frames = []
    for p in sorted(data_dir.glob("session_*.csv")):
        df = pd.read_csv(p)
        df['_src'] = p.stem
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"Không tìm thấy session_*.csv trong: {data_dir}\n"
            "Hãy chạy generate_dummy.py hoặc cung cấp --data đúng thư mục."
        )

    df_all = pd.concat(frames, ignore_index=True)
    sub    = df_all[df_all['repetition'].isin(SESSION_TEST_REPS)].reset_index(drop=True)

    sig  = sub['filtered_emg'].values.astype(np.float32).reshape(-1, 1)
    lbl  = sub['label'].map(SESSION_LABEL_MAP).values.astype(np.int64)
    X, y = create_sliding_windows(sig, lbl, window_size=40, step_size=20)
    return X, y, df_all


def _eval_sessions(data_dir: Path, weights_suffix: str, include_rest: bool) -> dict:
    weights_path = WEIGHTS_DIR / f"best_lstm_sessions{weights_suffix}.pth"
    norm_path    = WEIGHTS_DIR / f"norm_stats_sessions{weights_suffix}.npz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : Sessions (1ch / 4 classes)")
    print(f"Data   : {data_dir}")

    # Data
    print("Loading session test split (rep 18-20)...")
    X_test, y_test, df_all = _load_session_test(data_dir, include_rest)

    if norm_path.exists():
        stats  = np.load(norm_path)
        X_test = (X_test - stats['mean']) / stats['std']
        print(f"Normalization applied from: {norm_path.name}")
    else:
        # Fallback: z-score bằng chính tập test
        mu = X_test.mean(); sd = X_test.std() + 1e-8
        X_test = (X_test - mu) / sd
        print("Normalization: z-score từ test set (fallback)")

    loader = DataLoader(sEMGDataset(X_test, y_test), batch_size=64, shuffle=False)

    # Model
    model = _load_model(weights_path, input_size=1, num_classes=4, device=device)
    print(f"Weights: {weights_path.name}")

    # Inference
    print("Running inference...")
    preds, targets = _run_inference(model, loader, device)
    print(f"Test windows: {len(targets):,}")

    # ── Report 1: tất cả 4 classes (kể cả REST) ──────────────────────
    metrics = _print_metrics(preds, targets,
                              labels=list(range(4)),
                              names=SESSION_CLASSES,
                              title="Sessions — 4 classes (REST included)")
    _save_confusion(preds, targets,
                    labels=list(range(4)),
                    names=SESSION_CLASSES,
                    save_path=OUTPUT_DIR / f"confusion_matrix_sessions{weights_suffix}.png",
                    title="Sessions — Confusion Matrix (4 classes)")

    # ── Report 2: chỉ 3 gesture (loại REST) ──────────────────────────
    mask = targets != 0
    if mask.sum() > 0:
        g_labels = [1, 2, 3]
        _print_metrics(preds[mask], targets[mask],
                       labels=g_labels,
                       names=SESSION_GESTURE_CLASSES,
                       title="Sessions — 3 gestures only (REST excluded)")
        _save_confusion(preds[mask], targets[mask],
                        labels=g_labels,
                        names=SESSION_GESTURE_CLASSES,
                        save_path=OUTPUT_DIR / f"confusion_matrix_sessions_gestures{weights_suffix}.png",
                        title="Sessions — Confusion Matrix (gestures only)")

    # ── Signal preview ────────────────────────────────────────────────
    _plot_session_preview(df_all, weights_suffix)

    return metrics


def _plot_session_preview(df_all: pd.DataFrame, suffix: str = ""):
    """Plot 30s đầu của session_1 với màu nền theo nhãn."""
    sub = df_all[df_all['_src'] == 'session_1'].iloc[:3000]
    if sub.empty:
        return

    t   = sub['timestamp_ms'].values / 1000
    clr = {'rest': '#888', 'fist': '#e94560', 'flex': '#00b37e', 'pinch': '#c77dff'}

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle('Signal Preview — session_1 (first 30 s)', fontsize=11, fontweight='bold')
    fig.subplots_adjust(hspace=0.4)

    for ax, col, ylabel, color in [
        (axes[0], 'raw_emg',      'Raw EMG (mV)',      '#f0883e'),
        (axes[1], 'filtered_emg', 'Filtered EMG (mV)', '#58a6ff'),
    ]:
        ax.plot(t, sub[col].values, color=color, lw=0.7, alpha=0.85)
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)

        # Tô nền theo label
        lbls = sub['label'].values
        prev, pt = lbls[0], t[0]
        for i in range(1, len(sub)):
            if lbls[i] != prev or i == len(sub) - 1:
                ax.axvspan(pt, t[i], alpha=0.10, color=clr.get(prev, '#888'), lw=0)
                prev, pt = lbls[i], t[i]

    from matplotlib.patches import Patch
    axes[0].legend(
        [Patch(color=v, label=k, alpha=0.5) for k, v in clr.items()],
        list(clr.keys()), loc='upper right', fontsize=8, ncol=4
    )
    axes[1].set_xlabel('Time (s)')

    out = OUTPUT_DIR / f"signal_preview_sessions{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Signal preview : {out}")


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════
def evaluate_model(
    mode: str = 'ninapro',
    weights_suffix: str = "",
    save_suffix: str = "",
    data_dir: str | None = None,
    include_rest: bool = True,
) -> dict:
    """
    Đánh giá model và lưu kết quả vào outputs/.

    Tham số:
        mode           : 'ninapro' | 'sessions'
        weights_suffix : hậu tố file weights,  vd "_gamma2.0"
        save_suffix    : hậu tố file output,   vd "_gamma2.0"
        data_dir       : (sessions) thư mục chứa session_*.csv
                         mặc định: dummy_sessions/ trong project root
        include_rest   : (sessions) True = báo cáo cả REST trong confusion matrix

    Returns:
        dict: accuracy, f1_macro, f1_weighted, precision_macro, recall_macro
    """
    mode = mode.lower()
    if mode == 'ninapro':
        return _eval_ninapro(weights_suffix, save_suffix)
    elif mode == 'sessions':
        d = Path(data_dir) if data_dir else SESSIONS_DIR
        return _eval_sessions(d, weights_suffix, include_rest)
    else:
        raise ValueError(f"mode phải là 'ninapro' hoặc 'sessions', nhận được: '{mode}'")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GestureLSTM")
    parser.add_argument('--mode',    default='ninapro',
                        choices=['ninapro', 'sessions'],
                        help="Nguồn dữ liệu: ninapro | sessions")
    parser.add_argument('--suffix',  default='',
                        help="Hậu tố weights, vd: _gamma2.0")
    parser.add_argument('--data',    default=None,
                        help="(sessions) Thư mục chứa session_*.csv")
    parser.add_argument('--no-rest', action='store_true',
                        help="(sessions) Loại REST khỏi báo cáo chính")
    args = parser.parse_args()

    evaluate_model(
        mode           = args.mode,
        weights_suffix = args.suffix,
        save_suffix    = args.suffix,
        data_dir       = args.data,
        include_rest   = not args.no_rest,
    )
