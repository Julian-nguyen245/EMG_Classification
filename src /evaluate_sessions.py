"""
evaluate_sessions.py — Thực nghiệm Focal Loss Gamma trên dữ liệu session.

Chạy train + eval cho 6 giá trị gamma, lưu tất cả vào outputs/output_sessions/:
    - gamma_session_results.json      : metrics chi tiết mỗi gamma
    - loss_curves.png                 : train/val loss theo epoch
    - metrics_comparison.png          : so sánh accuracy/F1 theo gamma
    - confusion_matrix_gamma{g}.png   : confusion matrix mỗi gamma
    - per_class_f1.png                : F1 per class so sánh gamma

Dùng:
    cd "src "
    python evaluate_sessions.py
    python evaluate_sessions.py --gammas 0.0 1.0 2.0
    python evaluate_sessions.py --data ../my_sessions
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score,
                             precision_score, recall_score)

from model           import GestureLSTM
from dataset         import sEMGDataset
from session_loader  import (load_and_split_session_csv,
                              CLASS_NAMES, NUM_CLASSES)
from train_on_sessions import train_model, normalize, WEIGHTS_DIR, INPUT_SIZE, HIDDEN

# ── Output directory ───────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / 'outputs' / 'output_sessions'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
GESTURE_LABELS = [1, 2, 3]       # loại REST khi báo cáo gesture-only
GESTURE_NAMES  = CLASS_NAMES[1:] # ['fist', 'flex', 'pinch']


def load_and_evaluate(gamma, data_dir=None):
    """Load model đã train, chạy inference trên test set, trả về metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    suffix = f'_gamma{gamma}'
    weights_path = WEIGHTS_DIR / f'best_lstm_sessions{suffix}.pth'
    norm_path    = WEIGHTS_DIR / f'norm_stats_sessions{suffix}.npz'

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy weights: {weights_path}\n"
            f"Hãy chạy train_on_sessions.py --gamma {gamma} trước.")

    # Load test data
    (X_tr, y_tr), _, (X_te, y_te) = load_and_split_session_csv(data_dir)

    if norm_path.exists():
        stats = np.load(norm_path)
        mu, std = float(stats['mean']), float(stats['std'])
    else:
        mu  = X_tr.mean()
        std = X_tr.std() + 1e-8

    X_te = (X_te - mu) / std

    loader = DataLoader(sEMGDataset(X_te, y_te), batch_size=64, shuffle=False)

    # Load model
    model = GestureLSTM(INPUT_SIZE, HIDDEN, 2, NUM_CLASSES,
                        use_attention=True)
    model.load_state_dict(torch.load(weights_path, map_location=device,
                                     weights_only=False))
    model.to(device).eval()

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
            targets.extend(yb.numpy())

    preds   = np.array(preds)
    targets = np.array(targets)

    # ── Metrics: tất cả 4 lớp ─────────────────────────────────────────
    acc  = accuracy_score(targets, preds)
    f1m  = f1_score(targets, preds, average='macro',    zero_division=0)
    f1w  = f1_score(targets, preds, average='weighted', zero_division=0)
    prec = precision_score(targets, preds, average='macro', zero_division=0)
    rec  = recall_score(targets, preds, average='macro',    zero_division=0)

    # ── Per-class F1 ──────────────────────────────────────────────────
    per_class_f1 = f1_score(targets, preds, labels=list(range(NUM_CLASSES)),
                            average=None, zero_division=0)

    return {
        'accuracy':         acc,
        'f1_macro':         f1m,
        'f1_weighted':      f1w,
        'precision_macro':  prec,
        'recall_macro':     rec,
        'per_class_f1':     per_class_f1.tolist(),
        'preds':            preds,
        'targets':          targets,
    }


def run_experiment(gammas=None, data_dir=None):
    if gammas is None:
        gammas = DEFAULT_GAMMAS

    all_histories = {}
    all_metrics   = {}

    # ── 1. Train + Eval mỗi gamma ──────────────────────────────────────
    for gamma in gammas:
        print(f"\n{'='*60}")
        print(f"  gamma = {gamma}")
        print(f"{'='*60}")

        history, _ = train_model(gamma=gamma, data_dir=data_dir, verbose=True)
        all_histories[gamma] = history

        print(f"\n  Evaluating gamma={gamma}...")
        metrics = load_and_evaluate(gamma, data_dir)
        all_metrics[gamma] = metrics
        print(f"  Accuracy={metrics['accuracy']*100:.2f}%  "
              f"F1_macro={metrics['f1_macro']*100:.2f}%")

    # ── 2. Loss curves ─────────────────────────────────────────────────
    _plot_loss_curves(all_histories, gammas)

    # ── 3. Metrics comparison bar chart ────────────────────────────────
    _plot_metrics_comparison(all_metrics, gammas)

    # ── 4. Per-class F1 comparison ─────────────────────────────────────
    _plot_per_class_f1(all_metrics, gammas)

    # ── 5. Confusion matrices ──────────────────────────────────────────
    for gamma in gammas:
        _plot_confusion(all_metrics[gamma], gamma)

    # ── 6. Summary table ───────────────────────────────────────────────
    best_gamma = max(gammas, key=lambda g: all_metrics[g]['f1_macro'])
    print(f"\n{'='*70}")
    print(f"  BẢNG TỔNG HỢP — Session Gamma Experiment")
    print(f"{'='*70}")
    header = f"{'Gamma':>7} | {'Accuracy':>10} | {'F1 Macro':>10} | {'F1 Weighted':>12} | {'Precision':>10} | {'Recall':>8}"
    print(header)
    print('-' * 70)
    for gamma in gammas:
        m = all_metrics[gamma]
        star = ' ★' if gamma == best_gamma else '  '
        print(f"{gamma:>7.1f} | "
              f"{m['accuracy']*100:>9.2f}% | "
              f"{m['f1_macro']*100:>9.2f}% | "
              f"{m['f1_weighted']*100:>11.2f}% | "
              f"{m['precision_macro']*100:>9.2f}% | "
              f"{m['recall_macro']*100:>7.2f}%{star}")
    print(f"{'='*70}")
    print(f"  Best gamma (F1 macro): {best_gamma} → {all_metrics[best_gamma]['f1_macro']*100:.2f}%")

    # ── 7. Save JSON results ───────────────────────────────────────────
    results = {
        'gammas':      gammas,
        'best_gamma':  best_gamma,
        'best_f1':     all_metrics[best_gamma]['f1_macro'],
        'metrics': {
            str(g): {k: v for k, v in all_metrics[g].items()
                     if k not in ('preds', 'targets')}
            for g in gammas
        }
    }
    json_path = OUTPUT_DIR / 'gamma_session_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results JSON: {json_path}")
    print(f"  All outputs : {OUTPUT_DIR}")

    return all_histories, all_metrics


# ── Plot helpers ──────────────────────────────────────────────────────
def _plot_loss_curves(histories, gammas):
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gammas)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Session Data — Loss Curves theo Gamma', fontsize=13, fontweight='bold')

    for (gamma, hist), color in zip(histories.items(), colors):
        label = f'γ={gamma}'
        axes[0].plot(hist['train_loss'], label=label, color=color, lw=1.5)
        axes[1].plot(hist['val_loss'],   label=label, color=color, lw=1.5, ls='--')

    for ax, title in zip(axes, ['Train Loss', 'Validation Loss']):
        ax.set(title=title, xlabel='Epoch', ylabel='Focal Loss')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: loss_curves.png")


def _plot_metrics_comparison(metrics, gammas):
    metric_keys  = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    display_names = ['Accuracy', 'F1 Macro', 'F1 Weighted', 'Precision', 'Recall']
    colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#E91E63']

    x      = np.arange(len(gammas))
    width  = 0.14
    offsets = np.linspace(-(len(metric_keys)-1)/2, (len(metric_keys)-1)/2, len(metric_keys))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('Session Data — Classification Metrics theo Gamma',
                 fontsize=13, fontweight='bold')

    for i, (key, name, color) in enumerate(zip(metric_keys, display_names, colors)):
        vals = [metrics[g][key] * 100 for g in gammas]
        bars = ax.bar(x + offsets[i] * width, vals, width,
                      label=name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set(xlabel='Gamma (γ)', ylabel='Score (%)',
           xticks=x, xticklabels=[f'γ={g}' for g in gammas], ylim=(0, 108))
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_comparison.png")


def _plot_per_class_f1(metrics, gammas):
    """Heatmap F1 per class × gamma."""
    data = np.array([[metrics[g]['per_class_f1'][c]
                      for c in range(NUM_CLASSES)]
                     for g in gammas])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(data * 100, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=CLASS_NAMES, yticklabels=[f'γ={g}' for g in gammas],
                ax=ax, linewidths=0.5, vmin=0, vmax=100,
                annot_kws={'size': 10})
    ax.set_title('F1-score per Class theo Gamma (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class'); ax.set_ylabel('Gamma')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_class_f1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_class_f1.png")


def _plot_confusion(metrics, gamma):
    preds   = metrics['preds']
    targets = metrics['targets']
    cm      = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f'Confusion Matrix — γ={gamma}  '
                 f'(Acc={metrics["accuracy"]*100:.1f}%, '
                 f'F1={metrics["f1_macro"]*100:.1f}%)',
                 fontsize=11, fontweight='bold')

    for ax, data, fmt, title in [
        (axes[0], cm,      'd',    'Counts'),
        (axes[1], cm_norm, '.2f', 'Normalized'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, linewidths=0.5, annot_kws={'size': 11})
        ax.set(title=title, xlabel='Predicted', ylabel='True')

    plt.tight_layout()
    fname = f'confusion_matrix_gamma{gamma}.png'
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gamma experiment trên session data')
    parser.add_argument('--gammas', type=float, nargs='+',
                        default=DEFAULT_GAMMAS,
                        help='Danh sách gamma cần thử (default: 0 0.5 1 2 3 5)')
    parser.add_argument('--data', type=str, default=None,
                        help='Thư mục chứa session_*.csv')
    args = parser.parse_args()
    run_experiment(gammas=args.gammas, data_dir=args.data)
