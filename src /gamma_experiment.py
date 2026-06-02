"""
Thí nghiệm Focal Loss Gamma — sEMG Gesture Classification

Chạy train + evaluate với các gamma khác nhau, so sánh:
1. Loss curves (train/val) theo epoch
2. Classification metrics (F1, Accuracy, Precision, Recall)
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

from train import train_model
from evaluate import evaluate_model

# ==========================================
# CẤU HÌNH THÍ NGHIỆM
# ==========================================
GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
OUTPUT_DIR = '/home/ju1ian/Documents/EMG Classification/outputs'


def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_histories = {}   # gamma → history dict
    all_metrics   = {}   # gamma → metrics dict

    # ── 1. TRAIN + EVALUATE cho từng gamma ──────────────────────
    for gamma in GAMMAS:
        suffix = f"_gamma{gamma}"
        print("\n" + "=" * 70)
        print(f"  THÍ NGHIỆM: gamma = {gamma}")
        print("=" * 70)

        # Train
        print(f"\n--- [TRAIN] gamma={gamma} ---")
        history = train_model(gamma=gamma, weights_suffix=suffix)
        all_histories[gamma] = history

        # Evaluate
        print(f"\n--- [EVALUATE] gamma={gamma} ---")
        metrics = evaluate_model(weights_suffix=suffix, save_suffix=suffix)
        all_metrics[gamma] = metrics

    # ── 2. PLOT LOSS CURVES ─────────────────────────────────────
    print("\n\n📊 Đang vẽ đồ thị so sánh Loss curves...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(GAMMAS)))

    # Train Loss
    ax1 = axes[0]
    for (gamma, hist), color in zip(all_histories.items(), colors):
        ax1.plot(hist['train_loss'], label=f'γ={gamma}', color=color, linewidth=1.5)
    ax1.set_title('Train Loss theo Epoch', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Val Loss
    ax2 = axes[1]
    for (gamma, hist), color in zip(all_histories.items(), colors):
        ax2.plot(hist['val_loss'], label=f'γ={gamma}', color=color, linewidth=1.5)
    ax2.set_title('Validation Loss theo Epoch', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('So sánh Focal Loss Curves với các γ (gamma) khác nhau',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    loss_plot_path = os.path.join(OUTPUT_DIR, 'gamma_loss_curves.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {loss_plot_path}")

    # ── 3. PLOT METRICS BAR CHART ───────────────────────────────
    print("📊 Đang vẽ đồ thị so sánh Metrics...")

    metric_names = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    display_names = ['Accuracy', 'F1 (macro)', 'F1 (weighted)', 'Precision', 'Recall']

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(GAMMAS))
    width = 0.15
    offsets = np.arange(len(metric_names)) - len(metric_names) / 2 + 0.5

    bar_colors = ['#2196F3', '#4CAF50', '#8BC34A', '#FF9800', '#E91E63']

    for i, (metric, dname) in enumerate(zip(metric_names, display_names)):
        values = [all_metrics[g][metric] * 100 for g in GAMMAS]
        bars = ax.bar(x + offsets[i] * width, values, width,
                      label=dname, color=bar_colors[i], alpha=0.85)
        # Ghi giá trị lên đầu mỗi cột
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xlabel('Gamma (γ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('So sánh Classification Metrics theo Focal Loss γ', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'γ={g}' for g in GAMMAS], fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()

    metrics_plot_path = os.path.join(OUTPUT_DIR, 'gamma_metrics_comparison.png')
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu: {metrics_plot_path}")

    # ── 4. BẢNG TỔNG HỢP ───────────────────────────────────────
    print("\n\n" + "=" * 85)
    print("  BẢNG TỔNG HỢP KẾT QUẢ THÍ NGHIỆM FOCAL LOSS GAMMA")
    print("=" * 85)
    header = f"{'Gamma':>8} | {'Accuracy':>10} | {'F1 macro':>10} | {'F1 weighted':>12} | {'Precision':>10} | {'Recall':>10}"
    print(header)
    print("-" * 85)

    best_gamma = None
    best_f1 = -1

    for gamma in GAMMAS:
        m = all_metrics[gamma]
        row = (f"{gamma:>8.1f} | "
               f"{m['accuracy']*100:>9.2f}% | "
               f"{m['f1_macro']*100:>9.2f}% | "
               f"{m['f1_weighted']*100:>11.2f}% | "
               f"{m['precision_macro']*100:>9.2f}% | "
               f"{m['recall_macro']*100:>9.2f}%")
        print(row)
        if m['f1_macro'] > best_f1:
            best_f1 = m['f1_macro']
            best_gamma = gamma

    print("-" * 85)
    print(f"  🏆 Gamma tốt nhất (theo F1 macro): γ = {best_gamma} → F1 = {best_f1*100:.2f}%")
    print("=" * 85)

    # Lưu kết quả ra JSON
    results = {
        'gammas': GAMMAS,
        'metrics': {str(g): m for g, m in all_metrics.items()},
        'best_gamma': best_gamma,
        'best_f1_macro': best_f1
    }
    json_path = os.path.join(OUTPUT_DIR, 'gamma_experiment_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Kết quả JSON: {json_path}")

    return all_histories, all_metrics


if __name__ == "__main__":
    run_experiment()
