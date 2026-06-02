import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, precision_score, recall_score)
from torch.utils.data import DataLoader

# Import các modules tự viết
from model import GestureLSTM
from dataset import sEMGDataset
from ninapro_loader import load_and_split_ninapro_csv

def evaluate_model(weights_suffix="", save_suffix=""):
    """
    Đánh giá model trên tập test.
    
    Tham số:
    - weights_suffix: Hậu tố file weights (vd: "_gamma2.0")
    - save_suffix: Hậu tố cho file output (vd: "_gamma2.0")
    
    Returns:
    - metrics: dict chứa accuracy, f1_macro, f1_weighted, precision_macro, recall_macro
    """
    # 1. CẤU HÌNH
    WINDOW_SIZE = 40
    STEP_SIZE = 20
    NUM_CLASSES = 13
    BATCH_SIZE = 64
    
    weights_name = f'best_lstm{weights_suffix}.pth'
    WEIGHTS_PATH = f'/home/ju1ian/Documents/EMG Classification/src/weights/{weights_name}'
    DATA_PATH = '/home/ju1ian/Documents/EMG Classification/Data (Ninapro)/processed/ninapro_db1_ready.csv'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Đang đánh giá trên thiết bị: {device}")

    # 2. LOAD DỮ LIỆU TEST
    print(" Đang load tập Test...")
    _, _, (X_test, y_test) = load_and_split_ninapro_csv(DATA_PATH, WINDOW_SIZE, STEP_SIZE)
    test_loader = DataLoader(sEMGDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # 3. LOAD MÔ HÌNH
    print(f" Đang load trọng số từ: {WEIGHTS_PATH}")
    model = GestureLSTM(input_size=10, hidden_size=128, num_layers=2, num_classes=NUM_CLASSES)
    
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError("Không tìm thấy file weights. Hãy chắc chắn bạn đã chạy train.py trước!")
        
    # Load state_dict vào model
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=False))
    model.to(device)
    model.eval() # QUAN TRỌNG: Tắt Dropout khi đánh giá

    # 4. CHẠY DỰ ĐOÁN (INFERENCE)
    all_preds = []
    all_targets = []

    print("⚡ Đang chạy suy luận (Inference)...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Lấy index của class có xác suất cao nhất
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. LỌC BỎ CLASS 0 (REST) - Chỉ giữ lại 12 cử chỉ thực sự (class 1-12)
    # Class 0 là trạng thái nghỉ, không phải cử chỉ nên không tính vào đánh giá
    mask = all_targets != 0
    all_preds_filtered   = all_preds[mask]
    all_targets_filtered = all_targets[mask]

    gesture_labels = list(range(1, 13))  # [1, 2, ..., 12]
    gesture_names  = [f"Cử chỉ {i}" for i in gesture_labels]

    print(f"\n Số mẫu sau khi loại bỏ REST (class 0): {len(all_targets_filtered)} / {len(all_targets)}")

    # 6. TÍNH TOÁN CÁC CHỈ SỐ PHÂN LOẠI
    # ---------------------------------------------------------
    acc       = accuracy_score(all_targets_filtered, all_preds_filtered)
    f1_macro  = f1_score(all_targets_filtered, all_preds_filtered, labels=gesture_labels, average='macro', zero_division=0)
    f1_weight = f1_score(all_targets_filtered, all_preds_filtered, labels=gesture_labels, average='weighted', zero_division=0)
    prec      = precision_score(all_targets_filtered, all_preds_filtered, labels=gesture_labels, average='macro', zero_division=0)
    rec       = recall_score(all_targets_filtered, all_preds_filtered, labels=gesture_labels, average='macro', zero_division=0)

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weight,
        'precision_macro': prec,
        'recall_macro': rec
    }

    print("\n" + "="*55)
    print(" CLASSIFICATION METRICS (12 Cử Chỉ, không tính REST)")
    print("="*55)
    print(f"  Accuracy         : {acc*100:.2f}%")
    print(f"  F1 (macro)       : {f1_macro*100:.2f}%")
    print(f"  F1 (weighted)    : {f1_weight*100:.2f}%")
    print(f"  Precision (macro): {prec*100:.2f}%")
    print(f"  Recall (macro)   : {rec*100:.2f}%")

    # 7. IN BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT)
    print("\n" + "="*55)
    print(" BÁO CÁO PHÂN LOẠI - 12 Cử Chỉ Tay (Không tính REST)")
    print("="*55)
    print(classification_report(
        all_targets_filtered,
        all_preds_filtered,
        labels=gesture_labels,
        target_names=gesture_names,
        zero_division=0
    ))

    # 8. VẼ MA TRẬN NHẦM LẪN (CONFUSION MATRIX) - 12x12
    print("🎨 Đang vẽ Confusion Matrix (12 cử chỉ)...")
    cm = confusion_matrix(all_targets_filtered, all_preds_filtered, labels=gesture_labels)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=gesture_names,
        yticklabels=gesture_names,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.xlabel('Dự đoán của AI (Predicted)', fontsize=13, fontweight='bold')
    plt.ylabel('Thực tế (Ground Truth)', fontsize=13, fontweight='bold')
    plt.title(f'Ma trận nhầm lẫn - 12 Cử Chỉ Tay (gamma={save_suffix.replace("_gamma", "") or "default"})',
              fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Lưu file ảnh
    os.makedirs('/home/ju1ian/Documents/EMG Classification/outputs', exist_ok=True)
    save_path = f'/home/ju1ian/Documents/EMG Classification/outputs/confusion_matrix{save_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Đã lưu ma trận nhầm lẫn tại: {save_path}")

    return metrics

if __name__ == "__main__":
    evaluate_model()