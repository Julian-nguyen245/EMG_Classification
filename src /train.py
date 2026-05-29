import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from losses import GestureLoss

# Import các class chúng ta vừa viết
from dataset import sEMGDataset
from model import GestureLSTM
from ninapro_loader import load_and_split_ninapro_csv

def train_model():
    # 1. CẤU HÌNH THÔNG SỐ (Hyperparameters)
    # ---------------------------------------------------------
    EPOCHS         = 50
    BATCH_SIZE     = 64
    LEARNING_RATE  = 0.001
    WINDOW_SIZE    = 40
    STEP_SIZE      = 20
    NUM_CLASSES    = 13   # 0=REST, 1-12=12 cử chỉ
    HIDDEN_SIZE    = 128  # Tăng từ 64 lên 128
    PATIENCE       = 10   # Early stopping: dừng nếu val_loss không giảm sau 10 epoch

    WEIGHTS_DIR  = '/home/ju1ian/Documents/EMG Classification/src/weights'
    WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'best_lstm.pth')

    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang sử dụng thiết bị: {device}")

    # 2. CHUẨN BỊ DỮ LIỆU
    # ---------------------------------------------------------
    print("⏳ Đang chuẩn bị dữ liệu...")
    file_path = '/home/ju1ian/Documents/EMG Classification/Data (Ninapro)/processed/ninapro_db1_ready.csv'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy {file_path}!")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_ninapro_csv(
        file_path, WINDOW_SIZE, STEP_SIZE
    )

    # Đưa vào Dataset & DataLoader
    train_loader = DataLoader(sEMGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(sEMGDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"✅ Train: {len(X_train)} windows | Val: {len(X_val)} windows | Test: {len(X_test)} windows")

    # 3. TÍNH CLASS WEIGHTS TỰ ĐỘNG (Xử lý mất cân bằng dữ liệu)
    # ---------------------------------------------------------
    # Tính trọng số nghịch đảo tần suất cho từng class
    unique_classes = np.unique(y_train)
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )

    # Đảm bảo weights có đủ NUM_CLASSES phần tử (nếu có class nào vắng mặt trong train)
    class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls, w in zip(unique_classes, raw_weights):
        class_weights[cls] = w

    # Giới hạn weight tối đa để tránh gradient bùng nổ
    class_weights = np.clip(class_weights, 0.2, 10.0)

    print("\n📊 Class Weights (để cân bằng dữ liệu):")
    for i, w in enumerate(class_weights):
        label = "REST" if i == 0 else f"Cử chỉ {i}"
        print(f"   Class {i:2d} ({label:10s}): {w:.3f}")

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 4. KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER
    # ---------------------------------------------------------
    model = GestureLSTM(
        input_size=10,
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        num_classes=NUM_CLASSES
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: GestureLSTM | Hidden: {HIDDEN_SIZE} | Params: {total_params:,}")

    # Focal Loss + Class Weights: kép xử lý mất cân bằng
    criterion = GestureLoss(use_focal_loss=True, class_weights=class_weights.tolist())
    # Chuyển weight tensor trong criterion lên đúng device
    criterion.ce_loss.weight = criterion.ce_loss.weight.to(device)

    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # ReduceLROnPlateau: Giảm LR × 0.5 nếu val_loss không giảm sau 5 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Tạo thư mục lưu weights
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # 5. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
    # ---------------------------------------------------------
    print(f"\n🔥 Bắt đầu huấn luyện ({EPOCHS} epochs, early stopping patience={PATIENCE})...\n")

    best_val_loss    = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # --- PHA TRAIN ---
        model.train()
        train_loss    = 0.0
        train_correct = 0
        total_train   = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping: tránh gradient bùng nổ
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * inputs.size(0)
            _, predicted   = torch.max(outputs.data, 1)
            total_train   += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / total_train
        train_acc      = 100 * train_correct / total_train

        # --- PHA VALIDATION ---
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        total_val   = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

                val_loss    += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val   += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / total_val
        val_acc      = 100 * val_correct / total_val
        current_lr   = optimizer.param_groups[0]['lr']

        # --- CHECKPOINTING ---
        saved_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
            saved_msg = "⭐ Saved"
        else:
            epochs_no_improve += 1

        # In kết quả
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f} {saved_msg}")

        # Cập nhật LR scheduler
        scheduler.step(avg_val_loss)

        # --- EARLY STOPPING ---
        if epochs_no_improve >= PATIENCE:
            print(f"\n⚠️  Early stopping! Val loss không cải thiện sau {PATIENCE} epoch liên tiếp.")
            print(f"   Best val loss: {best_val_loss:.4f} (đã lưu tại epoch {epoch+1-PATIENCE})")
            break

    print(f"\n🎉 Hoàn tất! Trọng số tốt nhất: '{WEIGHTS_PATH}'")
    print(f"   Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model()