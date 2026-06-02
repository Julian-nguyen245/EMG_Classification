import os
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from losses import GestureLoss
from dataset import sEMGDataset
from model import GestureLSTM
from ninapro_loader import load_and_split_ninapro_csv


def normalize_emg(X_train, X_val, X_test):
    """Z-score normalization theo channel, tính thống kê từ train set để tránh data leakage."""
    # X shape: (N, window_size, channels)
    mean = X_train.mean(axis=(0, 1))  # (channels,)
    std  = X_train.std(axis=(0, 1)) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std
    return X_train, X_val, X_test, mean, std


def train_model(gamma=2.0, weights_suffix=""):
    """
    Huấn luyện model GestureLSTM với Focal Loss.

    Tham số:
    - gamma         : Focusing parameter cho Focal Loss (0.0 = CE baseline)
    - weights_suffix: Hậu tố cho tên file weights (vd: "_gamma2.0")

    Returns:
    - history: dict chứa {'train_loss', 'val_loss', 'train_acc', 'val_acc'} theo epoch
    """
    # 1. CẤU HÌNH THÔNG SỐ
    EPOCHS        = 60
    BATCH_SIZE    = 64
    LEARNING_RATE = 0.001
    WINDOW_SIZE   = 40
    STEP_SIZE     = 20
    NUM_CLASSES   = 13
    HIDDEN_SIZE   = 128
    PATIENCE      = 12   # Tăng từ 10 lên 12 để không dừng sớm trong giai đoạn warmup
    WARMUP_EPOCHS = 5    # Epochs để LR tăng tuyến tính lên max

    WEIGHTS_DIR  = '/home/ju1ian/Documents/EMG Classification/src/weights'
    WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, f'best_lstm{weights_suffix}.pth')
    NORM_PATH    = os.path.join(WEIGHTS_DIR, f'norm_stats{weights_suffix}.npz')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # 2. CHUẨN BỊ DỮ LIỆU
    print("Đang chuẩn bị dữ liệu...")
    file_path = '/home/ju1ian/Documents/EMG Classification/Data (Ninapro)/processed/ninapro_db1_ready.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy {file_path}!")

    (X_train, y_train), (X_val, y_val), (X_test, _) = load_and_split_ninapro_csv(
        file_path, WINDOW_SIZE, STEP_SIZE
    )

    # 3. CHUẨN HÓA TÍN HIỆU EMG (Z-score per channel)
    X_train, X_val, X_test, norm_mean, norm_std = normalize_emg(X_train, X_val, X_test)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    np.savez(NORM_PATH, mean=norm_mean, std=norm_std)
    print(f"Đã lưu normalization stats: {NORM_PATH}")

    train_loader = DataLoader(sEMGDataset(X_train, y_train), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(sEMGDataset(X_val,   y_val),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {len(X_train)} windows | Val: {len(X_val)} windows | Test: {len(X_test)} windows")

    # 4. TÍNH CLASS WEIGHTS
    unique_classes = np.unique(y_train)
    raw_weights    = compute_class_weight('balanced', classes=unique_classes, y=y_train)

    class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls, w in zip(unique_classes, raw_weights):
        class_weights[cls] = w
    class_weights = np.clip(class_weights, 0.2, 10.0)

    print("\nClass Weights:")
    for i, w in enumerate(class_weights):
        label = "REST" if i == 0 else f"Cử chỉ {i}"
        print(f"  Class {i:2d} ({label:10s}): {w:.3f}")

    # 5. KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER
    model = GestureLSTM(
        input_size=10,
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        num_classes=NUM_CLASSES,
        use_attention=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: GestureLSTM + Attention | Hidden: {HIDDEN_SIZE} | Params: {total_params:,}")

    criterion = GestureLoss(
        use_focal_loss=True,
        class_weights=class_weights.tolist(),
        gamma=gamma,
        label_smoothing=0.1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Warmup tuyến tính (epoch 0→WARMUP_EPOCHS) + Cosine decay (WARMUP_EPOCHS→EPOCHS)
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        # Cosine decay xuống 1% của LR ban đầu
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. TRAINING LOOP
    print(f"\nBắt đầu huấn luyện ({EPOCHS} epochs, gamma={gamma}, "
          f"warmup={WARMUP_EPOCHS} epochs, patience={PATIENCE})...\n")

    best_val_loss     = float('inf')
    epochs_no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * inputs.size(0)
            _, predicted   = torch.max(outputs.data, 1)
            total_train   += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / total_train
        train_acc      = 100 * train_correct / total_train

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0

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

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Cập nhật LR theo warmup+cosine schedule
        scheduler.step()

        # --- CHECKPOINTING ---
        saved_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), WEIGHTS_PATH)
            saved_msg = "* Saved"
        else:
            epochs_no_improve += 1

        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.1f}% | "
              f"LR: {current_lr:.6f} {saved_msg}")

        # --- EARLY STOPPING ---
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping! Val loss không cải thiện sau {PATIENCE} epoch.")
            print(f"Best val loss: {best_val_loss:.4f}")
            break

    print(f"\nHoàn tất! Weights: '{WEIGHTS_PATH}' | Best Val Loss: {best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    train_model()