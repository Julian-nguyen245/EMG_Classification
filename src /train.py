import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from losses import GestureLoss

# Import các class chúng ta vừa viết
from dataset import sEMGDataset
from model import GestureLSTM
from data_utils import create_sliding_windows

def train_model():
    # 1. CẤU HÌNH THÔNG SỐ (Hyperparameters)
    # ---------------------------------------------------------
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WINDOW_SIZE = 40
    STEP_SIZE = 20
    NUM_CLASSES = 5
    
    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang sử dụng thiết bị: {device}")

    # 2. CHUẨN BỊ DỮ LIỆU (Giả lập để test pipeline)
    # ---------------------------------------------------------
    print("⏳ Đang chuẩn bị dữ liệu...")
    # TODO: Thay thế đoạn này bằng code đọc file .mat thực tế của bạn
    # Ví dụ: X_raw, y_raw = load_ninapro_data('data/raw/S1_A1_E1.mat')
    X_raw_train = np.random.rand(8000, 10)  # 8000 mẫu (~80 giây)
    y_raw_train = np.random.randint(0, NUM_CLASSES, size=(8000,))
    
    X_raw_val = np.random.rand(2000, 10)    # 2000 mẫu (~20 giây)
    y_raw_val = np.random.randint(0, NUM_CLASSES, size=(2000,))

    # Cắt cửa sổ trượt
    X_train, y_train = create_sliding_windows(X_raw_train, y_raw_train, WINDOW_SIZE, STEP_SIZE)
    X_val, y_val = create_sliding_windows(X_raw_val, y_raw_val, WINDOW_SIZE, STEP_SIZE)

    # Đưa vào Dataset & DataLoader
    train_loader = DataLoader(sEMGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(sEMGDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    print(f"✅ Dữ liệu Train: {len(X_train)} windows | Dữ liệu Val: {len(X_val)} windows")

    # 3. KHỞI TẠO MÔ HÌNH, LOSS, OPTIMIZER
    # ---------------------------------------------------------
    model = GestureLSTM(input_size=10, hidden_size=64, num_layers=2, num_classes=NUM_CLASSES).to(device)
    
    # CrossEntropyLoss: Hàm mất mát chuẩn cho phân loại đa lớp
    criterion = GestureLoss(use_focal_loss=False)
    
    # Adam Optimizer: Thuật toán tối ưu trọng số hiệu quả nhất hiện nay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Tạo thư mục lưu weights nếu chưa có
    os.makedirs('../weights', exist_ok=True)
    best_val_loss = float('inf')

    # 4. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
    # ---------------------------------------------------------
    print("🔥 Bắt đầu huấn luyện...\n")
    for epoch in range(EPOCHS):
        # --- PHÁT TRAIN ---
        model.train() # Bật chế độ train (Kích hoạt Dropout)
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Xóa gradient cũ
            optimizer.zero_grad()

            # Chạy tiến (Forward pass)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Chạy lùi (Backward pass) & Cập nhật trọng số
            loss.backward()
            optimizer.step()

            # Thống kê
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1) # Lấy class có xác suất cao nhất
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / total_train
        train_acc = 100 * train_correct / total_train

        # --- PHA VALIDATION ---
        model.eval() # Bật chế độ đánh giá (Tắt Dropout)
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        # Không tính gradient ở pha này để tiết kiệm RAM và tăng tốc
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / total_val
        val_acc = 100 * val_correct / total_val

        # 5. CHECKPOINTING (Lưu model tốt nhất)
        # ---------------------------------------------------------
        saved_msg = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Lưu lại trạng thái của mạng
            torch.save(model.state_dict(), '../weights/best_lstm.pth')
            saved_msg = "⭐ (Đã lưu mô hình tốt nhất)"

        # In kết quả mỗi epoch
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}% {saved_msg}")

    print("\n🎉 Hoàn tất huấn luyện! Trọng số tốt nhất nằm ở 'weights/best_lstm.pth'")

if __name__ == "__main__":
    train_model()