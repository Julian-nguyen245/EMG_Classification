import pandas as pd
import numpy as np
from data_utils import create_sliding_windows

def load_and_split_ninapro_csv(file_path, window_size=40, step_size=20):
    """
    Đọc file Ninapro CSV đã qua xử lý và chia Train/Val/Test.
    """
    print(f"📦 Đang đọc dữ liệu từ CSV: {file_path}")
    df = pd.read_csv(file_path)
    
    # 1. Trích xuất các ma trận numpy từ DataFrame
    # Lấy 10 cột EMG
    emg_cols = [f'emg_ch{i}' for i in range(1, 11)]
    emg = df[emg_cols].values
    
    # Lấy nhãn restimulus (Nhãn đã fix độ trễ - Rất tốt!)
    labels = df['restimulus'].values
    
    # Lấy cột số lần lặp
    repetition = df['repetition'].values
    
    # 2. Thuật toán phân tách Train/Val/Test
    train_idx, val_idx, test_idx = [], [], []
    current_rep = 0
    
    for i in range(len(repetition)):
        # Nếu đang ở trong một lần lặp cụ thể
        if repetition[i] > 0:
            current_rep = repetition[i]
            
        # Chia theo luật đã định
        if current_rep in [1, 2, 3, 4, 5, 6, 7]:
            train_idx.append(i)
        elif current_rep == 8:
            val_idx.append(i)
        elif current_rep in [9, 10]:
            test_idx.append(i)
        else:
            # Đoạn rest đầu tiên khi chưa có cử chỉ nào
            train_idx.append(i)
            
    # 3. Cắt dữ liệu thô
    X_raw_train, y_raw_train = emg[train_idx], labels[train_idx]
    X_raw_val, y_raw_val     = emg[val_idx], labels[val_idx]
    X_raw_test, y_raw_test   = emg[test_idx], labels[test_idx]
    
    print(f"✂️ Đang áp dụng Sliding Window (Cửa sổ: {window_size}, Bước trượt: {step_size})...")
    X_train, y_train = create_sliding_windows(X_raw_train, y_raw_train, window_size, step_size)
    X_val, y_val     = create_sliding_windows(X_raw_val, y_raw_val, window_size, step_size)
    X_test, y_test   = create_sliding_windows(X_raw_test, y_raw_test, window_size, step_size)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    # Test thử script (đảm bảo file csv nằm cùng thư mục hoặc trỏ đúng đường dẫn)
    file_path = '../data/processed/ninapro_db1_ready.csv'
    # load_and_split_ninapro_csv(file_path)