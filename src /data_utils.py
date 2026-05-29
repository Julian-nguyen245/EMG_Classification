import numpy as np 

def create_sliding_windows(emg_signals, labels, window_size=40, step_size=20):
    """
    Cắt tín hiệu sEMG liên tục thành các cửa sổ trượt (Sliding Windows).
    
    Tham số:
    - emg_signals : numpy array (N_samples, 10_channels)
    - labels      : numpy array (N_samples,)
    - window_size : Kích thước cửa sổ. Với Ninapro 100Hz: 40 mẫu = 400ms.
    - step_size   : Bước trượt (Stride). Nếu = 20 (200ms) tức là overlap 50%.
    
    Trả về:
    - X: array (Số_lượng_window, window_size, 10_channels)
    - y: array (Số_lượng_window,) - Nhãn của mỗi window
    """
    X = []
    y = []

    n_sample = len(emg_signals)


    # Scan through the signal, each step=step_size
    for i in range(0, n_sample - window_size + 1, step_size):

        window = emg_signals[i : i + window_size, :]

        label = labels[i + window_size - 1]

        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)


    # -------------------------------------------------------------------
# Test thử hàm (Chỉ chạy khi file này được gọi trực tiếp)
if __name__ == "__main__":
    # Giả lập data: 1000 mẫu (10 giây), 10 điện cực
    dummy_emg = np.random.rand(1000, 10)
    dummy_labels = np.zeros(1000)
    dummy_labels[400:600] = 2 # Có cử chỉ từ giây thứ 4 đến 6
    
    X_windows, y_windows = create_sliding_windows(
        dummy_emg, 
        dummy_labels, 
        window_size=40,  # 400ms
        step_size=20     # 200ms (Overlap 50%)
    )
    
    print(f"Dữ liệu gốc     : {dummy_emg.shape}")
    print(f"Data sau cắt (X): {X_windows.shape} -> (Số window, Time-steps, Channels)")
    print(f"Nhãn sau cắt (y): {y_windows.shape}")