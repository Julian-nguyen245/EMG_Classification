import torch 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 

class sEMGDataset(Dataset):
    def __init__(self, X, y):
        """
        Khởi tạo Dataset cho sEMG.
        
        Tham số:
        - X: numpy array shape (N_windows, window_size, num_channels)
        - y: numpy array shape (N_windows,) chứa nhãn (0, 1, 2...)
        """
        # Chuyển đổi NumPy array sang PyTorch Tensors
        # LSTM yêu cầu input là float32
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        """Trả về tổng số lượng windows (cửa sổ) có trong dataset"""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Trả về 1 cặp (tín hiệu, nhãn) tại vị trí idx.
        DataLoader sẽ dùng hàm này để gom thành các batch.
        """
        return self.X[idx], self.y[idx]


# -------------------------------------------------------------------
# Unit Test: Chạy thử để xem luồng dữ liệu (Data Pipeline) hoạt động ra sao
if __name__ == "__main__":
    from data_utils import create_sliding_windows
    
    # 1. Giả lập dữ liệu sEMG (giống Bước 2)
    dummy_emg = np.random.rand(1000, 10)  # 1000 time-steps, 10 channels
    dummy_labels = np.random.randint(0, 5, size=(1000,)) # 5 class (0 đến 4)
    
    # 2. Cắt Sliding Window
    X_windows, y_windows = create_sliding_windows(dummy_emg, dummy_labels, window_size=40, step_size=20)
    
    # 3. Đưa vào PyTorch Dataset
    dataset = sEMGDataset(X_windows, y_windows)
    
    # 4. Đưa vào DataLoader để tự động trộn (shuffle) và chia batch
    # Trong thực tế, batch_size có thể là 32, 64, hoặc 128
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Tổng số mẫu trong Dataset: {len(dataset)}")
    
    # Lấy thử 1 batch ra để kiểm tra kích thước
    for batch_X, batch_y in dataloader:
        print(f"Kích thước 1 Batch X : {batch_X.shape} -> (Batch_size, Sequence_Length, Input_Size)")
        print(f"Kích thước 1 Batch y : {batch_y.shape} -> (Batch_size)")
        break # Chỉ in batch đầu tiên