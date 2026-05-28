import torch
import torch.nn as nn

class GestureLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, num_classes=5, dropout=0.2):
        """
        Mô hình LSTM phân loại cử chỉ tay qua sEMG.
        
        Tham số:
        - input_size : Số lượng điện cực (channels). Ninapro DB1 là 10.
        - hidden_size: Số lượng nơ-ron ẩn trong mỗi lớp LSTM (Sức mạnh biểu diễn).
        - num_layers : Số lớp LSTM chồng lên nhau (Deep LSTM).
        - num_classes: Số lượng cử chỉ cần dự đoán.
        - dropout    : Tỷ lệ ngắt kết nối nơ-ron ngẫu nhiên để chống Overfitting.
        """
        super(GestureLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Lớp LSTM cốt lõi
        # batch_first=True bắt buộc input phải có form: (Batch, Sequence, Feature)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Lớp Fully Connected (Linear) để ánh xạ từ hidden_size ra số lượng classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Luồng đi của dữ liệu.
        x có shape: (batch_size, seq_length, input_size) -> VD: (32, 40, 10)
        """
        # out lưu trữ đầu ra của tất cả các time-steps: (batch_size, seq_length, hidden_size)
        # h_n, c_n là trạng thái ẩn và cell state cuối cùng
        out, (h_n, c_n) = self.lstm(x)
        
        # CHÚ Ý: Chúng ta chỉ dùng thông tin ở bước thời gian cuối cùng (last time-step)
        # out[:, -1, :] nghĩa là lấy tất cả batch, tại sequence index cuối cùng, lấy tất cả features
        last_out = out[:, -1, :] 
        
        # Đưa qua lớp Linear để ra điểm số (logits) của từng class
        logits = self.fc(last_out)
        
        return logits

# -------------------------------------------------------------------
# Unit Test: Chạy thử để đảm bảo kích thước ma trận hoàn toàn khớp
if __name__ == "__main__":
    # 1. Cấu hình giả lập
    batch_size = 32
    seq_length = 40  # Tương đương window_size (400ms)
    input_size = 10  # 10 channels của Ninapro
    num_classes = 5  # Giả sử chúng ta đang nhận diện 5 cử chỉ
    
    # 2. Tạo mạng
    model = GestureLSTM(input_size=input_size, hidden_size=64, num_layers=2, num_classes=num_classes)
    
    # 3. Tạo một batch dữ liệu giả lập (Random Tensor)
    dummy_input = torch.randn(batch_size, seq_length, input_size)
    
    # 4. Chạy forward pass
    output = model(dummy_input)
    
    print(f"Tổng số tham số mạng: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    print(f"Kích thước Input    : {dummy_input.shape}")
    print(f"Kích thước Output   : {output.shape} -> Cần phải là (Batch_size, Num_classes) tức là (32, 5)")
    
    if output.shape == (batch_size, num_classes):
        print("✅ Unit test PASSED: Kiến trúc mạng hoạt động hoàn hảo!")
    else:
        print("❌ Unit test FAILED: Cần kiểm tra lại các chiều ma trận.")