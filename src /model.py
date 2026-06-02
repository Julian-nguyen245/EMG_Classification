import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    Soft attention trên trục thời gian của LSTM output.
    Thay vì chỉ dùng timestep cuối, học cách weight toàn bộ sequence.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq, hidden)
        scores = self.attn(lstm_out)               # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)     # (batch, seq, 1)
        context = (lstm_out * weights).sum(dim=1)  # (batch, hidden)
        return context


class GestureLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, num_classes=13,
                 dropout=0.3, use_attention=True):
        """
        Mô hình LSTM phân loại cử chỉ tay qua sEMG.

        Tham số:
        - input_size   : Số lượng điện cực. Ninapro DB1 là 10.
        - hidden_size  : Số nơ-ron ẩn mỗi lớp LSTM.
        - num_layers   : Số lớp LSTM chồng.
        - num_classes  : Số lượng cử chỉ.
        - dropout      : Tỷ lệ dropout để chống overfitting.
        - use_attention: Nếu True, dùng temporal attention thay vì chỉ lấy timestep cuối.
        """
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        if use_attention:
            self.attention = TemporalAttention(hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq, input_size)
        out, _ = self.lstm(x)  # out: (batch, seq, hidden)

        if self.use_attention:
            pooled = self.attention(out)   # weighted sum qua toàn sequence
        else:
            pooled = out[:, -1, :]         # chỉ lấy timestep cuối

        pooled = self.layer_norm(pooled)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


# -------------------------------------------------------------------
# Unit Test
if __name__ == "__main__":
    batch_size, seq_length, input_size, num_classes = 32, 40, 10, 13

    model_attn = GestureLSTM(input_size=input_size, hidden_size=128,
                             num_layers=2, num_classes=num_classes, use_attention=True)
    model_last = GestureLSTM(input_size=input_size, hidden_size=128,
                             num_layers=2, num_classes=num_classes, use_attention=False)

    dummy_input = torch.randn(batch_size, seq_length, input_size)

    out_attn = model_attn(dummy_input)
    out_last = model_last(dummy_input)

    print(f"Params (với attention) : {sum(p.numel() for p in model_attn.parameters()):,}")
    print(f"Params (không attention): {sum(p.numel() for p in model_last.parameters()):,}")
    print(f"Output shape: {out_attn.shape} -> kỳ vọng ({batch_size}, {num_classes})")

    assert out_attn.shape == (batch_size, num_classes)
    assert out_last.shape == (batch_size, num_classes)
    print("Unit test PASSED")