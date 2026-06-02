import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureLoss(nn.Module):
    def __init__(self, use_focal_loss=False, class_weights=None, gamma=2.0, label_smoothing=0.1):
        """
        Quản lý các hàm Loss cho dự án sEMG.

        Tham số:
        - use_focal_loss  : Nếu True, dùng Focal Loss. Nếu False, dùng CrossEntropyLoss.
        - class_weights   : List trọng số cho từng class để cân bằng dữ liệu.
        - gamma           : Focusing parameter của Focal Loss.
                            gamma=0 → tương đương CrossEntropy.
                            gamma cao → tập trung mạnh vào mẫu khó dự đoán.
        - label_smoothing : Làm mềm nhãn, giảm overconfidence. 0.1 là giá trị khuyến nghị.
        """
        super(GestureLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma

        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # Lưu weight để dùng chung cho cả CE và Focal Loss path
        self.register_buffer('weight', weight_tensor)

        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        """
        logits : (Batch_size, Num_classes)
        targets: (Batch_size,)
        """
        if not self.use_focal_loss:
            return self.ce_loss(logits, targets)

        # Focal Loss: áp dụng class_weights và label_smoothing từ self.ce_loss
        # BUG FIX: truyền weight vào F.cross_entropy để class imbalance được xử lý đúng
        weight = self.weight.to(logits.device) if self.weight is not None else None
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none',
                                  label_smoothing=self.ce_loss.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# -------------------------------------------------------------------
# Unit Test
if __name__ == "__main__":
    dummy_logits = torch.randn(32, 5)
    dummy_targets = torch.randint(0, 5, (32,))

    criterion_ce = GestureLoss(use_focal_loss=False)
    print(f"Cross Entropy Loss : {criterion_ce(dummy_logits, dummy_targets).item():.4f}")

    criterion_focal = GestureLoss(use_focal_loss=True, gamma=2.0)
    print(f"Focal Loss (γ=2.0) : {criterion_focal(dummy_logits, dummy_targets).item():.4f}")

    # Kiểm tra class weights được áp dụng đúng trong cả hai path
    weights = [0.5, 1.0, 1.5, 1.0, 2.0]
    criterion_weighted_focal = GestureLoss(use_focal_loss=True, class_weights=weights, gamma=2.0)
    print(f"Weighted Focal Loss: {criterion_weighted_focal(dummy_logits, dummy_targets).item():.4f}")