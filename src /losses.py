import torch
import torch.nn as nn
import torch.nn.functional as F

class GestureLoss(nn.Module):
    def __init__(self, use_focal_loss=False, class_weights=None):
        """
        Quản lý các hàm Loss cho dự án sEMG.
        
        Tham số:
        - use_focal_loss: Nếu True, sẽ dùng Focal Loss (tốt cho data mất cân bằng).
                          Nếu False, dùng CrossEntropyLoss mặc định.
        - class_weights: List trọng số cho từng class. VD: [0.1, 1.0, 1.0, 1.0, 1.0] 
                         (Phạt nhẹ class 0, phạt nặng class 1, 2, 3, 4)
        """
        super(GestureLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        
        # Chuyển list weights thành Tensor nếu có
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            
        # Hàm mặc định
        self.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(self, logits, targets):
        """
        Tính toán sai số.
        - logits: Đầu ra của mô hình (Batch_size, Num_classes)
        - targets: Nhãn thực tế (Batch_size)
        """
        if not self.use_focal_loss:
            return self.ce_loss(logits, targets)
        else:
            # Thuật toán Focal Loss cơ bản (Tự động tập trung vào các mẫu khó dự đoán)
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss) # Xác suất dự đoán đúng
            focal_loss = ((1 - pt) ** 2) * ce_loss
            return focal_loss.mean()

# -------------------------------------------------------------------
# Unit Test
if __name__ == "__main__":
    # Giả lập đầu ra của model (32 mẫu, 5 class)
    dummy_logits = torch.randn(32, 5)
    dummy_targets = torch.randint(0, 5, (32,))
    
    # Test 1: Cross Entropy bình thường
    criterion_normal = GestureLoss(use_focal_loss=False)
    loss_normal = criterion_normal(dummy_logits, dummy_targets)
    print(f"Cross Entropy Loss: {loss_normal.item():.4f}")
    
    # Test 2: Dùng Focal Loss (Nâng cao)
    criterion_focal = GestureLoss(use_focal_loss=True)
    loss_focal = criterion_focal(dummy_logits, dummy_targets)
    print(f"Focal Loss        : {loss_focal.item():.4f}")