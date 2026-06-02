# EMG Hand Gesture Classification

Phân loại cử chỉ tay (REST / Fist / Flex / Pinch) từ tín hiệu điện cơ bề mặt (sEMG)  
sử dụng **GestureLSTM + Temporal Attention** với **Focal Loss**.

---

## Cấu trúc project

```
EMG Classification/
├── src /                        # Source code chính (chú ý: tên thư mục có dấu cách)
│   ├── model.py                 # GestureLSTM + TemporalAttention
│   ├── losses.py                # GestureLoss (Focal Loss + class weights)
│   ├── dataset.py               # sEMGDataset (PyTorch Dataset)
│   ├── data_utils.py            # create_sliding_windows()
│   ├── session_loader.py        # Load & split session_*.csv
│   ├── train.py                 # Train trên NinaPro DB1 (reference)
│   ├── train_on_sessions.py     # Train trên dữ liệu session tự thu thập
│   ├── evaluate.py              # Eval: ninapro hoặc sessions (2 mode)
│   ├── evaluate_sessions.py     # Gamma experiment trên session data
│   ├── ninapro_loader.py        # Load NinaPro DB1 CSV
│   └── gamma_experiment.py      # Gamma experiment trên NinaPro (reference)
│
├── src/weights/                 # Model weights & normalization stats (gitignored)
│   ├── best_lstm_sessions_gamma{g}.pth
│   └── norm_stats_sessions_gamma{g}.npz
│
├── outputs/                     # Figures & JSON results (gitignored)
│   └── output_sessions/         # Kết quả thực nghiệm gamma trên sessions
│       ├── gamma_session_results.json
│       ├── loss_curves.png
│       ├── metrics_comparison.png
│       ├── per_class_f1.png
│       └── confusion_matrix_gamma{g}.png
│
├── dummy_sessions/              # Dữ liệu session (gitignored)
│   ├── session_1.csv
│   ├── session_2.csv
│   └── session_3.csv
│
├── main.cpp                     # Firmware Arduino (ESP32-S3)
├── test.py                      # GUI thu thập dữ liệu real-time (tkinter)
├── generate_dummy.py            # Sinh dữ liệu session mô phỏng
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
# 1. Clone repo
git clone <repo-url>
cd "EMG Classification"

# 2. Tạo môi trường conda (khuyến nghị)
conda create -n emg python=3.12
conda activate emg

# 3. Cài dependencies
pip install -r requirements.txt
```

---

## Quy trình đầy đủ

### Bước 1 — Chuẩn bị dữ liệu session

**Option A: Dùng dữ liệu mô phỏng (không cần phần cứng)**
```bash
python generate_dummy.py
# → dummy_sessions/session_1.csv, session_2.csv, session_3.csv
```

**Option B: Thu thập dữ liệu thực tế (cần ESP32-S3 + AD8232)**
1. Flash firmware lên ESP32-S3:
   ```bash
   # Dùng Arduino IDE hoặc PlatformIO, nạp main.cpp
   ```
2. Chạy GUI thu thập:
   ```bash
   python test.py
   # hoặc chỉ định port:
   python test.py --port /dev/ttyACM0
   ```
3. Trong GUI: nhấn **Kết nối ESP32** → chọn gesture → **Bắt đầu ghi** → **Lưu Session**

Format CSV output: `timestamp_ms, raw_emg, filtered_emg, label, repetition`

---

### Bước 2 — Thực nghiệm Gamma (train + eval tự động)

```bash
cd "src "

# Chạy toàn bộ thực nghiệm với 6 giá trị gamma mặc định [0, 0.5, 1, 2, 3, 5]
python evaluate_sessions.py

# Chỉ thử một số gamma
python evaluate_sessions.py --gammas 0.0 2.0 3.0

# Dùng thư mục dữ liệu tùy chỉnh
python evaluate_sessions.py --data /path/to/sessions --gammas 2.0
```

Kết quả lưu vào `outputs/output_sessions/`:
- `gamma_session_results.json` — metrics JSON của tất cả gamma
- `loss_curves.png` — so sánh loss convergence
- `metrics_comparison.png` — bar chart accuracy/F1 theo gamma
- `per_class_f1.png` — heatmap F1 per class × gamma
- `confusion_matrix_gamma{g}.png` — confusion matrix mỗi gamma

---

### Bước 3 — Train một gamma cụ thể

```bash
cd "src "

# Train với gamma=2.0 (mặc định)
python train_on_sessions.py

# Train với gamma khác
python train_on_sessions.py --gamma 0.0
python train_on_sessions.py --gamma 3.0 --data ../my_sessions
```

Weights lưu tại: `src/weights/best_lstm_sessions_gamma{g}.pth`

---

### Bước 4 — Evaluate model đã train

```bash
cd "src "

# Eval trên session data (dùng best_lstm_sessions_gamma2.0.pth)
python evaluate.py --mode sessions --suffix _gamma2.0

# Eval trên NinaPro (reference benchmark, cần train.py chạy trước)
python evaluate.py --mode ninapro --suffix _gamma0.0

# Chỉ định thư mục dữ liệu session
python evaluate.py --mode sessions --data ../my_sessions
```

---

## Kiến trúc mô hình

```
Input: (B, 40, 1)          — batch B windows, 40 timestep, 1 kênh EMG
    ↓
LSTM Layer 1  (hidden=128)
    ↓
LSTM Layer 2  (hidden=128, dropout=0.3)
    ↓  h_1...h_40 ∈ R^128
Temporal Attention
    e_t = w^T h_t           — score mỗi timestep
    α_t = softmax(e_t)      — trọng số attention
    c   = Σ α_t h_t         — context vector
    ↓
LayerNorm(128) + Dropout(0.3)
    ↓
Linear(128 → 4)
    ↓
Output: (B, 4)              — logits 4 lớp
```

**Tổng params:** ~200,068

---

## Cấu hình mặc định

| Tham số | Giá trị |
|---------|---------|
| Sampling rate | 100 Hz |
| Window size | 40 mẫu = 400 ms |
| Step size | 20 mẫu = 50% overlap |
| Train/Val/Test split | rep 1–14 / 15–17 / 18–20 |
| Hidden size | 128 |
| Batch size | 64 |
| Epochs | 60 (+ early stopping patience=12) |
| Learning rate | 0.001 với warmup 5ep + cosine decay |
| Loss | Focal Loss γ=2.0, class weights, label smoothing=0.1 |

---

## Kết quả

| Gamma | Accuracy | F1 Macro | Ghi chú |
|-------|----------|----------|---------|
| 0.0   | 66.05%   | 72.76%   | NinaPro reference (pre-fix) |
| 2.0   | 66.84%   | 72.06%   | NinaPro reference (pre-fix) |
| **2.0** | **77.30%** | **72.44%** | **Sessions (pipeline đã sửa)** |

---

## Phần cứng

| Thành phần | Mô tả |
|------------|-------|
| **ESP32-S3** | Vi điều khiển 32-bit, ADC 12-bit, USB-Serial |
| **AD8232** | IC khuếch đại EMG, instrumentation amp + bandpass filter |
| Điện cực | 3 miếng dán bề mặt: IN+, IN−, DRL (ground) |
| Sampling | 1000 Hz firmware → 100 Hz sau downsample |

---

## Yêu cầu

```
torch==2.5.1
numpy==1.26.4
scipy==1.13.1
matplotlib==3.9.2
scikit-learn==1.5.2
pandas
seaborn
pyserial       # cho test.py (thu thập dữ liệu)
```
