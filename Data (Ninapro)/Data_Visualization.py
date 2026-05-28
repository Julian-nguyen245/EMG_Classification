import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc file CSV
df = pd.read_csv('ninapro_db1_ready.csv')

# Tìm khoảng dữ liệu có chứa hành động (restimulus > 0)
# Nghỉ ngơi (0) chiếm đa số, nên ta tìm lần co cơ đầu tiên
action_indices = df.index[df['restimulus'] > 0].tolist()

if action_indices:
    # Lấy mẫu bắt đầu từ trước khi có hành động 1000 samples, 
    # và kéo dài khoảng 5000 samples để thấy rõ cả nghỉ và làm
    start_idx = max(0, action_indices[0] - 1000)
    end_idx = min(len(df), start_idx + 5000)
else:
    # Nếu không tìm thấy, lấy 5000 mẫu đầu tiên
    start_idx = 0
    end_idx = min(len(df), 5000)

df_subset = df.iloc[start_idx:end_idx].copy()
time_axis = np.arange(start_idx, end_idx)

# Bắt đầu vẽ
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# 1. Vẽ 10 kênh EMG trên trục trên (ax1)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(10):
    # Cột EMG từ 0 đến 9
    col_name = f'emg_ch{i+1}'
    # Cộng thêm offset để tách biệt các đường
    ax1.plot(time_axis, df_subset[col_name] + i*0.15, label=f'Ch {i+1}', color=colors[i], linewidth=1)

ax1.set_title('Tín hiệu 10 kênh EMG (Đã thêm offset)', fontsize=14)
ax1.set_ylabel('Biên độ EMG', fontsize=12)
ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
ax1.grid(True, linestyle='--', alpha=0.5)

# 2. Vẽ Nhãn (restimulus) và Lần lặp (repetition) trên trục dưới (ax2)
ax2.plot(time_axis, df_subset['restimulus'], color='red', label='Restimulus (Nhãn)', linewidth=2)
ax2.plot(time_axis, df_subset['repetition'], color='blue', linestyle='--', label='Repetition (Lần lặp)', linewidth=2)

ax2.set_title('Nhãn cử chỉ và Lần lặp tương ứng', fontsize=14)
ax2.set_xlabel('Thời gian (Samples)', fontsize=12)
ax2.set_ylabel('ID / Số lần', fontsize=12)
ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('emg_with_labels.png', dpi=300, bbox_inches='tight')
print("Saved plot to emg_with_labels.png")
print(f"Plotted from sample {start_idx} to {end_idx}")