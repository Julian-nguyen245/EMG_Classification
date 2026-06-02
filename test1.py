import serial
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
import csv
import glob

# ==========================================
# CẤU HÌNH THÔNG SỐ 
# ==========================================
# Auto-detect port trên Linux
ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
SERIAL_PORT = ports[0] if ports else '/dev/ttyACM0'
print(f"Su dung port: {SERIAL_PORT}")

BAUD_RATE = 115200
FS = 1000
MAX_SAMPLES = 10000  

CSV_FILE_NAME = "emg_last_2min_snapshot.csv"

# Khởi tạo bộ đệm cuốn chiếu cho dữ liệu gốc và dữ liệu sau lọc
raw_buffer = collections.deque([2048] * MAX_SAMPLES, maxlen=MAX_SAMPLES)
filtered_buffer = collections.deque([0.0] * MAX_SAMPLES, maxlen=MAX_SAMPLES)

# ==========================================
# THIẾT LẬP BỘ LỌC KÉP TỐI ƯU VÀ TRẠNG THÁI ZI
# ==========================================
b_notch, a_notch = iirnotch(50.0, 30.0, FS)
nyq = 0.5 * FS
low_cutoff = 20.0 / nyq
high_cutoff = 450.0 / nyq
b_band, a_band = butter(4, [low_cutoff, high_cutoff], btype='band')

# Khởi tạo giá trị trạng thái ban đầu cho bộ lọc 
zi_notch = lfilter_zi(b_notch, a_notch) * 2048.0
zi_band = lfilter_zi(b_band, a_band) * 0.0

# KẾT NỐI SERIAL
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    ser.reset_input_buffer()
    print(f"Đã kết nối thành công tới {SERIAL_PORT}")
except Exception as e:
    print(f"Lỗi kết nối cổng Serial: {e}")
    exit()

# CẤU HÌNH ĐỒ THỊ
fig, ax = plt.subplots(figsize=(14, 7))

# Khởi tạo đường màu cam cho tín hiệu THÔ
line_raw, = ax.plot(range(MAX_SAMPLES), [0] * MAX_SAMPLES,
                    color='darkorange', linewidth=0.5, alpha=0.7, label='Tín hiệu THÔ (nền 0)')

# Khởi tạo đường màu xanh cho tín hiệu ĐÃ LỌC
line_filtered, = ax.plot(range(MAX_SAMPLES), [0] * MAX_SAMPLES,
                         color='limegreen', linewidth=1.0, label='Tín hiệu ĐÃ LỌC SẠCH')

# MỞ RỘNG TRỤC Y (Vì tín hiệu thô biên độ rất lớn, để -1500 sẽ bị kịch trần)
ax.set_ylim(-2200, 2200)
ax.grid(True)
ax.legend(loc='upper right') # Thêm bảng chú thích góc trên bên phải

# HÀM CẬP NHẬT ĐỒ THỊ (Tối ưu hóa xả lũ + Khử trễ pha thời gian thực)
def update(frame):
    global zi_notch, zi_band # Gọi biến trạng thái bộ lọc toàn cục
    
    # Đọc TOÀN BỘ số byte đang bị nghẽn trong hàng đợi Serial cùng một lúc
    bytes_available = ser.in_waiting
    if bytes_available >= 2:
        # Đảm bảo số byte đọc vào là số chẵn (mỗi mẫu = 2 bytes)
        bytes_to_read = bytes_available - (bytes_available % 2)
        raw_bytes = ser.read(bytes_to_read)
        
        # Chuyển đổi hàng loạt bytes sang số nguyên bằng Numpy
        raw_samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        high_bytes = raw_samples[0::2].astype(np.uint16)
        low_bytes = raw_samples[1::2].astype(np.uint16)
        vals = (high_bytes << 8) | low_bytes
        
        # Lọc các giá trị hợp lệ (0 - 4095)
        valid_vals = vals[(vals >= 0) & (vals <= 4095)].astype(float)
        
        if len(valid_vals) > 0:
            # Đẩy dữ liệu thô vào bộ đệm gốc
            for val in valid_vals:
                raw_buffer.append(val)
            
            # Chỉ lọc DUY NHẤT gói dữ liệu mới về và cập nhật liên tục trạng thái bộ nhớ (zi)
            data_notch, zi_notch = lfilter(b_notch, a_notch, valid_vals, zi=zi_notch)
            data_filtered, zi_band = lfilter(b_band, a_band, data_notch, zi=zi_band)
            
            # Đẩy dữ liệu sau lọc vào bộ đệm hiển thị
            for f_val in data_filtered:
                filtered_buffer.append(f_val)
            
            # Trừ đường nền 2048 cho mảng thô để đưa về gốc không (0) trước khi vẽ
            raw_shifted = np.array(raw_buffer, dtype=float) - 2048.0
            
            # Cập nhật đồ thị song song theo thời gian thực (Không bị lệch pha, không kéo dài đuôi)
            line_raw.set_ydata(raw_shifted)
            line_filtered.set_ydata(np.array(filtered_buffer))
            
    return line_raw, line_filtered

# Chạy Animation
ani = animation.FuncAnimation(fig, update, blit=True, interval=10, save_count=100)

try:
    plt.show() # Chương trình sẽ đứng ở đây phục vụ bạn quan sát đồ thị
except KeyboardInterrupt:
    pass
finally:
    # ĐÓNG CỔNG SERIAL AN TOÀN
    ser.close()
    
    print(f"\n[INFO] Đang xử lý bộ lọc cho {len(raw_buffer)} mẫu cuối cùng...")
    # Tính toán lại một lần cuối bằng mảng tĩnh trên bộ đệm thô để đảm bảo file CSV chuẩn nhất
    raw_array = np.array(raw_buffer, dtype=float)
    data_notch_final = lfilter(b_notch, a_notch, raw_array)
    final_filtered = lfilter(b_band, a_band, data_notch_final)
    
    print(f"[INFO] Đang xuất dữ liệu ra file '{CSV_FILE_NAME}'...")
    try:
        with open(CSV_FILE_NAME, mode='w', newline='', encoding='utf-8') as f_csv:
            csv_writer = csv.writer(f_csv)
            # CHỈ GIỮ LẠI 2 CỘT NHƯ BẠN MUỐN
            csv_writer.writerow(['Raw_ADC', 'Filtered_EMG'])           
            # GHI ĐÚNG GIÁ TRỊ THÔ GỐC VÀ GIÁ TRỊ LỌC SẠCH
            for r, f in zip(raw_buffer, final_filtered):
                csv_writer.writerow([r, round(f, 2)])
        print(f" THÀNH CÔNG! Đã lưu đúng {len(raw_buffer)} dòng vào file '{CSV_FILE_NAME}'.")
    except Exception as e:
        print(f"Lỗi khi ghi file CSV: {e}")
