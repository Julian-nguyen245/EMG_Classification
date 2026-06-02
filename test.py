"""
sEMG Recorder — ESP32-C3 + AD8232
Thu thập tín hiệu EMG, gán nhãn gesture, lưu CSV 100Hz
Gestures: rest | fist | flex | pinch
"""

import serial, threading, time, csv, os
import sys
import glob
import argparse
import numpy as np
import collections
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import butter, iirnotch, tf2sos, sosfilt, sosfilt_zi

# ── Cấu hình ──────────────────────────────────────────────────────
def _detect_port():
    """Tự tìm port ESP32: /dev/ttyACM* → lấy cái đầu tiên."""
    ports = sorted(glob.glob('/dev/ttyACM*'))
    return ports[0] if ports else '/dev/ttyACM0'

_parser = argparse.ArgumentParser()
_parser.add_argument('--port', default=None)
_args, _ = _parser.parse_known_args()
SERIAL_PORT = _args.port if _args.port else _detect_port()

BAUD_RATE   = 115200
FS          = 1000          # Hz firmware gửi
DOWNSAMPLE  = 10            # 1000 → 100 Hz
FS_OUT      = FS // DOWNSAMPLE
# DELAY FIX 3: Thu nhỏ từ 8000 (8s) xuống 3000 (3s).
# Mỗi pixel trên màn hình bây giờ = ~3ms thay vì ~8ms — tín hiệu rõ, delay cảm nhận giảm.
DISPLAY_N   = 3000
TARGET_REPS = 20
SESSIONS    = 3
GESTURES    = ['fist', 'flex', 'pinch']
SAVE_DIR    = os.path.join(os.path.expanduser('~'), 'Documents', 'EMG Classification')

GESTURE_COLORS = {'rest': '#aaa', 'fist': '#e94560',
                  'flex': '#00ff7f', 'pinch': '#c77dff'}


class EMGRecorder:
    def __init__(self):
        # DELAY FIX 2: Hạ bậc lọc từ 4 xuống 2 → group delay giảm ~50%.
        # Dùng SOS (second-order sections) thay tf2sos cho ổn định số hơn.
        b_n, a_n = iirnotch(50.0, 30.0, FS)
        nyq = 0.5 * FS
        self.sos_n = tf2sos(b_n, a_n)                            # notch 50Hz
        self.sos_b = butter(2, [20.0/nyq, 450.0/nyq],            # bandpass bậc 2
                            btype='band', output='sos')

        # Initial conditions: notch khởi tạo tại DC bias 2048, bandpass tại 0
        self.zi_n = sosfilt_zi(self.sos_n) * 2048.0
        self.zi_b = sosfilt_zi(self.sos_b) * 0.0

        # Buffer hiển thị (3 giây tín hiệu)
        self.raw_disp  = collections.deque([0.0] * DISPLAY_N, maxlen=DISPLAY_N)
        self.filt_disp = collections.deque([0.0] * DISPLAY_N, maxlen=DISPLAY_N)

        # Trạng thái
        self.recording     = False
        self.current_label = 'rest'
        self.gesture_id    = 0
        self.rep_counts    = {g: 0 for g in GESTURES}
        self.csv_rows      = []
        self.ds_accum      = []
        self.sample_ts     = 0
        self.session_num   = 1
        self._rec_start    = 0.0

        self.ser      = None
        self.stop_evt = threading.Event()
        self.lock     = threading.Lock()

        self._build_gui()

    # ── GUI ───────────────────────────────────────────────────────────
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("sEMG Recorder")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry('1080x860')
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Hai subplot tách biệt: Raw (trên) và Filtered (dưới) ──────
        fig = Figure(figsize=(10.5, 5.8), facecolor='#0d1117')
        fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.1, hspace=0.55)

        self.ax_raw  = fig.add_subplot(211, facecolor='#161b22')
        self.ax_filt = fig.add_subplot(212, facecolor='#161b22')

        # X-axis ticks theo giây (−3s → 0s), ticks mỗi 0.5s = 500 mẫu
        step = FS // 2
        xtick_pos = np.arange(0, DISPLAY_N + 1, step)
        xtick_lbl = [f"{(p - DISPLAY_N) / 1000:.1f}s" for p in xtick_pos]

        xs = np.arange(DISPLAY_N)

        def _style_ax(ax, title, ylabel, color):
            ax.set_xlim(0, DISPLAY_N - 1)
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(xtick_lbl, fontsize=8, color='#8b949e')
            ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.set_title(title, color='white', fontsize=10, fontweight='bold',
                         loc='left', pad=4)
            ax.grid(True, color='#21262d', linewidth=0.5, linestyle='--')
            ax.set_xlabel("Thời gian (giây trước hiện tại)", color='#8b949e', fontsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#30363d')

        _style_ax(self.ax_raw,  "Raw EMG  (ADC, centered)",
                  "ADC − 2048", 'darkorange')
        _style_ax(self.ax_filt, "Filtered EMG  (20–450 Hz bandpass + notch 50 Hz)",
                  "mV", '#00d4ff')

        # Lines — raw dày hơn cũ, filtered sắc nét
        self.ln_raw,  = self.ax_raw.plot(
            xs, np.zeros(DISPLAY_N), color='#f0883e', lw=0.9, alpha=0.85)
        self.ln_filt, = self.ax_filt.plot(
            xs, np.zeros(DISPLAY_N), color='#58a6ff', lw=1.4)

        # Đường baseline y=0 để dễ đọc biên độ
        self.ax_raw.axhline(0, color='#30363d', lw=0.8, ls=':')
        self.ax_filt.axhline(0, color='#30363d', lw=0.8, ls=':')

        self.canvas = FigureCanvasTkAgg(fig, self.root)
        self.canvas.get_tk_widget().pack(fill='x', padx=6, pady=(6, 0))

        # ── Status bar ──────────────────────────────────────────────
        sf = tk.Frame(self.root, bg='#161b22', padx=12, pady=5)
        sf.pack(fill='x', padx=6)

        self.lbl_sess = tk.Label(sf, text=f"Session {self.session_num}/{SESSIONS}",
                                  font=('Arial', 12, 'bold'), fg='#e94560', bg='#161b22')
        self.lbl_sess.pack(side='left')

        self.lbl_lbl = tk.Label(sf, text="▶ REST",
                                 font=('Arial', 13, 'bold'), fg='#aaa', bg='#161b22')
        self.lbl_lbl.pack(side='left', padx=20)

        self.lbl_cnt = tk.Label(sf, text="0 samples", font=('Arial', 10),
                                 fg='#8b949e', bg='#161b22')
        self.lbl_cnt.pack(side='left')

        self.lbl_time = tk.Label(sf, text="00:00", font=('Arial', 12, 'bold'),
                                  fg='#ffd60a', bg='#161b22')
        self.lbl_time.pack(side='right')

        # ── Gesture buttons ─────────────────────────────────────────
        gf = tk.Frame(self.root, bg='#1a1a2e', pady=8)
        gf.pack()
        self.g_btns = {}
        specs = [('rest',  '⬜ REST',  '#3d3d3d'),
                 ('fist',  '✊ FIST',  '#9b2335'),
                 ('flex',  '🤙 FLEX',  '#1a6b3c'),
                 ('pinch', '🤌 PINCH', '#5a2d82')]
        for col, (g, txt, clr) in enumerate(specs):
            lbl_txt = txt if g == 'rest' else f"{txt}\n0 / {TARGET_REPS}"
            btn = tk.Button(gf, text=lbl_txt,
                            font=('Arial', 11, 'bold'), width=14, height=2,
                            bg=clr, fg='white', relief='flat', cursor='hand2',
                            activebackground=clr, activeforeground='white',
                            command=lambda x=g: self._set_label(x))
            btn.grid(row=0, column=col, padx=8)
            self.g_btns[g] = btn

        # ── Control buttons ─────────────────────────────────────────
        cf = tk.Frame(self.root, bg='#1a1a2e', pady=6)
        cf.pack()

        self.btn_conn = tk.Button(cf, text="🔌 Kết nối ESP32",
                                   font=('Arial', 11), bg='#1d4e3f', fg='white',
                                   relief='flat', width=18, height=2,
                                   command=self._connect)
        self.btn_conn.grid(row=0, column=0, padx=8)

        self.btn_rec = tk.Button(cf, text="▶ Bắt đầu ghi",
                                  font=('Arial', 11, 'bold'), bg='#2d6a4f', fg='white',
                                  relief='flat', width=18, height=2,
                                  state='disabled', command=self._toggle_rec)
        self.btn_rec.grid(row=0, column=1, padx=8)

        self.btn_save = tk.Button(cf, text=f"💾 Lưu Session {self.session_num}",
                                   font=('Arial', 11, 'bold'), bg='#1d3557', fg='white',
                                   relief='flat', width=20, height=2,
                                   state='disabled', command=self._save)
        self.btn_save.grid(row=0, column=2, padx=8)

    # ── Label / gesture ───────────────────────────────────────────────
    def _set_label(self, label):
        if not self.recording:
            return
        with self.lock:
            if self.current_label == 'rest' and label == 'rest':
                return
            self.current_label = label
            self.gesture_id += 1
            if label in GESTURES:
                self.rep_counts[label] += 1

        color = GESTURE_COLORS.get(label, '#aaa')
        self.lbl_lbl.config(text=f"▶ {label.upper()}", fg=color)
        for g in GESTURES:
            emoji = {'fist': '✊', 'flex': '🤙', 'pinch': '🤌'}[g]
            self.g_btns[g].config(
                text=f"{emoji} {g.upper()}\n{self.rep_counts[g]} / {TARGET_REPS}")

    # ── Serial ────────────────────────────────────────────────────────
    def _connect(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            # Chờ DTR reset (ESP32 có thể reboot khi Python mở port)
            time.sleep(0.5)
            # Xóa garbage boot — an toàn vì chưa stream dữ liệu thực
            self.ser.reset_input_buffer()
            self.stop_evt.clear()
            threading.Thread(target=self._reader, daemon=True).start()
            # BUG FIX 2: gửi S\n để bắt đầu stream ngay khi kết nối.
            # Cần thiết cho cả lần đầu lẫn các lần sau khi ESP32 ở streaming=false.
            self.ser.write(b'S\n')
            self.btn_conn.config(text="✅ Đã kết nối", state='disabled', bg='#1d3557')
            self.btn_rec.config(state='normal')
        except Exception as e:
            messagebox.showerror("Lỗi kết nối", str(e))

    def _reader(self):
        _invalid_streak = 0
        while not self.stop_evt.is_set():
            try:
                avail = self.ser.in_waiting
                if avail < 2:
                    time.sleep(0.001)
                    continue
                n = avail - (avail % 2)
                raw_bytes = self.ser.read(n)
            except Exception:
                break

            arr   = np.frombuffer(raw_bytes, dtype=np.uint8)
            highs = arr[0::2].astype(np.uint16)
            lows  = arr[1::2].astype(np.uint16)
            vals  = ((highs << 8) | lows).astype(float)
            valid = vals[vals <= 4095]

            if len(valid) == 0:
                _invalid_streak += 1
                # Self-healing: nếu liên tục nhận byte lệch, dịch alignment +1
                if _invalid_streak >= 3 and self.ser.in_waiting >= 1:
                    self.ser.read(1)
                    _invalid_streak = 0
                continue
            _invalid_streak = 0

            # Lọc tín hiệu (sosfilt + SOS initial conditions)
            notched,  self.zi_n = sosfilt(self.sos_n, valid, zi=self.zi_n)
            filtered, self.zi_b = sosfilt(self.sos_b, notched, zi=self.zi_b)

            # Đưa vào buffer hiển thị (ADC centered)
            raw_centered = valid - 2048.0
            for v, f in zip(raw_centered, filtered):
                self.raw_disp.append(v)
                self.filt_disp.append(f)

            if not self.recording:
                continue

            with self.lock:
                lbl = self.current_label
                gid = self.gesture_id

            for rv, fv in zip(valid, filtered):
                self.ds_accum.append((rv, fv))
                if len(self.ds_accum) >= DOWNSAMPLE:
                    chunk = np.array(self.ds_accum)
                    avg_raw_adc = np.mean(chunk[:, 0])
                    avg_filt    = np.mean(chunk[:, 1])
                    # Convert ADC → mV (AD8232 centered at VCC/2 = 1.65V)
                    raw_mv  = round((avg_raw_adc - 2048) / 4095 * 3300, 4)
                    filt_mv = round((avg_filt    - 2048) / 4095 * 3300, 4)
                    with self.lock:
                        self.csv_rows.append([
                            self.sample_ts, raw_mv, filt_mv, lbl, gid
                        ])
                    self.sample_ts += int(1000 / FS_OUT)
                    self.ds_accum.clear()

    # ── Recording ─────────────────────────────────────────────────────
    def _toggle_rec(self):
        if not self.recording:
            # Reset state phía Python
            self.csv_rows.clear()
            self.ds_accum.clear()
            self.sample_ts = 0
            self.gesture_id = 0
            self.rep_counts = {g: 0 for g in GESTURES}
            self.current_label = 'rest'

            # Reset filter state để tránh spike khi bắt đầu lại
            self.zi_n = sosfilt_zi(self.sos_n) * 2048.0
            self.zi_b = sosfilt_zi(self.sos_b) * 0.0

            if self.ser:
                # BUG FIX 1: KHÔNG gọi reset_input_buffer() ở đây —
                # nó có thể xóa buffer giữa chừng 1 sample 2-byte, gây lệch
                # byte vĩnh viễn. Chỉ gửi S\n để ESP32 tự reset ring buffer.
                self.ser.write(b'S\n')

            # BUG FIX 3: set recording=True SAU khi gửi S\n để reader
            # loại bỏ các byte cũ (recording=False) trước khi bắt đầu ghi CSV.
            self._rec_start = time.time()
            self.recording = True

            self.lbl_lbl.config(text="▶ REST", fg=GESTURE_COLORS['rest'])
            self.btn_rec.config(text="⏸ Đang ghi...", bg='#c1121f')
            self.btn_save.config(state='normal')
        else:
            self.recording = False
            if self.ser:
                self.ser.write(b'X\n')
            self.btn_rec.config(text="▶ Tiếp tục ghi", bg='#2d6a4f')

    # ── Save ──────────────────────────────────────────────────────────
    def _save(self):
        with self.lock:
            rows = list(self.csv_rows)
        if not rows:
            messagebox.showwarning("Rỗng", "Chưa có dữ liệu!")
            return

        path = os.path.join(SAVE_DIR, f'session_{self.session_num}.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['timestamp_ms', 'raw_adc', 'filtered_emg',
                        'label', 'gesture_id'])
            w.writerows(rows)

        dur  = len(rows) / FS_OUT
        reps = {g: self.rep_counts[g] for g in GESTURES}
        messagebox.showinfo("Đã lưu",
                            f"📁 {path}\n"
                            f"Mẫu: {len(rows)} (~{dur:.0f}s)\n"
                            f"Reps: {reps}")

        if self.session_num < SESSIONS:
            self.session_num += 1
            self.recording = False
            self.csv_rows.clear()
            self.rep_counts = {g: 0 for g in GESTURES}
            self.current_label = 'rest'
            self.gesture_id = 0
            self.lbl_sess.config(text=f"Session {self.session_num}/{SESSIONS}")
            self.btn_rec.config(text="▶ Bắt đầu ghi", bg='#2d6a4f')
            self.btn_save.config(text=f"💾 Lưu Session {self.session_num}",
                                  state='disabled')
        else:
            messagebox.showinfo("Xong!", "✅ Đã thu thập đủ 3 sessions!")
            self.btn_save.config(state='disabled')
            self.btn_rec.config(state='disabled')

    # ── UI update loop ────────────────────────────────────────────────
    def _update_ui(self):
        raw_arr  = np.array(self.raw_disp)
        filt_arr = np.array(self.filt_disp)

        self.ln_raw.set_ydata(raw_arr)
        self.ln_filt.set_ydata(filt_arr)

        # DELAY FIX 4: tính Y-limit thủ công — nhanh hơn relim()+autoscale_view()
        for ax, arr, margin in [(self.ax_raw, raw_arr, 80.0),
                                (self.ax_filt, filt_arr, 20.0)]:
            lo, hi = float(arr.min()), float(arr.max())
            span = hi - lo
            pad = max(margin, span * 0.12)
            ax.set_ylim(lo - pad, hi + pad)

        self.canvas.draw_idle()

        with self.lock:
            n = len(self.csv_rows)
        self.lbl_cnt.config(text=f"{n} samples @ 100 Hz")

        if self.recording:
            elapsed = time.time() - self._rec_start
            m, s = divmod(int(elapsed), 60)
            self.lbl_time.config(text=f"{m:02d}:{s:02d}")

        # DELAY FIX 1: refresh 40ms = 25fps (was 100ms = 10fps)
        self.root.after(40, self._update_ui)

    def _on_close(self):
        self.stop_evt.set()
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b'X\n')
                self.ser.close()
            except Exception:
                pass
        self.root.destroy()

    def run(self):
        self.root.after(300, self._update_ui)
        self.root.mainloop()


if __name__ == '__main__':
    app = EMGRecorder()
    app.run()
