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
from scipy.signal import butter, iirnotch, lfilter_zi, lfilter

# ── Cấu hình ──────────────────────────────────────────────────────
def _detect_port():
    """Tự tìm port ESP32: /dev/ttyACM* → lấy cái đầu tiên."""
    import glob
    ports = sorted(glob.glob('/dev/ttyACM*'))
    return ports[0] if ports else '/dev/ttyACM0'

import argparse as _ap
_parser = _ap.ArgumentParser()
_parser.add_argument('--port', default=None)
_args, _ = _parser.parse_known_args()
SERIAL_PORT = _args.port if _args.port else _detect_port()

BAUD_RATE   = 115200
FS          = 1000        # Hz firmware gửi
DOWNSAMPLE  = 10     # 1000 → 100 Hz
FS_OUT      = FS // DOWNSAMPLE
DISPLAY_N   = FS * 8      # 8 giây hiển thị
TARGET_REPS = 20
SESSIONS    = 3
GESTURES    = ['fist', 'flex', 'pinch']
SAVE_DIR    = os.path.join(os.path.expanduser('~'),
                           'Documents', 'EMG Classification')

GESTURE_COLORS = {'rest': '#555', 'fist': '#e94560',
                  'flex': '#0f7', 'pinch': '#c77dff'}


class EMGRecorder:
    def __init__(self):
        # Bộ lọc tín hiệu
        b_n, a_n = iirnotch(50.0, 30.0, FS)
        nyq = 0.5 * FS
        b_b, a_b = butter(4, [20.0/nyq, 450.0/nyq], btype='band')
        self.b_n, self.a_n = b_n, a_n
        self.b_b, self.a_b = b_b, a_b
        self.zi_n = lfilter_zi(b_n, a_n) * 2048.0
        self.zi_b = lfilter_zi(b_b, a_b) * 0.0

        # Buffer hiển thị
        self.raw_disp  = collections.deque([0.0]*DISPLAY_N, maxlen=DISPLAY_N)
        self.filt_disp = collections.deque([0.0]*DISPLAY_N, maxlen=DISPLAY_N)

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
        self.root.geometry('960x760')
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Plot
        fig = Figure(figsize=(9.6, 4.5), facecolor='#0f3460')
        self.ax = fig.add_subplot(111, facecolor='#16213e')
        self.ax.set_xlim(0, DISPLAY_N)
        self.ax.tick_params(colors='white')
        self.ax.set_ylabel('mV (centered)', color='white', fontsize=9)
        self.ax.set_autoscaley_on(True)
        for sp in self.ax.spines.values():
            sp.set_edgecolor('#333')
        xs = np.arange(DISPLAY_N)
        self.ln_raw,  = self.ax.plot(xs, np.zeros(DISPLAY_N),
                                     color='darkorange', lw=0.5,
                                     alpha=0.6, label='Raw')
        self.ln_filt, = self.ax.plot(xs, np.zeros(DISPLAY_N),
                                     color='#00d4ff', lw=1.2, label='Filtered')
        self.ax.legend(loc='upper right', facecolor='#16213e',
                       labelcolor='white', fontsize=8)
        self.canvas = FigureCanvasTkAgg(fig, self.root)
        self.canvas.get_tk_widget().pack(fill='x', padx=8, pady=(8, 2))

        # Status bar
        sf = tk.Frame(self.root, bg='#16213e', padx=12, pady=5)
        sf.pack(fill='x', padx=8)
        self.lbl_sess = tk.Label(sf, text=f"Session {self.session_num}/{SESSIONS}",
                                  font=('Arial', 12, 'bold'), fg='#e94560', bg='#16213e')
        self.lbl_sess.pack(side='left')
        self.lbl_lbl = tk.Label(sf, text="▶ REST",
                                 font=('Arial', 13, 'bold'), fg='#00d4ff', bg='#16213e')
        self.lbl_lbl.pack(side='left', padx=24)
        self.lbl_cnt = tk.Label(sf, text="0 samples", font=('Arial', 10),
                                 fg='#888', bg='#16213e')
        self.lbl_cnt.pack(side='left')
        self.lbl_time = tk.Label(sf, text="00:00", font=('Arial', 12, 'bold'),
                                  fg='#ffd60a', bg='#16213e')
        self.lbl_time.pack(side='right')

        # Gesture buttons
        gf = tk.Frame(self.root, bg='#1a1a2e', pady=10)
        gf.pack()
        self.g_btns = {}
        specs = [('rest', '⬜ REST', '#555'),
                 ('fist', '✊ FIST', '#e94560'),
                 ('flex', '🤙 FLEX', '#0a7'),
                 ('pinch','🤌 PINCH','#7b2d8b')]
        for col, (g, txt, clr) in enumerate(specs):
            lbl_txt = txt if g == 'rest' else f"{txt}\n0 / {TARGET_REPS}"
            btn = tk.Button(gf, text=lbl_txt,
                            font=('Arial', 11, 'bold'), width=13, height=2,
                            bg=clr, fg='white', relief='flat', cursor='hand2',
                            command=lambda x=g: self._set_label(x))
            btn.grid(row=0, column=col, padx=8)
            self.g_btns[g] = btn

        # Control buttons
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
            # Chỉ bỏ qua nếu đang REST mà click REST lại
            if self.current_label == 'rest' and label == 'rest':
                return
            self.current_label = label
            self.gesture_id += 1
            if label in GESTURES:
                self.rep_counts[label] += 1

        self.lbl_lbl.config(text=f"▶ {label.upper()}",
                             fg=GESTURE_COLORS.get(label, '#00d4ff'))
        # Cập nhật số rep trên tất cả các nút gesture
        for g in GESTURES:
            emoji = {'fist': '✊', 'flex': '🤙', 'pinch': '🤌'}[g]
            self.g_btns[g].config(
                text=f"{emoji} {g.upper()}\n{self.rep_counts[g]} / {TARGET_REPS}"
            )

    # ── Serial ────────────────────────────────────────────────────────
    def _connect(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
            self.ser.reset_input_buffer()
            self.stop_evt.clear()
            threading.Thread(target=self._reader, daemon=True).start()
            self.btn_conn.config(text="✅ Đã kết nối", state='disabled', bg='#1d3557')
            self.btn_rec.config(state='normal')
        except Exception as e:
            messagebox.showerror("Lỗi kết nối", str(e))

    def _reader(self):
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
                continue

            # Lọc tín hiệu
            notched, self.zi_n = lfilter(self.b_n, self.a_n, valid, zi=self.zi_n)
            filtered, self.zi_b = lfilter(self.b_b, self.a_b, notched, zi=self.zi_b)

            for v, f in zip(valid - 2048.0, filtered):
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
                    # Convert ADC → mV (AD8232 centered at VCC/2 = 1.65V)
                    # raw_mv: giá trị tín hiệu sau khi trừ DC offset, đơn vị mV
                    avg_raw_adc = np.mean(chunk[:, 0])
                    avg_filt    = np.mean(chunk[:, 1])
                    raw_mv   = round((avg_raw_adc - 2048) / 4095 * 3300, 4)
                    filt_mv  = round((avg_filt    - 2048) / 4095 * 3300, 4)
                    with self.lock:
                        self.csv_rows.append([
                            self.sample_ts,
                            raw_mv,
                            filt_mv,
                            lbl, gid
                        ])
                    self.sample_ts += int(1000 / FS_OUT)
                    self.ds_accum.clear()

    # ── Recording ─────────────────────────────────────────────────────
    def _toggle_rec(self):
        if not self.recording:
            self.recording = True
            self.csv_rows.clear()
            self.ds_accum.clear()
            self.sample_ts = 0
            self.gesture_id = 0
            self.rep_counts = {g: 0 for g in GESTURES}
            self.current_label = 'rest'
            self._rec_start = time.time()

            # Reset filter state để tránh "tín hiệu nhảy" khi bắt đầu lại
            self.zi_n = lfilter_zi(self.b_n, self.a_n) * 2048.0
            self.zi_b = lfilter_zi(self.b_b, self.a_b) * 0.0

            # Flush serial buffer cũ trước khi ghi
            if self.ser:
                self.ser.reset_input_buffer()
                self.ser.write(b'S\n')

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

        dur = len(rows) / FS_OUT
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
        raw_arr = np.array(self.raw_disp)
        filt_arr = np.array(self.filt_disp)
        self.ln_raw.set_ydata(raw_arr)
        self.ln_filt.set_ydata(filt_arr)
        # Auto-scale trục Y theo tín hiệu hiện tại
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw_idle()

        with self.lock:
            n = len(self.csv_rows)
        self.lbl_cnt.config(text=f"{n} samples (100Hz)")

        if self.recording:
            elapsed = time.time() - self._rec_start
            m, s = divmod(int(elapsed), 60)
            self.lbl_time.config(text=f"{m:02d}:{s:02d}")

        self.root.after(100, self._update_ui)

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
