"""
session_loader.py — Data loader cho bộ dữ liệu session tự thu thập.

Format CSV yêu cầu các cột:
    timestamp_ms | raw_emg | filtered_emg | label | repetition

Tương thích với:
    - create_sliding_windows()  trong data_utils.py
    - sEMGDataset               trong dataset.py
    - GestureLSTM               trong model.py  (input_size=1)

Cách dùng:
    from session_loader import load_and_split_session_csv, CLASS_NAMES

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_split_session_csv()
    # X_*: (N, 40, 1)   y_*: (N,)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Đường dẫn mặc định ────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / 'dummy_sessions'

# ── Label mapping ─────────────────────────────────────────────────────
LABEL_MAP  = {'rest': 0, 'fist': 1, 'flex': 2, 'pinch': 3}
CLASS_NAMES = ['rest', 'fist', 'flex', 'pinch']
NUM_CLASSES = len(LABEL_MAP)

# ── Repetition split (tương tự NinaPro convention) ────────────────────
TRAIN_REPS = set(range(1, 15))   # rep 1–14  (~70%)
VAL_REPS   = set(range(15, 18))  # rep 15–17 (~15%)
TEST_REPS  = set(range(18, 21))  # rep 18–20 (~15%)


def load_session_csvs(data_dir=None):
    """
    Load tất cả session_*.csv trong data_dir thành một DataFrame.

    Trả về:
        df (DataFrame) với cột bổ sung 'label_id' (integer 0-3)
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA
    data_dir = Path(data_dir)

    files = sorted(data_dir.glob('session_*.csv'))
    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy session_*.csv trong: {data_dir}\n"
            f"Hãy kiểm tra đường dẫn hoặc chạy generate_dummy.py.")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df['session'] = f.stem
        frames.append(df)
        print(f"  Loaded {f.name}: {len(df):,} rows")

    df_all = pd.concat(frames, ignore_index=True)

    # Validate required columns
    required = {'filtered_emg', 'label', 'repetition'}
    missing = required - set(df_all.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    # Encode string label → integer
    df_all['label_id'] = df_all['label'].map(LABEL_MAP)
    if df_all['label_id'].isna().any():
        unknown = df_all.loc[df_all['label_id'].isna(), 'label'].unique()
        raise ValueError(f"Nhãn không nhận dạng được: {unknown}")
    df_all['label_id'] = df_all['label_id'].astype(np.int64)

    print(f"\nTổng: {len(df_all):,} mẫu | "
          f"Sessions: {df_all['session'].nunique()} | "
          f"Labels: {dict(df_all['label'].value_counts().sort_index())}")
    return df_all


def _extract_windows(df, rep_set, window_size, step_size):
    """Lọc theo rep_set rồi tạo sliding windows."""
    from data_utils import create_sliding_windows

    sub = df[df['repetition'].isin(rep_set)].reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"Không có dữ liệu cho reps: {rep_set}")

    sig = sub['filtered_emg'].values.astype(np.float32).reshape(-1, 1)
    lbl = sub['label_id'].values.astype(np.int64)

    X, y = create_sliding_windows(sig, lbl, window_size, step_size)
    return X, y


def load_and_split_session_csv(data_dir=None, window_size=40, step_size=20):
    """
    Load sessions và chia train/val/test theo repetition index.

    Tham số:
        data_dir    : thư mục chứa session_*.csv (mặc định: dummy_sessions/)
        window_size : kích thước cửa sổ (mẫu), mặc định 40 = 400ms
        step_size   : bước trượt (mẫu), mặc định 20 = 50% overlap

    Trả về:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        X shape: (N, window_size, 1)    — 1 kênh filtered_emg
        y shape: (N,)                   — integer label [0–3]
    """
    df = load_session_csvs(data_dir)

    print("\nTạo sliding windows...")
    splits = {}
    for name, rep_set in [('train', TRAIN_REPS),
                           ('val',   VAL_REPS),
                           ('test',  TEST_REPS)]:
        X, y = _extract_windows(df, rep_set, window_size, step_size)
        splits[name] = (X, y)
        print(f"  {name:5s} (rep {sorted(rep_set)}): {len(X):,} windows")

    return splits['train'], splits['val'], splits['test']


if __name__ == '__main__':
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_and_split_session_csv()
    print(f"\nX_train: {X_tr.shape}  y_train: {y_tr.shape}")
    print(f"X_val  : {X_va.shape}  y_val  : {y_va.shape}")
    print(f"X_test : {X_te.shape}  y_test : {y_te.shape}")
    print(f"Classes: {CLASS_NAMES}")
