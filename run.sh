#!/bin/bash
# ============================================================
#  sEMG Acquisition — Script chạy nhanh
#  Usage:
#    ./run.sh          → chỉ chạy test.py (không upload firmware)
#    ./run.sh upload   → upload firmware rồi chạy test.py
# ============================================================

EMG_DIR="$HOME/Documents/EMG Classification"
PIO_DIR="$HOME/Documents/PlatformIO/Projects/EMG_Firmware"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
ENV_NAME="emg"

# Màu sắc terminal
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   sEMG Recorder — Startup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Kích hoạt conda
source "$CONDA_SH"
conda activate "$ENV_NAME"

# ── Upload firmware nếu có tham số "upload" ──────────────────
if [ "$1" == "upload" ]; then
    echo -e "\n${YELLOW}[1/2] Copy firmware → PlatformIO project...${NC}"
    cp "$EMG_DIR/main.cpp" "$PIO_DIR/src/main.cpp"
    echo -e "      Done: main.cpp → $PIO_DIR/src/main.cpp"

    echo -e "\n${YELLOW}[2/2] Compile + Upload lên ESP32-C3...${NC}"
    cd "$PIO_DIR"
    pio run --target upload

    if [ $? -ne 0 ]; then
        echo -e "\n${RED}❌ Upload thất bại! Kiểm tra lại kết nối USB.${NC}"
        exit 1
    fi
    echo -e "\n${GREEN}✅ Upload thành công!${NC}"
else
    echo -e "\n${YELLOW}[Tip] Dùng './run.sh upload' để upload firmware trước.${NC}"
fi

# ── Kiểm tra ESP32 đã kết nối ───────────────────────────────
echo -e "\n${YELLOW}Kiểm tra kết nối ESP32...${NC}"
if ls /dev/ttyACM* 1>/dev/null 2>&1; then
    PORT=$(ls /dev/ttyACM* | head -1)
    echo -e "${GREEN}✅ Tìm thấy: $PORT${NC}"
else
    echo -e "${RED}⚠️  Không tìm thấy /dev/ttyACM* — cắm USB vào chưa?${NC}"
    echo -e "   Vẫn tiếp tục chạy test.py..."
fi

# ── Chạy test.py ─────────────────────────────────────────────
echo -e "\n${GREEN}▶ Khởi động sEMG Recorder GUI...${NC}\n"
cd "$EMG_DIR"
python test.py ${PORT:+--port $PORT}
