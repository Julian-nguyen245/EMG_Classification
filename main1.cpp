#include <Arduino.h>

#define EMG_PIN A1

// Tạo một bộ đệm vòng (Ring Buffer) nhỏ để chuyển tiếp dữ liệu từ hàm ngắt ra
// loop
const int BUFFER_SIZE = 512;
volatile uint16_t ringBuffer[BUFFER_SIZE];
volatile int head = 0;
volatile int tail = 0;

// Khai báo cấu trúc Timer theo chuẩn Core v2.x của bạn
hw_timer_t *timer = NULL;

// Hàm ngắt (ISR) chạy ngầm - Cứ mỗi 1ms tự động đọc 1 mẫu
void IRAM_ATTR onTimer() {
  int nextHead = (head + 1) % BUFFER_SIZE;

  // Nếu bộ đệm chưa bị đầy, ghi giá trị ADC mới vào
  if (nextHead != tail) {
    ringBuffer[head] = analogRead(EMG_PIN);
    head = nextHead;
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  // CẤU HÌNH TIMER CHUẨN CORE v2.x (Đã sửa đổi dùng Timer 1 tránh xung đột)
  timer = timerBegin(1, 80, true);
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 1000, true); // 1000 microgiây = 1ms (1000Hz)
  timerAlarmEnable(timer);
}

void loop() {
  // Hàm loop sẽ liên tục kiểm tra xem 'ringBuffer' có dữ liệu mới không
  while (tail != head) {
    // Lấy dữ liệu ra từ bộ đệm
    uint16_t data = ringBuffer[tail];
    tail = (tail + 1) % BUFFER_SIZE;

    // TÁCH SỐ 16-BIT THÀNH 2 BYTE NHỊ PHÂN ĐỂ GỬI QUA PYTHON
    uint8_t lowByte = data & 0xFF;         // Lấy 8 bit thấp
    uint8_t highByte = (data >> 8) & 0xFF; // Lấy 8 bit cao

    // Bắn dữ liệu thô dạng nhị phân cực nhanh
    Serial.write(highByte);
    Serial.write(lowByte);
  }
}