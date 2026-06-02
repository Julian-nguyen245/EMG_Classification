#include <Arduino.h>

#define EMG_PIN A1  // GPIO3 — GP1 trên XIAO ESP32-C3

// Ring buffer
const int BUFFER_SIZE = 256;
volatile uint16_t ringBuffer[BUFFER_SIZE];
volatile int head = 0;
volatile int tail = 0;
volatile bool streaming = false;

hw_timer_t *timer = NULL;

// ISR: chạy 1000Hz, đọc ADC vào ring buffer
void IRAM_ATTR onTimer() {
    if (!streaming) return;
    int nextHead = (head + 1) % BUFFER_SIZE;
    if (nextHead != tail) {
        ringBuffer[head] = analogRead(EMG_PIN);
        head = nextHead;
    }
}

void setup() {
    Serial.begin(115200);
    analogReadResolution(12);
    analogSetAttenuation(ADC_11db); // AD8232 output 0-3.3V → full 12-bit range

    streaming = true; // luôn stream để GUI hiển thị realtime ngay khi kết nối

    // Timer API cũ (framework-arduinoespressif32 @ 3.x)
    timer = timerBegin(0, 80, true);          // timer 0, divider 80 → 1MHz
    timerAttachInterrupt(timer, &onTimer, true);
    timerAlarmWrite(timer, 1000, true);       // 1000us = 1ms → 1000Hz
    timerAlarmEnable(timer);
}

void loop() {
    // Nhận lệnh từ Python: "S\n" = bắt đầu, "X\n" = dừng
    while (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd == "S") {
            head = tail = 0; // xóa buffer cũ
            streaming = true;
        } else if (cmd == "X") {
            streaming = false;
        }
    }

    // Gửi dữ liệu: 2 bytes/sample (high byte trước)
    while (tail != head) {
        uint16_t data = ringBuffer[tail];
        tail = (tail + 1) % BUFFER_SIZE;
        Serial.write((uint8_t)((data >> 8) & 0xFF));
        Serial.write((uint8_t)(data & 0xFF));
    }
}