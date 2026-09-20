"""
VoiceArmor-Edge: Arduino UNO Q USB-CDC Serial Bridge
Dispatches out-of-band hardware alerts upon detection of synthetic voice spoofing.
"""

import time

ALERT_BYTE_TRIGGER = b'\xA1'
ALERT_BYTE_CLEAR = b'\xA0'


class ArduinoHardwareBridge:
    def __init__(self, port="COM3", baudrate=115200, enabled=True):
        self.port = port
        self.baudrate = baudrate
        self.enabled = enabled
        self.serial_conn = None
        if self.enabled:
            self._connect()

    def _connect(self):
        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(1.5)  # Allow Arduino bootloader to initialize
            print(f"[INFO] Successfully connected to Arduino UNO Q on {self.port}")
        except Exception as e:
            print(f"[Notice] Arduino UNO Q hardware module not detected ({e}). Continuing in software-only mode.")
            self.serial_conn = None

    def trigger_threat_alert(self):
        """Sends hardware alert command to trigger LED, buzzer, and isolation relay."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(ALERT_BYTE_TRIGGER)
                self.serial_conn.flush()
            except Exception as e:
                print(f"[Error] Failed to write to Arduino: {e}")
        else:
            print("[HARDWARE SIMULATION] -> Arduino UNO Q: RED ALERT LED ON | AUDIO RELAY OPEN")

    def clear_threat_alert(self):
        """Resets Arduino hardware alert state to normal standby."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(ALERT_BYTE_CLEAR)
                self.serial_conn.flush()
            except Exception as e:
                print(f"[Error] Failed to write to Arduino: {e}")
        else:
            print("[HARDWARE SIMULATION] -> Arduino UNO Q: Standby Mode (Green LED)")

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
