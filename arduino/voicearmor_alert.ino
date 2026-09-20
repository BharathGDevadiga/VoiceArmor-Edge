/*
  VoiceArmor-Edge: Hardware Alert Module for Arduino UNO Q
  Listens for serial alert commands from Snapdragon PC and drives physical indicators.
*/

const int PIN_RED_LED = 8;
const int PIN_GREEN_LED = 9;
const int PIN_BUZZER = 10;
const int PIN_AUDIO_RELAY = 11; // Normally closed relay for headset circuit

const byte CMD_ALERT_TRIGGER = 0xA1;
const byte CMD_ALERT_CLEAR   = 0xA0;

bool alertActive = false;

void setup() {
  Serial.begin(115200);
  
  pinMode(PIN_RED_LED, OUTPUT);
  pinMode(PIN_GREEN_LED, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_AUDIO_RELAY, OUTPUT);
  
  // Initial state: Safe
  setSafeState();
}

void loop() {
  if (Serial.available() > 0) {
    byte command = Serial.read();
    
    if (command == CMD_ALERT_TRIGGER && !alertActive) {
      setAlertState();
      alertActive = true;
    } else if (command == CMD_ALERT_CLEAR && alertActive) {
      setSafeState();
      alertActive = false;
    }
  }
}

void setAlertState() {
  digitalWrite(PIN_GREEN_LED, LOW);
  digitalWrite(PIN_RED_LED, HIGH);
  tone(PIN_BUZZER, 1000);          // Audible warning tone (1 kHz)
  digitalWrite(PIN_AUDIO_RELAY, HIGH); // Disconnect microphone circuit
}

void setSafeState() {
  digitalWrite(PIN_RED_LED, LOW);
  noTone(PIN_BUZZER);
  digitalWrite(PIN_GREEN_LED, HIGH);
  digitalWrite(PIN_AUDIO_RELAY, LOW);  // Microphone circuit normal
}
