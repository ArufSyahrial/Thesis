#define LED1 2
#define LED2 3
//#define Sensor 5

void setup() {
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(Sensor, INPUT);
}
void loop() {
  bool value = digitalRead(Sensor);
  if (value == 1) {
    digitalWrite(LED1, HIGH);
    digitalWrite(LED2, LOW);
  } else if (value == 0) {
    digitalWrite(LED2, HIGH);
    digitalWrite(LED1, LOW);
  }
}