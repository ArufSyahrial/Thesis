const int redLight = 12;
const int yellowLight = 13;
const int greenLight = 14;
const int buzzer = 8;
const int button = 2;

void setup() {
  pinMode(redLight, OUTPUT);
  pinMode(yellowLight, OUTPUT);
  pinMode(greenLight, OUTPUT);
  pinMode(buzzer, OUTPUT);
  pinMode(button, INPUT_PULLUP); // Use internal pull-up resistor

  tone(buzzer, 1000, 2000);
}

void loop() {
  // Normal traffic light sequence
  digitalWrite(greenLight, HIGH);
  digitalWrite(yellowLight, LOW);
  digitalWrite(redLight, LOW);
  delay(10000); // Green light for 10 seconds

  digitalWrite(greenLight, LOW);
  digitalWrite(yellowLight, HIGH);
  delay(2000); // Yellow light for 2 seconds

  digitalWrite(yellowLight, LOW);
  digitalWrite(redLight, HIGH);
  delay(10000); // Red light for 10 seconds

  // Check if the pedestrian button is pressed
  if (digitalRead(button) == LOW) {
    digitalWrite(redLight, HIGH);
    beepBuzzer(5000); // Beep for 5 seconds

    delay(2000); // Additional delay to ensure pedestrians have crossed
  }

  digitalWrite(redLight, LOW);
}

void beepBuzzer(int duration) {
  for (int i = 0; i < duration / 100; i++) {
    tone(buzzer, 440); // A4
    delay(50);
    noTone(buzzer);
    delay(50);
  }
}
