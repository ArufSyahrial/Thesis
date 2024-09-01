//Circuit
// Arduino Uno  -->   TCRT5000
// 5v           --->   VCC
// Grnd         --->   Grnd
// A0           --->   A0
// D8           --->   D0


#define pinIRd 8
#define pinLED 12
int IRvalueA = 0;
int IRvalueD = 0;

void setup()
{
  Serial.begin(9600);
  pinMode(pinIRd,INPUT);
  pinMode(pinLED,OUTPUT);

}

void loop()
{
  Serial.print("\t Digital Reading=");
  Serial.println(IRvalueD);

    if (IRvalueD == LOW) {
    digitalWrite(pinLED, HIGH);
  }
  else {
    digitalWrite(pinLED, LOW);
  }


  delay(500);
  
  IRvalueD = digitalRead(pinIRd);

}