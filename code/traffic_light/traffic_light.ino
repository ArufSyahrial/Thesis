// traffic light 1
#define red_1 12
#define yellow_1 13
#define green_1 14
#define button_1 2
#define buzzer_1 8 

void setup()
{
  // setting the traffic light 1 pins
  pinMode(red_1, OUTPUT);
  pinMode(yellow_1, OUTPUT);
  pinMode(green_1, OUTPUT);
  pinMode(button_1, INPUT);
  pinMode(buzzer_1, OUTPUT);

  // serial monitor
  Serial.begin(9600);

  // set traffic light 1 off
  digitalWrite(red_1, LOW);
  digitalWrite(yellow_1, LOW);
  digitalWrite(green_1, LOW);
}

void loop()
{  
  // turn green LED 1 on
  digitalWrite(red_1, LOW);
  digitalWrite(green_1, HIGH);
  Serial.println("GREEN");
  green_1_on();

  // turn yellow LED 1 on
  digitalWrite(green_1, LOW);
  digitalWrite(yellow_1, HIGH);
  Serial.println("\nYELLOW");
  delay(3000);

  // turn red LED 1 on
  digitalWrite(yellow_1, LOW);
  digitalWrite(red_1, HIGH);
  Serial.println("RED");
  delay(20000);
}

void green_1_on()
{
  int i = 0;
  
  // checking if button 1 is pressed during green LED cycle
  for(i = 1; i <= 20; i++)
  {
    int state_button_1 = digitalRead(button_1);

    // if button 1 is pressed == LOW
    if(state_button_1 == HIGH)
    {
      // print green LED 1 cycle duration 
      Serial.print(i);
      delay(1000);
      Serial.print(", ");
    }
    else if(state_button_1 == LOW)
    {
      Serial.print("PEDESTRIAN CROSSING ");

      // turn green LED 1 off & turn red LED 1 on
      digitalWrite(green_1, LOW);
      digitalWrite(red_1, HIGH);

      // turn buzzer 1 on
      buzzer_1_tone(buzzer_1, 1000, 5000);

      // turn green LED 1 on & turn red LED 1 off & continue the rest of green LED 1 cycle duration 
      digitalWrite(red_1, LOW);
      digitalWrite(green_1, HIGH);

      for (int j = i + 1; j <= 20; j++) {
        Serial.print(j);
        if (j < 20) {
          Serial.print(", ");
        }
        delay(1000);
      }

      return;
    }
  }
}

void buzzer_1_tone(int pin, int frequency, int duration)
{
  int period = 1000000L / frequency;  // Calculate the period in microseconds
  int pulse = period / 2;             // Calculate the pulse width

  for (long i = 0; i < duration * 1000L; i += period)
  {
    digitalWrite(pin, HIGH);  // Generate tone
    delayMicroseconds(pulse);
    digitalWrite(pin, LOW);
    delayMicroseconds(pulse);
  }
}
