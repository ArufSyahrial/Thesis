#define button_1 2
#define red_led_1 12
#define yellow_led_1 13
#define green_led_1 14

void setup() 
{
  pinMode(button_1, INPUT_PULLUP);
  pinMode(red_led_1, OUTPUT);
  pinMode(yellow_led_1, OUTPUT);
  pinMode(green_led_1, OUTPUT);
}

void loop() 
{
  // if button is pressed == LOW
  if(digitalRead(button_1) == LOW)
  {
    digitalWrite(red_led_1, HIGH);
  }
  else
  {
    digitalWrite(red_led_1, LOW);
  }
}
