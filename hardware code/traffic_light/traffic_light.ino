#define button_1 11
#define red_led_1 12
#define yellow_led_1 13
#define green_led_1 14

int button_1_state = 0;

void setup() 
{
  pinMode(button_1, INPUT);
  pinMode(red_led_1, OUTPUT);
  pinMode(yellow_led_1, OUTPUT);
  pinMode(green_led_1, OUTPUT);
}

void loop() 
{
  // read state of button 1
  button_1_state = digitalRead(button_1);

  // if button is pressed, state == HIGH
  if(button_1_state == HIGH)
  {
    digitalWrite(red_led_1, HIGH);
  }
  else
  {
    digitalWrite(red_led_1, LOW);
  }
}
