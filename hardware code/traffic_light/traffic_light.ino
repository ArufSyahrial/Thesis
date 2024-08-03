#define pin_tombol 11
#define pin_led_merah 12
#define pin_led_kuning 13
#define pin_led_hijau 14

bool tombol_siap = true;
bool lampu_hijau_menyala = false;

void setup() 
{
  pinMode(pin_led_merah, OUTPUT);
  pinMode(pin_led_kuning, OUTPUT);
  pinMode(pin_led_hijau, OUTPUT);
  pinMode(pin_tombol, INPUT_PULLUP);
}

void loop() 
{
  if(digitalRead(pin_tombol) == LOW && tombol_siap)
  {
    // Ketika tombol ditekan, nyalakan LED merah selama beberapa detik
    digitalWrite(pin_led_merah, HIGH);
    digitalWrite(pin_led_kuning, LOW);
    digitalWrite(pin_led_hijau, LOW);
    delay(3000); // LED merah menyala selama 3 detik (atur sesuai kebutuhan)
    tombol_siap = false;
  }
  else if(digitalRead(pin_tombol) == HIGH && !tombol_siap)
  {
    tombol_siap = true;
  }
  else if(tombol_siap)
  {
    // Simulasikan lampu lalu lintas ketika tombol tidak ditekan
    digitalWrite(pin_led_merah, HIGH);
    digitalWrite(pin_led_kuning, LOW);
    digitalWrite(pin_led_hijau, LOW);
    delay(5000); // LED merah menyala selama 5 detik

    digitalWrite(pin_led_merah, LOW);
    digitalWrite(pin_led_kuning, HIGH);
    digitalWrite(pin_led_hijau, LOW);
    delay(2000); // LED kuning menyala selama 2 detik

    // Mengatur status lampu hijau menyala
    lampu_hijau_menyala = true;
    digitalWrite(pin_led_merah, LOW);
    digitalWrite(pin_led_kuning, LOW);
    digitalWrite(pin_led_hijau, HIGH);
    delay(5000); // LED hijau menyala selama 5 detik
    lampu_hijau_menyala = false; // Reset status lampu hijau setelah delay
  }

  // Pengecekan tambahan jika tombol ditekan saat lampu hijau menyala
  if(digitalRead(pin_tombol) == LOW && lampu_hijau_menyala && tombol_siap)
  {
    digitalWrite(pin_led_merah, HIGH);
    digitalWrite(pin_led_kuning, LOW);
    digitalWrite(pin_led_hijau, LOW);
    delay(3000); // LED merah menyala selama 3 detik (atur sesuai kebutuhan)
    tombol_siap = false;
  }
}
