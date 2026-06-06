#include <Arduino.h>
#include "driver/i2s.h"

// TensorFlow Lite untuk ESP32
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model yang sudah di-convert
#include "model_ambulance_siren.h"

// Konfigurasi sampling audio
#define SAMPLE_RATE 16000
#define RECORD_MS 1000
#define NUM_SAMPLES (SAMPLE_RATE * RECORD_MS / 1000)
#define NUM_FEATURES 80

// Pin untuk 3 MODUL LAMPU LALU LINTAS TERPISAH (GPIO yang aman untuk ESP32-S3)
// Modul 1
#define RED_1    GPIO_NUM_1
#define YELLOW_1 GPIO_NUM_2
#define GREEN_1  GPIO_NUM_42

// Modul 2
#define RED_2    GPIO_NUM_4
#define YELLOW_2 GPIO_NUM_5
#define GREEN_2  GPIO_NUM_6

// Modul 3
#define RED_3    GPIO_NUM_7
#define YELLOW_3 GPIO_NUM_15
#define GREEN_3  GPIO_NUM_16

// Pin I2S untuk dua mikrofon INMP441 (Stereo - 1 I2S Bus)
// Shared pins untuk kedua mikrofon
#define I2S_WS      GPIO_NUM_41  // Word Select (LRCL)
#define I2S_SCK     GPIO_NUM_40  // Bit Clock (BCLK)
// Data pins terpisah untuk tiap mikrofon
#define I2S_SD_L    GPIO_NUM_39  // Mikrofon Kiri (SD)
#define I2S_SD_R    GPIO_NUM_38  // Mikrofon Kanan (SD) - akan di-wire ke SD

// TensorFlow Lite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Memory untuk TensorFlow Lite - dikurangi untuk stabilitas
  constexpr int kTensorArenaSize = 50 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// Buffer audio untuk stereo (Left + Right channel)
// Ukuran lebih kecil untuk menghindari memory issue
#define BUFFER_SIZE 512
int32_t stereo_buffer[BUFFER_SIZE];
int32_t audio_buffer_left[NUM_SAMPLES];
int32_t audio_buffer_right[NUM_SAMPLES];
float features_left[NUM_FEATURES];
float features_right[NUM_FEATURES];

// Status traffic light
enum TrafficLightState {
  NORMAL_OPERATION,
  AMBULANCE_EMERGENCY,
  SYSTEM_ERROR
};

TrafficLightState currentState = NORMAL_OPERATION;
unsigned long lastStateChange = 0;
unsigned long lastAudioCheck = 0;

// Timing untuk siklus lampu lalu lintas real
const unsigned long RED_DURATION = 15000;      // Merah: 15 detik
const unsigned long YELLOW_DURATION = 5000;    // Kuning: 5 detik
const unsigned long GREEN_DURATION = 20000;    // Hijau: 20 detik
const unsigned long EMERGENCY_TIME = 30000;    // Emergency: 30 detik
const unsigned long AUDIO_CHECK_INTERVAL = 1500; // Check audio setiap 1.5 detik

// Statistik
unsigned long totalInferenceTime = 0;
unsigned int inferenceCount = 0;
unsigned int ambulanceDetectionCount = 0;

// Konfigurasi I2S driver untuk stereo (2 mikrofon)
void initI2S() {
  Serial.println("🎤 Configuring I2S for dual INMP441 microphones (Stereo)...");
  
  // Konfigurasi I2S untuk stereo input
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,  // Stereo
      .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = BUFFER_SIZE,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0
  };

  // Pin configuration untuk stereo
  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD_L  // Hanya perlu 1 data pin untuk stereo
  };

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf("❌ I2S driver install failed: %d\n", err);
    setErrorState();
    return;
  }
  
  err = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err != ESP_OK) {
    Serial.printf("❌ I2S pin config failed: %d\n", err);
    setErrorState();
    return;
  }
  
  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println("✅ Stereo I2S initialized successfully");
}

void blinkAllYellow(int times) {
  for (int i = 0; i < times; i++) {
    // Nyalakan semua kuning
    digitalWrite(YELLOW_1, HIGH);
    digitalWrite(YELLOW_2, HIGH);
    digitalWrite(YELLOW_3, HIGH);
    delay(300);
    // Matikan semua kuning
    digitalWrite(YELLOW_1, LOW);
    digitalWrite(YELLOW_2, LOW);
    digitalWrite(YELLOW_3, LOW);
    delay(300);
  }
}

void setErrorState() {
  currentState = SYSTEM_ERROR;
  Serial.println("\n❌❌❌ SYSTEM ERROR - Entering error state ❌❌❌");
  Serial.println("⚠️  Press RESET button to restart");
  
  while (true) {
    // Kedipkan semua merah
    digitalWrite(RED_1, HIGH);
    digitalWrite(RED_2, HIGH);
    digitalWrite(RED_3, HIGH);
    digitalWrite(YELLOW_1, LOW);
    digitalWrite(YELLOW_2, LOW);
    digitalWrite(YELLOW_3, LOW);
    digitalWrite(GREEN_1, LOW);
    digitalWrite(GREEN_2, LOW);
    digitalWrite(GREEN_3, LOW);
    delay(400);
    digitalWrite(RED_1, LOW);
    digitalWrite(RED_2, LOW);
    digitalWrite(RED_3, LOW);
    delay(400);
  }
}

void setAllTrafficLights(int red, int yellow, int green) {
  // Set semua modul ke state yang sama
  digitalWrite(RED_1, red);
  digitalWrite(YELLOW_1, yellow);
  digitalWrite(GREEN_1, green);
  
  digitalWrite(RED_2, red);
  digitalWrite(YELLOW_2, yellow);
  digitalWrite(GREEN_2, green);
  
  digitalWrite(RED_3, red);
  digitalWrite(YELLOW_3, yellow);
  digitalWrite(GREEN_3, green);

  static int lastRed = -1, lastYellow = -1, lastGreen = -1;
  if (red != lastRed || yellow != lastYellow || green != lastGreen) {
    lastRed = red; lastYellow = yellow; lastGreen = green;
    
    const char* color = "";
    if (red) color = "🔴 RED";
    else if (yellow) color = "🟡 YELLOW";
    else if (green) color = "🟢 GREEN";
    
    Serial.printf("🚦 All Traffic Lights: %s\n", color);
  }
}

// Ekstraksi fitur audio sederhana
void extractFeatures(int32_t* audio, float* features) {
  // Normalisasi audio ke range [-1, 1]
  float normalized[NUM_SAMPLES];
  for (int i = 0; i < NUM_SAMPLES; i++) {
    normalized[i] = (float)audio[i] / 2147483648.0f;
  }

  // Pre-emphasis filter
  float emphasized[NUM_SAMPLES];
  emphasized[0] = normalized[0];
  for (int i = 1; i < NUM_SAMPLES; i++) {
    emphasized[i] = normalized[i] - 0.97f * normalized[i - 1];
  }

  // Hitung energi per frame
  int frame_size = NUM_SAMPLES / NUM_FEATURES;
  for (int frame = 0; frame < NUM_FEATURES; frame++) {
    float frame_energy = 0.0f;
    int start = frame * frame_size;
    int end = min(start + frame_size, NUM_SAMPLES);

    for (int i = start; i < end; i++) {
      frame_energy += emphasized[i] * emphasized[i];
    }
    features[frame] = sqrtf(frame_energy / (end - start + 1e-8f));
  }

  // Normalisasi features
  float mean = 0.0f, std_dev = 0.0f;
  for (int i = 0; i < NUM_FEATURES; i++) {
    mean += features[i];
  }
  mean /= NUM_FEATURES;

  for (int i = 0; i < NUM_FEATURES; i++) {
    float diff = features[i] - mean;
    std_dev += diff * diff;
  }
  std_dev = sqrtf(std_dev / NUM_FEATURES);

  if (std_dev < 1e-8f) std_dev = 1e-8f;

  for (int i = 0; i < NUM_FEATURES; i++) {
    features[i] = (features[i] - mean) / std_dev;
  }
}

// Baca audio stereo dan pisahkan ke left/right channel
bool readStereoAudio() {
  int samples_read_left = 0;
  int samples_read_right = 0;
  
  Serial.println("🎤 Reading stereo audio...");
  
  // Baca data stereo dalam chunks
  while (samples_read_left < NUM_SAMPLES) {
    size_t bytesRead = 0;
    esp_err_t err = i2s_read(I2S_NUM_0, stereo_buffer, sizeof(stereo_buffer), 
                              &bytesRead, pdMS_TO_TICKS(100));
    
    if (err != ESP_OK) {
      Serial.printf("⚠️ I2S read error: %d\n", err);
      return false;
    }
    
    if (bytesRead == 0) {
      Serial.println("⚠️ No data read from I2S");
      continue;
    }
    
    // Pisahkan stereo data ke left dan right channel
    int samples_in_buffer = bytesRead / 8; // 4 bytes per sample × 2 channels
    
    for (int i = 0; i < samples_in_buffer && samples_read_left < NUM_SAMPLES; i++) {
      audio_buffer_left[samples_read_left] = stereo_buffer[i * 2];      // Left channel
      audio_buffer_right[samples_read_right] = stereo_buffer[i * 2 + 1]; // Right channel
      samples_read_left++;
      samples_read_right++;
    }
  }
  
  Serial.printf("✅ Read %d samples from each microphone\n", samples_read_left);
  return true;
}

// Klasifikasi audio dengan voting dari 2 mikrofon
void classifyAudio() {
  // Baca audio stereo
  if (!readStereoAudio()) {
    Serial.println("⚠️ Failed to read audio, skipping classification");
    return;
  }
  
  float ambulance_prob_left = 0.0f;
  float ambulance_prob_right = 0.0f;
  float traffic_prob_left = 0.0f;
  float traffic_prob_right = 0.0f;
  
  // Proses mikrofon kiri
  extractFeatures(audio_buffer_left, features_left);
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features_left[i];
  }
  
  unsigned long start = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inference_time_left = millis() - start;
  
  if (invoke_status == kTfLiteOk) {
    ambulance_prob_left = output->data.f[0];
    traffic_prob_left = output->data.f[1];
    totalInferenceTime += inference_time_left;
    inferenceCount++;
  } else {
    Serial.println("❌ Inference failed for left mic");
  }
  
  // Proses mikrofon kanan
  extractFeatures(audio_buffer_right, features_right);
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features_right[i];
  }
  
  start = millis();
  invoke_status = interpreter->Invoke();
  unsigned long inference_time_right = millis() - start;
  
  if (invoke_status == kTfLiteOk) {
    ambulance_prob_right = output->data.f[0];
    traffic_prob_right = output->data.f[1];
    totalInferenceTime += inference_time_right;
    inferenceCount++;
  } else {
    Serial.println("❌ Inference failed for right mic");
  }
  
  // Voting mechanism: rata-rata probabilitas dari 2 mikrofon
  float avg_ambulance_prob = (ambulance_prob_left + ambulance_prob_right) / 2.0f;
  float avg_traffic_prob = (traffic_prob_left + traffic_prob_right) / 2.0f;
  
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  Serial.printf("🎤 LEFT  Mic - Ambulance: %.3f | Traffic: %.3f | %lums\n", 
                ambulance_prob_left, traffic_prob_left, inference_time_left);
  Serial.printf("🎤 RIGHT Mic - Ambulance: %.3f | Traffic: %.3f | %lums\n", 
                ambulance_prob_right, traffic_prob_right, inference_time_right);
  Serial.printf("📊 AVERAGE   - Ambulance: %.3f | Traffic: %.3f\n", 
                avg_ambulance_prob, avg_traffic_prob);
  
  // Threshold untuk deteksi (lebih konservatif dengan voting)
  float threshold = 0.65f;
  
  // Deteksi ambulance jika:
  // 1. Rata-rata probabilitas > threshold
  // 2. Ambulance prob > traffic prob
  // 3. Minimal 1 mikrofon detect dengan confidence tinggi (> 0.75)
  bool high_confidence = (ambulance_prob_left > 0.75f) || (ambulance_prob_right > 0.75f);
  
  if (avg_ambulance_prob > threshold && avg_ambulance_prob > avg_traffic_prob && high_confidence) {
    ambulanceDetectionCount++;
    Serial.println("✅ VERDICT: AMBULANCE DETECTED!");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("\n🚨🚨🚨 AMBULANCE EMERGENCY! 🚨🚨🚨");
    Serial.println("🚑 Emergency mode activated");
    Serial.println("🟢 Switching ALL lights to GREEN\n");
    
    setAllTrafficLights(LOW, LOW, HIGH); // Semua hijau
    currentState = AMBULANCE_EMERGENCY;
    lastStateChange = millis();
  } else {
    Serial.println("❌ VERDICT: Normal Traffic");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  }
}

void controlNormalTraffic() {
  static int currentPhase = 0; // 0=merah, 1=hijau, 2=kuning
  unsigned long currentTime = millis();
  unsigned long elapsed = currentTime - lastStateChange;

  switch (currentPhase) {
    case 0: // FASE MERAH
      if (elapsed < 100) {
        setAllTrafficLights(HIGH, LOW, LOW); // Semua merah
        Serial.println("🚦 Traffic Cycle: RED phase (Stop - 15s)");
      }
      if (elapsed >= RED_DURATION) {
        currentPhase = 1;
        lastStateChange = currentTime;
      }
      break;

    case 1: // FASE HIJAU
      if (elapsed < 100) {
        setAllTrafficLights(LOW, LOW, HIGH); // Semua hijau
        Serial.println("🚦 Traffic Cycle: GREEN phase (Go - 20s)");
      }
      if (elapsed >= GREEN_DURATION) {
        currentPhase = 2;
        lastStateChange = currentTime;
      }
      break;

    case 2: // FASE KUNING
      if (elapsed < 100) {
        setAllTrafficLights(LOW, HIGH, LOW); // Semua kuning
        Serial.println("🚦 Traffic Cycle: YELLOW phase (Caution - 5s)");
      }
      if (elapsed >= YELLOW_DURATION) {
        currentPhase = 0;
        lastStateChange = currentTime;
      }
      break;
  }
}

// void setup() {
//   Serial.begin(115200);
//   delay(2000);
  
//   Serial.println("\n\n");
//   Serial.println("╔════════════════════════════════════════════════════╗");
//   Serial.println("║  🚦 ESP32-S3 TINML TRAFFIC SYSTEM v2.0 🚦           ║");
//   Serial.println("║     Dual Microphone Ambulance Detection            ║");
//   Serial.println("║                                                    ║");
//   Serial.println("╚════════════════════════════════════════════════════╝");
//   Serial.println();

//   // Setup pin untuk 3 modul lampu lalu lintas
//   Serial.println("🔧 Initializing 3 Traffic Light Modules...");
  
//   // Modul 1
//   pinMode(RED_1, OUTPUT);
//   pinMode(YELLOW_1, OUTPUT);
//   pinMode(GREEN_1, OUTPUT);
  
//   // Modul 2
//   pinMode(RED_2, OUTPUT);
//   pinMode(YELLOW_2, OUTPUT);
//   pinMode(GREEN_2, OUTPUT);
  
//   // Modul 3
//   pinMode(RED_3, OUTPUT);
//   pinMode(YELLOW_3, OUTPUT);
//   pinMode(GREEN_3, OUTPUT);

//   // Test semua modul
//   Serial.println("🔧 Testing all modules...");
//   blinkAllYellow(2);
//   Serial.println("✅ All modules operational");

//   // Inisialisasi I2S Stereo
//   initI2S();
//   delay(500);

//   // Load TensorFlow Lite Model
//   Serial.println("\n📦 Loading TensorFlow Lite model...");
  
//   model = tflite::GetModel(model_ambulance_siren_tflite);
//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     Serial.printf("❌ Model version mismatch!\n");
//     Serial.printf("   Expected: %d, Got: %d\n", TFLITE_SCHEMA_VERSION, model->version());
//     setErrorState();
//   }
//   Serial.println("✅ Model schema validated");

//   // Setup operations resolver
//   static tflite::MicroMutableOpResolver<10> resolver;
//   resolver.AddFullyConnected();
//   resolver.AddSoftmax();
//   resolver.AddReshape();
//   resolver.AddQuantize();
//   resolver.AddDequantize();
  
//   Serial.println("✅ Operations resolver configured");

//   // Build interpreter
//   static tflite::MicroInterpreter static_interpreter(
//       model, resolver, tensor_arena, kTensorArenaSize);
//   interpreter = &static_interpreter;

//   // Allocate tensors
//   // TfLiteStatus allocate_status = interpreter->AllocateTensors();
//   // if (allocate_status != kTfLiteOk) {
//   //   Serial.println("❌ AllocateTensors failed!");
//   //   setErrorState();
//   // }
//   // Serial.println("✅ Tensors allocated");
//   TfLiteStatus allocate_status = interpreter->AllocateTensors();
//   if (allocate_status != kTfLiteOk) {
//   Serial.println("⚠️ Warning: AllocateTensors failed!");
//   Serial.println("   → Model inference disabled, running in NORMAL TRAFFIC MODE.");
  
//   // Tandai bahwa model tidak aktif
//   model = nullptr; 
//   } else {
//   Serial.println("✅ Tensors allocated successfully");
//   }

//   // Get input/output tensors
//   input = interpreter->input(0);
//   output = interpreter->output(0);

//   // Validasi dimensi
//   Serial.printf("📊 Model Configuration:\n");
//   Serial.printf("   Input shape: [");
//   for (int i = 0; i < input->dims->size; i++) {
//     Serial.printf("%d", input->dims->data[i]);
//     if (i < input->dims->size - 1) Serial.printf(", ");
//   }
//   Serial.printf("]\n");
  
//   Serial.printf("   Output shape: [");
//   for (int i = 0; i < output->dims->size; i++) {
//     Serial.printf("%d", output->dims->data[i]);
//     if (i < output->dims->size - 1) Serial.printf(", ");
//   }
//   Serial.printf("]\n");
  
//   Serial.printf("   Arena used: %d / %d bytes (%.1f%%)\n", 
//                 interpreter->arena_used_bytes(), 
//                 kTensorArenaSize,
//                 100.0f * interpreter->arena_used_bytes() / kTensorArenaSize);

//   Serial.println("\n📋 System Configuration:");
//   Serial.println("   🔴 RED Duration    : 15 seconds");
//   Serial.println("   🟢 GREEN Duration  : 20 seconds");
//   Serial.println("   🟡 YELLOW Duration : 5 seconds");
//   Serial.println("   🚨 Emergency Mode  : 30 seconds");
//   Serial.println("   🎤 Microphones     : 2 (Stereo)");
//   Serial.println("   🔊 Audio Check     : Every 1.5 seconds");
//   Serial.println("   🚦 Traffic Modules : 3 (Synchronized)");
//   Serial.println("   🎯 Detection Logic : Voting mechanism");
//   Serial.println();

//   // Mulai dengan semua lampu merah
//   setAllTrafficLights(HIGH, LOW, LOW);
//   lastStateChange = millis();
//   lastAudioCheck = millis();

//   Serial.println("╔════════════════════════════════════════════════════╗");
//   Serial.println("║  ✅ SYSTEM READY - AI Model Loaded              ║");
//   Serial.println("╚════════════════════════════════════════════════════╝");
//   Serial.println();
// }
void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n\n");
  Serial.println("╔════════════════════════════════════════════════════╗");
  Serial.println("║  🚦 ESP32-S3 TINML TRAFFIC SYSTEM v2.0 🚦           ║");
  Serial.println("║     Dual Microphone Ambulance Detection            ║");
  Serial.println("║                                                    ║");
  Serial.println("╚════════════════════════════════════════════════════╝");
  Serial.println();

  // Setup pin untuk 3 modul lampu lalu lintas
  Serial.println("🔧 Initializing 3 Traffic Light Modules...");
  
  // Modul 1
  pinMode(RED_1, OUTPUT);
  pinMode(YELLOW_1, OUTPUT);
  pinMode(GREEN_1, OUTPUT);
  
  // Modul 2
  pinMode(RED_2, OUTPUT);
  pinMode(YELLOW_2, OUTPUT);
  pinMode(GREEN_2, OUTPUT);
  
  // Modul 3
  pinMode(RED_3, OUTPUT);
  pinMode(YELLOW_3, OUTPUT);
  pinMode(GREEN_3, OUTPUT);

  // Test semua modul
  Serial.println("🔧 Testing all modules...");
  blinkAllYellow(2);
  Serial.println("✅ All modules operational");

  // Inisialisasi I2S Stereo
  initI2S();
  delay(500);

  // Load TensorFlow Lite Model
  Serial.println("\n📦 Loading TensorFlow Lite model...");
  
  model = tflite::GetModel(model_ambulance_siren_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("❌ Model version mismatch!\n");
    Serial.printf("   Expected: %d, Got: %d\n", TFLITE_SCHEMA_VERSION, model->version());
    setErrorState();
  }
  Serial.println("✅ Model schema validated");

  // Setup operations resolver
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  
  Serial.println("✅ Operations resolver configured");

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate tensors (MODIFIED)
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("⚠️ Warning: AllocateTensors failed!");
    Serial.println("   → Model inference disabled, running in NORMAL TRAFFIC MODE.");
    
    // 🔧 Modified: Jangan hentikan sistem, hanya nonaktifkan model
    model = nullptr;
  } else {
    Serial.println("✅ Tensors allocated successfully");
  }

  // Get input/output tensors hanya jika model valid
  if (model != nullptr) {
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Validasi dimensi
    Serial.printf("📊 Model Configuration:\n");
    Serial.printf("   Input shape: [");
    for (int i = 0; i < input->dims->size; i++) {
      Serial.printf("%d", input->dims->data[i]);
      if (i < input->dims->size - 1) Serial.printf(", ");
    }
    Serial.printf("]\n");
    
    Serial.printf("   Output shape: [");
    for (int i = 0; i < output->dims->size; i++) {
      Serial.printf("%d", output->dims->data[i]);
      if (i < output->dims->size - 1) Serial.printf(", ");
    }
    Serial.printf("]\n");
    
    Serial.printf("   Arena used: %d / %d bytes (%.1f%%)\n", 
                  interpreter->arena_used_bytes(), 
                  kTensorArenaSize,
                  100.0f * interpreter->arena_used_bytes() / kTensorArenaSize);
  } else {
    Serial.println("⚠️ Skipping model tensor validation (model inactive)");
  }

  Serial.println("\n📋 System Configuration:");
  Serial.println("   🔴 RED Duration    : 15 seconds");
  Serial.println("   🟢 GREEN Duration  : 20 seconds");
  Serial.println("   🟡 YELLOW Duration : 5 seconds");
  Serial.println("   🚨 Emergency Mode  : 30 seconds");
  Serial.println("   🎤 Microphones     : 2 (Stereo)");
  Serial.println("   🔊 Audio Check     : Every 1.5 seconds");
  Serial.println("   🚦 Traffic Modules : 3 (Synchronized)");
  Serial.println("   🎯 Detection Logic : Voting mechanism");
  Serial.println();

  // Mulai dengan semua lampu merah
  setAllTrafficLights(HIGH, LOW, LOW);
  lastStateChange = millis();
  lastAudioCheck = millis();

  Serial.println("╔════════════════════════════════════════════════════╗");
  if (model != nullptr) {
    Serial.println("║  ✅ SYSTEM READY - AI Model Loaded              ║");
  } else {
    Serial.println("║  ⚠️ SYSTEM READY - AI Model Disabled (Fallback) ║");
  }
  Serial.println("╚════════════════════════════════════════════════════╝");
  Serial.println();
}

// void loop() {
//   unsigned long now = millis();

//   switch (currentState) {
//     case NORMAL_OPERATION:
//       controlNormalTraffic();
      
//       // Check audio secara non-blocking
//       if (now - lastAudioCheck >= AUDIO_CHECK_INTERVAL) {
//         lastAudioCheck = now;
//         classifyAudio();
//       }
//       break;

//     case AMBULANCE_EMERGENCY:
//       // Tetap hijau selama emergency
//       if (now - lastStateChange >= EMERGENCY_TIME) {
//         Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
//         Serial.println("🔄 Emergency ended - Returning to normal");
//         Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
//         currentState = NORMAL_OPERATION;
//         lastStateChange = now;
//         setAllTrafficLights(HIGH, LOW, LOW); // Kembali ke merah
//       }
//       break;

//     case SYSTEM_ERROR:
//       // Handled by setErrorState()
//       break;
//   }

//   // Print statistik setiap 1 menit
//   static unsigned long lastStatusPrint = 0;
//   if (now - lastStatusPrint >= 60000) {
//     lastStatusPrint = now;
    
//     Serial.println("\n╔════════════════════════════════════════════════════╗");
//     Serial.println("║           📊 SYSTEM STATISTICS                       ║");
//     Serial.println("╠════════════════════════════════════════════════════╣");
    
//     if (inferenceCount > 0) {
//       float avgTime = (float)totalInferenceTime / inferenceCount;
//       Serial.printf("║  Audio Samples      : %-27d║\n", inferenceCount);
//       Serial.printf("║  Ambulance Detected : %-27d║\n", ambulanceDetectionCount);
//       Serial.printf("║  Avg Inference Time : %-23.2f ms ║\n", avgTime);
//       Serial.printf("║  Detection Rate     : %-23.1f %% ║\n", 
//                     100.0f * ambulanceDetectionCount / inferenceCount);
//     }
    
//     unsigned long uptime = millis() / 1000;
//     Serial.printf("║  System Uptime      : %-23lu sec ║\n", uptime);
//     Serial.printf("║  Free Heap          : %-23lu KB  ║\n", ESP.getFreeHeap() / 1024);
//     Serial.println("╚════════════════════════════════════════════════════╝\n");
//   }
  
//   // Yield untuk watchdog
//   delay(10);
// }
void loop() {
  unsigned long now = millis();

  switch (currentState) {
    case NORMAL_OPERATION:
      controlNormalTraffic();
      
      // Check audio secara non-blocking
      if (now - lastAudioCheck >= AUDIO_CHECK_INTERVAL) {
        lastAudioCheck = now;

        // 🔧 Tambahan: Cek apakah model aktif
        if (model != nullptr) {
          classifyAudio();
        } else {
          Serial.println("⚠️ Model inactive - skipping audio classification");
        }
      }
      break;

    case AMBULANCE_EMERGENCY:
      // Tetap hijau selama emergency
      if (now - lastStateChange >= EMERGENCY_TIME) {
        Serial.println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        Serial.println("🔄 Emergency ended - Returning to normal");
        Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        currentState = NORMAL_OPERATION;
        lastStateChange = now;
        setAllTrafficLights(HIGH, LOW, LOW); // Kembali ke merah
      }
      break;

    case SYSTEM_ERROR:
      // Handled by setErrorState()
      break;
  }

  // Print statistik setiap 1 menit
  static unsigned long lastStatusPrint = 0;
  if (now - lastStatusPrint >= 60000) {
    lastStatusPrint = now;
    
    Serial.println("\n╔════════════════════════════════════════════════════╗");
    Serial.println("║           📊 SYSTEM STATISTICS                       ║");
    Serial.println("╠════════════════════════════════════════════════════╣");
    
    if (inferenceCount > 0) {
      float avgTime = (float)totalInferenceTime / inferenceCount;
      Serial.printf("║  Audio Samples      : %-27d║\n", inferenceCount);
      Serial.printf("║  Ambulance Detected : %-27d║\n", ambulanceDetectionCount);
      Serial.printf("║  Avg Inference Time : %-23.2f ms ║\n", avgTime);
      Serial.printf("║  Detection Rate     : %-23.1f %% ║\n", 
                    100.0f * ambulanceDetectionCount / inferenceCount);
    } else {
      Serial.println("║  AI Model Status    : Inactive or Not Loaded       ║");
    }
    
    unsigned long uptime = millis() / 1000;
    Serial.printf("║  System Uptime      : %-23lu sec ║\n", uptime);
    Serial.printf("║  Free Heap          : %-23lu KB  ║\n", ESP.getFreeHeap() / 1024);
    Serial.println("╚════════════════════════════════════════════════════╝\n");
  }
  
  // Yield untuk watchdog
  delay(10);
}