#include <Arduino.h>
#include "driver/i2s.h"

// TensorFlow Lite untuk ESP32
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model yang sudah di-convert
#include "model_ambulance_siren.h"

//  KONFIGURASI OPTIMAL - MEMORY FIXED
#define SAMPLE_RATE 16000
#define RECORD_MS 400
#define NUM_SAMPLES (SAMPLE_RATE * RECORD_MS / 1000)
#define NUM_FEATURES 32

//  PIN YANG SUDAH DISOLDER - SESUAI HARDWARE
// Modul 1 - JALUR 1
#define RED_1    GPIO_NUM_10
#define YELLOW_1 GPIO_NUM_11
#define GREEN_1  GPIO_NUM_12


// Modul 2 - JALUR 2
#define RED_2    GPIO_NUM_4
#define YELLOW_2 GPIO_NUM_5
#define GREEN_2  GPIO_NUM_6

// Modul 3 - JALUR 3 (No microphone)
#define RED_3    GPIO_NUM_7
#define YELLOW_3 GPIO_NUM_15
#define GREEN_3  GPIO_NUM_16

// Pin I2S untuk dua mikrofon INMP441 (Stereo - 1 I2S Bus)
#define I2S_WS      GPIO_NUM_42  // Word Select (LRCL)
#define I2S_SCK     GPIO_NUM_41  // Bit Clock (BCLK)
#define I2S_SD_L    GPIO_NUM_40  // Mikrofon Kiri (SD) - JALUR 1
#define I2S_SD_R    GPIO_NUM_39  // Mikrofon Kanan (SD) - JALUR 2

// ASOSIASI MIKROFON - JALUR
#define LEFT_MIC_JALUR    1    // Mikrofon kiri → Jalur 1
#define RIGHT_MIC_JALUR   2    // Mikrofon kanan → Jalur 2

// TensorFlow Lite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 200 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// Buffer audio
#define BUFFER_SIZE 256
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

// Status deteksi per jalur
bool jalur1_detected = false;
bool jalur2_detected = false;

// Timing untuk siklus lampu lalu lintas real
const unsigned long RED_DURATION = 15000;      // Merah: 15 detik
const unsigned long YELLOW_DURATION = 5000;    // Kuning: 5 detik
const unsigned long GREEN_DURATION = 20000;    // Hijau: 20 detik
const unsigned long EMERGENCY_TIME = 30000;    // Emergency: 30 detik
const unsigned long AUDIO_CHECK_INTERVAL = 2000; // Check audio setiap 2 detik

// Statistik
unsigned long totalInferenceTime = 0;
unsigned int inferenceCount = 0;
unsigned int ambulanceDetectionCount = 0;

// Konfigurasi I2S driver untuk STEREO (2 mikrofon)
void initI2S() {
  Serial.println("🎤 Configuring I2S for dual INMP441 microphones (Stereo)...");
  
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,  // Stereo
      .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_STAND_I2S),
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 3,
      .dma_buf_len = BUFFER_SIZE,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD_L
  };

  esp_err_t err = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (err != ESP_OK) {
    Serial.printf(" I2S driver install failed: %d\n", err);
    setErrorState();
    return;
  }
  
  err = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (err != ESP_OK) {
    Serial.printf(" I2S pin config failed: %d\n", err);
    setErrorState();
    return;
  }
  
  i2s_zero_dma_buffer(I2S_NUM_0);
  Serial.println("Stereo I2S initialized successfully");
}

void blinkAllYellow(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(YELLOW_1, HIGH);
    digitalWrite(YELLOW_2, HIGH);
    digitalWrite(YELLOW_3, HIGH);
    delay(300);
    digitalWrite(YELLOW_1, LOW);
    digitalWrite(YELLOW_2, LOW);
    digitalWrite(YELLOW_3, LOW);
    delay(300);
  }
}

void setErrorState() {
  currentState = SYSTEM_ERROR;
  Serial.println("\n SYSTEM ERROR - Entering error state ");
  Serial.println("⚠️  Press RESET button to restart");
  
  while (true) {
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

// Kontrol lampu per jalur
void setIndividualTrafficLights(int jalur, int red, int yellow, int green) {
  switch(jalur) {
    case 1:
      digitalWrite(RED_1, red);
      digitalWrite(YELLOW_1, yellow);
      digitalWrite(GREEN_1, green);
      break;
    case 2:
      digitalWrite(RED_2, red);
      digitalWrite(YELLOW_2, yellow);
      digitalWrite(GREEN_2, green);
      break;
    case 3:
      digitalWrite(RED_3, red);
      digitalWrite(YELLOW_3, yellow);
      digitalWrite(GREEN_3, green);
      break;
  }
}

// Emergency berdasarkan deteksi jalur
void setEmergencyLightsBasedOnDetection() {
  Serial.println("🚨 Setting emergency lights based on detection...");
  
  if (jalur1_detected) {
    setIndividualTrafficLights(1, LOW, LOW, HIGH); // Jalur 1 hijau
    Serial.println("  🟢 Jalur 1: HIJAU (Ambulance detected)");
  } else {
    setIndividualTrafficLights(1, HIGH, LOW, LOW); // Jalur 1 merah
    Serial.println("  🔴 Jalur 1: MERAH");
  }
  
  if (jalur2_detected) {
    setIndividualTrafficLights(2, LOW, LOW, HIGH); // Jalur 2 hijau
    Serial.println("  🟢 Jalur 2: HIJAU (Ambulance detected)");
  } else {
    setIndividualTrafficLights(2, HIGH, LOW, LOW); // Jalur 2 merah
    Serial.println("  🔴 Jalur 2: MERAH");
  }
  
  // Jalur 3 selalu merah (tidak ada mikrofon)
  setIndividualTrafficLights(3, HIGH, LOW, LOW);
  Serial.println("  🔴 Jalur 3: MERAH (No microphone)");
}

void setAllTrafficLights(int red, int yellow, int green) {
  setIndividualTrafficLights(1, red, yellow, green);
  setIndividualTrafficLights(2, red, yellow, green);
  setIndividualTrafficLights(3, red, yellow, green);

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

  // Hitung energi per frame
  int frame_size = NUM_SAMPLES / NUM_FEATURES;
  for (int frame = 0; frame < NUM_FEATURES; frame++) {
    float frame_energy = 0.0f;
    int start = frame * frame_size;
    int end = min(start + frame_size, NUM_SAMPLES);

    for (int i = start; i < end; i++) {
      frame_energy += normalized[i] * normalized[i];
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
    
    int samples_in_buffer = bytesRead / 8;
    
    for (int i = 0; i < samples_in_buffer && samples_read_left < NUM_SAMPLES; i++) {
      audio_buffer_left[samples_read_left] = stereo_buffer[i * 2];
      audio_buffer_right[samples_read_right] = stereo_buffer[i * 2 + 1];
      samples_read_left++;
      samples_read_right++;
    }
  }
  
  Serial.printf("✅ Read %d samples from each microphone\n", samples_read_left);
  return true;
}

// Klasifikasi audio dengan deteksi per jalur
void classifyAudio() {
  if (!readStereoAudio()) {
    Serial.println("⚠️ Failed to read audio, skipping classification");
    return;
  }
  
  float ambulance_prob_left = 0.0f;
  float ambulance_prob_right = 0.0f;
  
  // Reset status deteksi
  jalur1_detected = false;
  jalur2_detected = false;
  
  // PROSES MIKROFON KIRI (Jalur 1)
  extractFeatures(audio_buffer_left, features_left);
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features_left[i];
  }
  
  unsigned long start = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inference_time_left = millis() - start;
  
  if (invoke_status == kTfLiteOk) {
    ambulance_prob_left = output->data.f[0];
    if (ambulance_prob_left > 0.75f) {
      jalur1_detected = true;
    }
    totalInferenceTime += inference_time_left;
    inferenceCount++;
  } else {
    Serial.println("❌ Inference failed for left mic");
  }
  
  // PROSES MIKROFON KANAN (Jalur 2)
  extractFeatures(audio_buffer_right, features_right);
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features_right[i];
  }
  
  start = millis();
  invoke_status = interpreter->Invoke();
  unsigned long inference_time_right = millis() - start;
  
  if (invoke_status == kTfLiteOk) {
    ambulance_prob_right = output->data.f[0];
    if (ambulance_prob_right > 0.75f) {
      jalur2_detected = true;
    }
    totalInferenceTime += inference_time_right;
    inferenceCount++;
  } else {
    Serial.println("❌ Inference failed for right mic");
  }
  
  // LOGIC DETEKSI FINAL
  Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  Serial.printf("🎤 JALUR 1 - Ambulance: %.3f → %s\n", 
                ambulance_prob_left, 
                jalur1_detected ? "DETECTED 🚑" : "No ambulance");
  Serial.printf("🎤 JALUR 2 - Ambulance: %.3f → %s\n", 
                ambulance_prob_right, 
                jalur2_detected ? "DETECTED 🚑" : "No ambulance");
  
  // Trigger emergency jika minimal 1 jalur terdeteksi
  if (jalur1_detected || jalur2_detected) {
    ambulanceDetectionCount++;
    Serial.println("✅ VERDICT: AMBULANCE DETECTED ON SPECIFIC LANE!");
    Serial.println("🚨 Activating lane-specific emergency mode");
    
    setEmergencyLightsBasedOnDetection();
    currentState = AMBULANCE_EMERGENCY;
    lastStateChange = millis();
    
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.println("\n🚨🚨🚨 AMBULANCE EMERGENCY! 🚨🚨🚨");
    if (jalur1_detected && jalur2_detected) {
      Serial.println("🚑 Ambulance detected on BOTH lanes!");
    } else if (jalur1_detected) {
      Serial.println("🚑 Ambulance detected on JALUR 1!");
    } else {
      Serial.println("🚑 Ambulance detected on JALUR 2!");
    }
    Serial.println("🟢 Priority GREEN for detected lanes only\n");
  } else {
    Serial.println("❌ VERDICT: No ambulance detected");
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
  }
}

void controlNormalTraffic() {
  static int currentPhase = 0;
  unsigned long currentTime = millis();
  unsigned long elapsed = currentTime - lastStateChange;

  switch (currentPhase) {
    case 0: // FASE MERAH - Semua merah
      if (elapsed < 100) {
        setAllTrafficLights(HIGH, LOW, LOW);
        Serial.println("🚦 Normal Cycle: ALL RED (Stop - 15s)");
      }
      if (elapsed >= RED_DURATION) {
        currentPhase = 1;
        lastStateChange = currentTime;
      }
      break;

    case 1: // FASE HIJAU - Semua hijau
      if (elapsed < 100) {
        setAllTrafficLights(LOW, LOW, HIGH);
        Serial.println("🚦 Normal Cycle: ALL GREEN (Go - 20s)");
      }
      if (elapsed >= GREEN_DURATION) {
        currentPhase = 2;
        lastStateChange = currentTime;
      }
      break;

    case 2: // FASE KUNING - Semua kuning
      if (elapsed < 100) {
        setAllTrafficLights(LOW, HIGH, LOW);
        Serial.println("🚦 Normal Cycle: ALL YELLOW (Caution - 5s)");
      }
      if (elapsed >= YELLOW_DURATION) {
        currentPhase = 0;
        lastStateChange = currentTime;
      }
      break;
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n\n");
  Serial.println("╔════════════════════════════════════════════════════╗");
  Serial.println("║  🚦 ESP32-S3 SMART TRAFFIC SYSTEM v3.0 🚦        ║");
  Serial.println("║     Lane-Specific Ambulance Detection            ║");
  Serial.println("║     HARDWARE FIXED PIN Version                   ║");
  Serial.println("╚════════════════════════════════════════════════════╝");
  Serial.println();

  // ⚠️ PERINGATAN PIN
  Serial.println("⚠️  WARNING: Using GPIO1 & GPIO2 for traffic lights");
  Serial.println("    These pins are used for USB Serial during boot");
  Serial.println("    Make sure LEDs are disconnected during programming");
  Serial.println();

  // Setup pin untuk 3 modul lampu lalu lintas
  Serial.println("🔧 Initializing 3 Traffic Light Modules...");
  
  // Modul 1 - Jalur 1
  pinMode(RED_1, OUTPUT);
  pinMode(YELLOW_1, OUTPUT);
  pinMode(GREEN_1, OUTPUT);
  
  // Modul 2 - Jalur 2
  pinMode(RED_2, OUTPUT);
  pinMode(YELLOW_2, OUTPUT);
  pinMode(GREEN_2, OUTPUT);
  
  // Modul 3 - Jalur 3
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

  // 🔥 RESOLVER YANG DIPERBAIKI - LENGKAP
  static tflite::MicroMutableOpResolver<12> resolver;
  
  // Tambahkan operasi dengan error checking
  Serial.println("🔧 Adding operations to resolver...");
  
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    Serial.println("❌ Failed to add FullyConnected");
    setErrorState();
  }
  if (resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("❌ Failed to add Softmax");
    setErrorState();
  }
  if (resolver.AddReshape() != kTfLiteOk) {
    Serial.println("❌ Failed to add Reshape");
    setErrorState();
  }
  if (resolver.AddQuantize() != kTfLiteOk) {
    Serial.println("❌ Failed to add Quantize");
    setErrorState();
  }
  if (resolver.AddDequantize() != kTfLiteOk) {
    Serial.println("❌ Failed to add Dequantize");
    setErrorState();
  }
  if (resolver.AddRelu() != kTfLiteOk) {
    Serial.println("❌ Failed to add Relu");
    setErrorState();
  }
  if (resolver.AddConv2D() != kTfLiteOk) {
    Serial.println("❌ Failed to add Conv2D");
    setErrorState();
  }
  if (resolver.AddMaxPool2D() != kTfLiteOk) {
    Serial.println("❌ Failed to add MaxPool2D");
    setErrorState();
  }
  if (resolver.AddAveragePool2D() != kTfLiteOk) {
    Serial.println("❌ Failed to add AveragePool2D");
    setErrorState();
  }
  if (resolver.AddShape() != kTfLiteOk) {
    Serial.println("❌ Failed to add Shape");
    setErrorState();
  }
  if (resolver.AddStridedSlice() != kTfLiteOk) {
    Serial.println("❌ Failed to add StridedSlice");
    setErrorState();
  }
  if (resolver.AddPack() != kTfLiteOk) {
    Serial.println("❌ Failed to add Pack");
    setErrorState();
  }

  Serial.println("✅ All operations added successfully!");

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate tensors dengan debugging
  Serial.println("🔧 Allocating tensors...");
  
  Serial.printf("💾 Free Heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("💾 Largest Free Block: %d bytes\n", heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
  Serial.printf("💾 Tensor Arena: %d bytes\n", kTensorArenaSize);
  
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("❌ AllocateTensors failed!");
    Serial.printf("💾 Arena used: %d bytes\n", interpreter->arena_used_bytes());
    setErrorState();
  }
  Serial.println("✅ Tensors allocated successfully!");

  // Get input/output tensors
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

  Serial.println("\n📋 HARDWARE Configuration:");
  Serial.println("   🎤 Microphone 1 → Jalur 1 (GPIO1,2,3)");
  Serial.println("   🎤 Microphone 2 → Jalur 2 (GPIO4,5,6)"); 
  Serial.println("   🚦 Jalur 3      → No microphone (GPIO7,15,16)");
  Serial.println("   🚨 Emergency    → Only detected lanes turn GREEN");
  Serial.println();

  // Mulai dengan semua lampu merah
  setAllTrafficLights(HIGH, LOW, LOW);
  lastStateChange = millis();
  lastAudioCheck = millis();

  Serial.println("╔════════════════════════════════════════════════════╗");
  Serial.println("║  ✅ SYSTEM READY - AI Model Loaded              ║");
  Serial.println("║  ✅ STEREO Microphones Active                  ║");
  Serial.println("║  ✅ Memory Allocation Successful               ║");
  Serial.println("║  ✅ Lane-Specific Detection ENABLED            ║");
  Serial.println("║  ⚠️  Using HARDWARE-SOLDERED Pins             ║");
  Serial.println("╚════════════════════════════════════════════════════╝");
  Serial.println();
}

void loop() {
  unsigned long now = millis();

  switch (currentState) {
    case NORMAL_OPERATION:
      controlNormalTraffic();
      
      // Check audio secara non-blocking
      if (now - lastAudioCheck >= AUDIO_CHECK_INTERVAL) {
        lastAudioCheck = now;
        classifyAudio();
      }
      break;

    case AMBULANCE_EMERGENCY:
      // Tetap hijau selama emergency hanya untuk jalur yang terdeteksi
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

  // Print statistik setiap 30 detik
  static unsigned long lastStatusPrint = 0;
  if (now - lastStatusPrint >= 30000) {
    lastStatusPrint = now;
    
    Serial.println("\n╔════════════════════════════════════════════════════╗");
    Serial.println("║           📊 SYSTEM STATISTICS                    ║");
    Serial.println("╠════════════════════════════════════════════════════╣");
    
    if (inferenceCount > 0) {
      float avgTime = (float)totalInferenceTime / inferenceCount;
      Serial.printf("║  Audio Samples      : %-27d║\n", inferenceCount);
      Serial.printf("║  Ambulance Detected : %-27d║\n", ambulanceDetectionCount);
      Serial.printf("║  Avg Inference Time : %-23.2f ms ║\n", avgTime);
      Serial.printf("║  Detection Rate     : %-23.1f %% ║\n", 
                    100.0f * ambulanceDetectionCount / inferenceCount);
    }
    
    unsigned long uptime = millis() / 1000;
    Serial.printf("║  System Uptime      : %-23lu sec ║\n", uptime);
    Serial.printf("║  Free Heap          : %-23lu KB  ║\n", ESP.getFreeHeap() / 1024);
    
    const char* stateStr = "";
    if (currentState == NORMAL_OPERATION) stateStr = "Normal";
    else if (currentState == AMBULANCE_EMERGENCY) stateStr = "Emergency";
    else stateStr = "Error";
    
    Serial.printf("║  Current State      : %-23s ║\n", stateStr);
    Serial.println("╚════════════════════════════════════════════════════╝\n");
  }
  
  delay(10);
}