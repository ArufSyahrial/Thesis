#include <Arduino.h>
#include "driver/i2s.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_ambulance_siren_int8.h"

// ==============================
//  KONFIGURASI SISTEM
// ==============================
#define SAMPLE_RATE 16000
#define RECORD_MS 200
#define NUM_SAMPLES (SAMPLE_RATE * RECORD_MS / 1000)
#define NUM_FEATURES 24
#define BUFFER_SIZE 256

// ==============================
//  PIN LAMPU LALU LINTAS
// ==============================
#define RED_1    GPIO_NUM_10
#define YELLOW_1 GPIO_NUM_11
#define GREEN_1  GPIO_NUM_12
#define RED_2    GPIO_NUM_4
#define YELLOW_2 GPIO_NUM_5
#define GREEN_2  GPIO_NUM_6
#define RED_3    GPIO_NUM_7
#define YELLOW_3 GPIO_NUM_15
#define GREEN_3  GPIO_NUM_16

// ==============================
//  PIN I2S - DUAL MIC
// ==============================
#define I2S_WS      GPIO_NUM_42
#define I2S_SCK     GPIO_NUM_41
#define I2S_SD      GPIO_NUM_40

// ==============================
//  GLOBAL OBJEK DAN VARIABEL
// ==============================
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 160 * 1024;
  uint8_t* tensor_arena = nullptr;
}

// Buffer audio
int32_t stereo_buffer[BUFFER_SIZE];
int32_t audio_buffer_left[NUM_SAMPLES];
int32_t audio_buffer_right[NUM_SAMPLES];
float features_left[NUM_FEATURES];
float features_right[NUM_FEATURES];

enum TrafficLightState {
  NORMAL_OPERATION,
  AMBULANCE_EMERGENCY_JALUR_1,
  AMBULANCE_EMERGENCY_JALUR_2,
  SYSTEM_ERROR
};
TrafficLightState currentState = NORMAL_OPERATION;

unsigned long lastStateChange = 0;
unsigned long lastAudioCheck = 0;
unsigned long totalInferenceTime = 0;
unsigned int inferenceCount = 0;
unsigned int ambulanceDetectionCount = 0;

// Threshold deteksi
const float DETECTION_THRESHOLD = 0.9f;
static int detection_count_jalur1 = 0;
static int detection_count_jalur2 = 0;
const int REQUIRED_CONSECUTIVE_DETECTIONS = 2;

// Timing untuk siklus normal
const unsigned long GREEN_DURATION = 15000;    // 15 detik hijau
const unsigned long YELLOW_DURATION = 3000;    // 3 detik kuning
const unsigned long ALL_RED_DURATION = 2000;   // 2 detik semua merah antar siklus
const unsigned long EMERGENCY_TIME = 20000;    // 20 detik emergency
const unsigned long AUDIO_CHECK_INTERVAL = 2000;

// Variabel untuk siklus normal
enum TrafficPhase {
  PHASE_JALUR1_GREEN,
  PHASE_JALUR1_YELLOW,
  PHASE_ALL_RED_1,
  PHASE_JALUR2_GREEN,
  PHASE_JALUR2_YELLOW,
  PHASE_ALL_RED_2,
  PHASE_JALUR3_GREEN,
  PHASE_JALUR3_YELLOW,
  PHASE_ALL_RED_3
};
TrafficPhase currentPhase = PHASE_JALUR1_GREEN;

// VARIABLE BARU: Menyimpan fase sebelum emergency
TrafficPhase phaseBeforeEmergency = PHASE_JALUR1_GREEN;

// ===========================
//  ERROR HANDLER
// ===========================
void setErrorState() {
  currentState = SYSTEM_ERROR;
  Serial.println("\n⚠️ SYSTEM ERROR - Entering safe state!");
  while (true) {
    digitalWrite(RED_1, HIGH);
    digitalWrite(RED_2, HIGH);
    digitalWrite(RED_3, HIGH);
    delay(400);
    digitalWrite(RED_1, LOW);
    digitalWrite(RED_2, LOW);
    digitalWrite(RED_3, LOW);
    delay(400);
  }
}

// ===========================
//  TEST MIKROFON
// ===========================
void testMicrophones() {
  Serial.println("\n🎯 TESTING DUAL MICROPHONES");
  
  for (int test = 1; test <= 2; test++) {
    Serial.printf("\n🧪 Test %d/2:\n", test);
    
    if (!readStereoAudio()) {
      Serial.println("❌ Failed to read audio");
      continue;
    }
    
    long left_sum = 0, right_sum = 0;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
      left_sum += abs(audio_buffer_left[i]);
      right_sum += abs(audio_buffer_right[i]);
    }
    
    float left_avg = (float)left_sum / NUM_SAMPLES;
    float right_avg = (float)right_sum / NUM_SAMPLES;
    
    Serial.println("📊 CHANNEL ANALYSIS:");
    Serial.printf("   JALUR 1: Avg=%8.1f ", left_avg);
    if (left_avg > 1000) Serial.println("✅ BERFUNGSI");
    else Serial.println("🔇 MASALAH");
    
    Serial.printf("   JALUR 2: Avg=%8.1f ", right_avg);
    if (right_avg > 1000) Serial.println("✅ BERFUNGSI");
    else Serial.println("🔇 MASALAH");
    
    delay(1000);
  }
}

// ===========================
//  INISIALISASI I2S
// ===========================
void initI2S() {
  Serial.println("🎤 Configuring I2S for DUAL INMP441 microphones...");
  
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = BUFFER_SIZE,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD
  };

  i2s_driver_uninstall(I2S_NUM_0); 
  if (i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL) != ESP_OK) {
    Serial.println("❌ I2S driver install failed");
    setErrorState();
  }
  
  if (i2s_set_pin(I2S_NUM_0, &pin_config) != ESP_OK) {
    Serial.println("❌ I2S pin config failed");
    setErrorState();
  }

  i2s_zero_dma_buffer(I2S_NUM_0);
  
  testMicrophones();
}

// ===========================
//  FUNGSI BANTU LAMPU
// ===========================
void setAllTrafficLights(int red, int yellow, int green) {
  digitalWrite(RED_1, red); digitalWrite(YELLOW_1, yellow); digitalWrite(GREEN_1, green);
  digitalWrite(RED_2, red); digitalWrite(YELLOW_2, yellow); digitalWrite(GREEN_2, green);
  digitalWrite(RED_3, red); digitalWrite(YELLOW_3, yellow); digitalWrite(GREEN_3, green);
}

void setAllRed() {
  setAllTrafficLights(HIGH, LOW, LOW);
}

void setAllYellow() {
  setAllTrafficLights(LOW, HIGH, LOW);
}

void setJalur1Green() {
  digitalWrite(RED_1, LOW); digitalWrite(YELLOW_1, LOW); digitalWrite(GREEN_1, HIGH);
  digitalWrite(RED_2, HIGH); digitalWrite(YELLOW_2, LOW); digitalWrite(GREEN_2, LOW);
  digitalWrite(RED_3, HIGH); digitalWrite(YELLOW_3, LOW); digitalWrite(GREEN_3, LOW);
}

void setJalur2Green() {
  digitalWrite(RED_1, HIGH); digitalWrite(YELLOW_1, LOW); digitalWrite(GREEN_1, LOW);
  digitalWrite(RED_2, LOW); digitalWrite(YELLOW_2, LOW); digitalWrite(GREEN_2, HIGH);
  digitalWrite(RED_3, HIGH); digitalWrite(YELLOW_3, LOW); digitalWrite(GREEN_3, LOW);
}

void setJalur3Green() {
  digitalWrite(RED_1, HIGH); digitalWrite(YELLOW_1, LOW); digitalWrite(GREEN_1, LOW);
  digitalWrite(RED_2, HIGH); digitalWrite(YELLOW_2, LOW); digitalWrite(GREEN_2, LOW);
  digitalWrite(RED_3, LOW); digitalWrite(YELLOW_3, LOW); digitalWrite(GREEN_3, HIGH);
}

void setJalur1Yellow() {
  digitalWrite(RED_1, LOW); digitalWrite(YELLOW_1, HIGH); digitalWrite(GREEN_1, LOW);
  digitalWrite(RED_2, HIGH); digitalWrite(YELLOW_2, LOW); digitalWrite(GREEN_2, LOW);
  digitalWrite(RED_3, HIGH); digitalWrite(YELLOW_3, LOW); digitalWrite(GREEN_3, LOW);
}

void setJalur2Yellow() {
  digitalWrite(RED_1, HIGH); digitalWrite(YELLOW_1, LOW); digitalWrite(GREEN_1, LOW);
  digitalWrite(RED_2, LOW); digitalWrite(YELLOW_2, HIGH); digitalWrite(GREEN_2, LOW);
  digitalWrite(RED_3, HIGH); digitalWrite(YELLOW_3, LOW); digitalWrite(GREEN_3, LOW);
}

void setJalur3Yellow() {
  digitalWrite(RED_1, HIGH); digitalWrite(YELLOW_1, LOW); digitalWrite(GREEN_1, LOW);
  digitalWrite(RED_2, HIGH); digitalWrite(YELLOW_2, LOW); digitalWrite(GREEN_2, LOW);
  digitalWrite(RED_3, LOW); digitalWrite(YELLOW_3, HIGH); digitalWrite(GREEN_3, LOW);
}

void setEmergencyLightsForJalur1() {
  Serial.println("🚨 Emergency mode - Ambulance detected in JALUR 1");
  setJalur1Green();
  Serial.println("  🟢 Jalur 1: HIJAU");
  Serial.println("  🔴 Jalur 2: MERAH");
  Serial.println("  🔴 Jalur 3: MERAH");
}

void setEmergencyLightsForJalur2() {
  Serial.println("🚨 Emergency mode - Ambulance detected in JALUR 2");
  setJalur2Green();
  Serial.println("  🔴 Jalur 1: MERAH");
  Serial.println("  🟢 Jalur 2: HIJAU");
  Serial.println("  🔴 Jalur 3: MERAH");
}

void blinkAllYellow(int times) {
  for (int i = 0; i < times; i++) {
    setAllYellow();
    delay(300);
    setAllRed();
    delay(300);
  }
}

// ===========================
//  FUNGSI EKSTRAK FITUR
// ===========================
void extractFeatures(int32_t* audio, float* features) {
  int32_t max_val = 1;
  for (int i = 0; i < NUM_SAMPLES; i++) {
    if (abs(audio[i]) > max_val) max_val = abs(audio[i]);
  }
  
  int frame_size = NUM_SAMPLES / NUM_FEATURES;
  
  for (int frame = 0; frame < NUM_FEATURES; frame++) {
    float frame_energy = 0.0f;
    int start = frame * frame_size;
    int end = (frame + 1) * frame_size;
    if (end > NUM_SAMPLES) end = NUM_SAMPLES;
    int frame_samples = end - start;
    
    for (int i = start; i < end; i++) {
      float normalized_sample = (float)audio[i] / max_val;
      frame_energy += normalized_sample * normalized_sample;
    }
    features[frame] = sqrtf(frame_energy / frame_samples);
  }
  
  float max_feature = 0.0f;
  for (int i = 0; i < NUM_FEATURES; i++) {
    if (features[i] > max_feature) max_feature = features[i];
  }
  
  if (max_feature > 0.0f) {
    for (int i = 0; i < NUM_FEATURES; i++) {
      features[i] /= max_feature;
    }
  }
}

// ===========================
//  BACA AUDIO STEREO
// ===========================
bool readStereoAudio() {
  int samples_read = 0;
  unsigned long start_time = millis();
  
  while (samples_read < NUM_SAMPLES && (millis() - start_time) < 1000) {
    size_t bytesRead = 0;
    esp_err_t result = i2s_read(I2S_NUM_0, stereo_buffer, sizeof(stereo_buffer), &bytesRead, pdMS_TO_TICKS(100));
    
    if (result != ESP_OK) {
      return false;
    }
    
    if (bytesRead == 0) {
      continue;
    }
    
    int samples_in_buffer = bytesRead / 8;
    
    for (int i = 0; i < samples_in_buffer && samples_read < NUM_SAMPLES; i++) {
      audio_buffer_left[samples_read] = stereo_buffer[i * 2];
      audio_buffer_right[samples_read] = stereo_buffer[i * 2 + 1];
      samples_read++;
    }
  }
  
  return samples_read >= NUM_SAMPLES;
}

// ===========================
//  KLASIFIKASI AUDIO - DUAL MIC
// ===========================
void classifyAudio() {
  if (!readStereoAudio()) {
    Serial.println("⚠️ Audio read failed");
    return;
  }
  
  float prob_jalur1 = 0.0f, prob_jalur2 = 0.0f;
  bool jalur1_detected = false;
  bool jalur2_detected = false;

  // PROSES MIKROFON JALUR 1
  extractFeatures(audio_buffer_left, features_left);
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features_left[i];
  }
  
  unsigned long start = micros();
  if (interpreter->Invoke() == kTfLiteOk) {
    prob_jalur1 = output->data.f[0];
    totalInferenceTime += (micros() - start);
    inferenceCount++;
  }

  // PROSES MIKROFON JALUR 2
  extractFeatures(audio_buffer_right, features_right);
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features_right[i];
  }
  
  start = micros();
  if (interpreter->Invoke() == kTfLiteOk) {
    prob_jalur2 = output->data.f[0];
    totalInferenceTime += (micros() - start);
    inferenceCount++;
  }

  // LOGIKA DETEKSI
  Serial.printf("🎤 J1:%.2f J2:%.2f", prob_jalur1, prob_jalur2);
  
  if (prob_jalur1 > DETECTION_THRESHOLD) {
    detection_count_jalur1++;
    Serial.printf(" ✅J1(%d)", detection_count_jalur1);
    
    if (detection_count_jalur1 >= REQUIRED_CONSECUTIVE_DETECTIONS) {
      jalur1_detected = true;
      detection_count_jalur1 = 0;
    }
  } else {
    detection_count_jalur1 = 0;
  }
  
  if (prob_jalur2 > DETECTION_THRESHOLD) {
    detection_count_jalur2++;
    Serial.printf(" ✅J2(%d)", detection_count_jalur2);
    
    if (detection_count_jalur2 >= REQUIRED_CONSECUTIVE_DETECTIONS) {
      jalur2_detected = true;
      detection_count_jalur2 = 0;
    }
  } else {
    detection_count_jalur2 = 0;
  }

  if (jalur1_detected && jalur2_detected) {
    Serial.println(" -> 🚨 AMBULANCE DI JALUR 1 & 2! (Prioritas J1)");
    // SIMPAN FASE SEBELUM EMERGENCY
    phaseBeforeEmergency = currentPhase;
    setEmergencyLightsForJalur1();
    currentState = AMBULANCE_EMERGENCY_JALUR_1;
    lastStateChange = millis();
    ambulanceDetectionCount++;
  } else if (jalur1_detected) {
    Serial.println(" -> 🚨 AMBULANCE DI JALUR 1!");
    // SIMPAN FASE SEBELUM EMERGENCY
    phaseBeforeEmergency = currentPhase;
    setEmergencyLightsForJalur1();
    currentState = AMBULANCE_EMERGENCY_JALUR_1;
    lastStateChange = millis();
    ambulanceDetectionCount++;
  } else if (jalur2_detected) {
    Serial.println(" -> 🚨 AMBULANCE DI JALUR 2!");
    // SIMPAN FASE SEBELUM EMERGENCY
    phaseBeforeEmergency = currentPhase;
    setEmergencyLightsForJalur2();
    currentState = AMBULANCE_EMERGENCY_JALUR_2;
    lastStateChange = millis();
    ambulanceDetectionCount++;
  } else {
    Serial.println(" -> No ambulance");
  }
}

// ===========================
//  MENGEMBALIKAN KE FASE NORMAL SETELAH EMERGENCY
// ===========================
void returnToNormalPhase() {
  Serial.println("🔄 Emergency ended - Returning to previous phase");
  
  // Kembalikan ke fase sebelum emergency
  currentPhase = phaseBeforeEmergency;
  currentState = NORMAL_OPERATION;
  lastStateChange = millis();
  lastAudioCheck = millis();
  detection_count_jalur1 = 0;
  detection_count_jalur2 = 0;
  
  // Tampilkan fase yang dikembalikan
  switch(currentPhase) {
    case PHASE_JALUR1_GREEN:
      Serial.println("   ↪️ Kembali ke: JALUR 1 HIJAU");
      setJalur1Green();
      break;
    case PHASE_JALUR1_YELLOW:
      Serial.println("   ↪️ Kembali ke: JALUR 1 KUNING");
      setJalur1Yellow();
      break;
    case PHASE_ALL_RED_1:
      Serial.println("   ↪️ Kembali ke: SEMUA MERAH (setelah J1)");
      setAllRed();
      break;
    case PHASE_JALUR2_GREEN:
      Serial.println("   ↪️ Kembali ke: JALUR 2 HIJAU");
      setJalur2Green();
      break;
    case PHASE_JALUR2_YELLOW:
      Serial.println("   ↪️ Kembali ke: JALUR 2 KUNING");
      setJalur2Yellow();
      break;
    case PHASE_ALL_RED_2:
      Serial.println("   ↪️ Kembali ke: SEMUA MERAH (setelah J2)");
      setAllRed();
      break;
    case PHASE_JALUR3_GREEN:
      Serial.println("   ↪️ Kembali ke: JALUR 3 HIJAU");
      setJalur3Green();
      break;
    case PHASE_JALUR3_YELLOW:
      Serial.println("   ↪️ Kembali ke: JALUR 3 KUNING");
      setJalur3Yellow();
      break;
    case PHASE_ALL_RED_3:
      Serial.println("   ↪️ Kembali ke: SEMUA MERAH (setelah J3)");
      setAllRed();
      break;
  }
}

// ===========================
//  MODE NORMAL - SIKLUS PERTIGAAN
// ===========================
void controlNormalTraffic() {
  unsigned long now = millis();
  unsigned long elapsed = now - lastStateChange;

  switch (currentPhase) {
    case PHASE_JALUR1_GREEN:
      setJalur1Green();
      if (elapsed > GREEN_DURATION) {
        currentPhase = PHASE_JALUR1_YELLOW;
        lastStateChange = now;
        Serial.println("🟡 Phase change: JALUR 1 HIJAU -> KUNING");
      }
      break;
      
    case PHASE_JALUR1_YELLOW:
      setJalur1Yellow();
      if (elapsed > YELLOW_DURATION) {
        currentPhase = PHASE_ALL_RED_1;
        lastStateChange = now;
        Serial.println("🔴 Phase change: JALUR 1 KUNING -> SEMUA MERAH");
      }
      break;
      
    case PHASE_ALL_RED_1:
      setAllRed();
      if (elapsed > ALL_RED_DURATION) {
        currentPhase = PHASE_JALUR2_GREEN;
        lastStateChange = now;
        Serial.println("🟢 Phase change: SEMUA MERAH -> JALUR 2 HIJAU");
      }
      break;
      
    case PHASE_JALUR2_GREEN:
      setJalur2Green();
      if (elapsed > GREEN_DURATION) {
        currentPhase = PHASE_JALUR2_YELLOW;
        lastStateChange = now;
        Serial.println("🟡 Phase change: JALUR 2 HIJAU -> KUNING");
      }
      break;
      
    case PHASE_JALUR2_YELLOW:
      setJalur2Yellow();
      if (elapsed > YELLOW_DURATION) {
        currentPhase = PHASE_ALL_RED_2;
        lastStateChange = now;
        Serial.println("🔴 Phase change: JALUR 2 KUNING -> SEMUA MERAH");
      }
      break;
      
    case PHASE_ALL_RED_2:
      setAllRed();
      if (elapsed > ALL_RED_DURATION) {
        currentPhase = PHASE_JALUR3_GREEN;
        lastStateChange = now;
        Serial.println("🟢 Phase change: SEMUA MERAH -> JALUR 3 HIJAU");
      }
      break;
      
    case PHASE_JALUR3_GREEN:
      setJalur3Green();
      if (elapsed > GREEN_DURATION) {
        currentPhase = PHASE_JALUR3_YELLOW;
        lastStateChange = now;
        Serial.println("🟡 Phase change: JALUR 3 HIJAU -> KUNING");
      }
      break;
      
    case PHASE_JALUR3_YELLOW:
      setJalur3Yellow();
      if (elapsed > YELLOW_DURATION) {
        currentPhase = PHASE_ALL_RED_3;
        lastStateChange = now;
        Serial.println("🔴 Phase change: JALUR 3 KUNING -> SEMUA MERAH");
      }
      break;
      
    case PHASE_ALL_RED_3:
      setAllRed();
      if (elapsed > ALL_RED_DURATION) {
        currentPhase = PHASE_JALUR1_GREEN;
        lastStateChange = now;
        Serial.println("🟢 Phase change: SEMUA MERAH -> JALUR 1 HIJAU");
      }
      break;
  }
}

// ===========================
//  SETUP
// ===========================
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n🚦 ESP32-S3 SMART TRAFFIC SYSTEM - CONTINUOUS CYCLE");
  Serial.println("🎯 SIKLUS NORMAL + KEMBALI KE FASE SEBELUM EMERGENCY");
  Serial.printf("🔧 Detection Threshold: %.2f\n", DETECTION_THRESHOLD);

  // Inisialisasi pin
  pinMode(RED_1, OUTPUT); pinMode(YELLOW_1, OUTPUT); pinMode(GREEN_1, OUTPUT);
  pinMode(RED_2, OUTPUT); pinMode(YELLOW_2, OUTPUT); pinMode(GREEN_2, OUTPUT);
  pinMode(RED_3, OUTPUT); pinMode(YELLOW_3, OUTPUT); pinMode(GREEN_3, OUTPUT);
  
  blinkAllYellow(2);

  // Memory allocation
  if (psramFound()) {
    Serial.println("✅ PSRAM available - Using PSRAM");
    tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
  } else {
    Serial.println("⚠️ PSRAM not found - Using DRAM");
    tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
  }
  
  if (!tensor_arena) {
    Serial.println("❌ Memory allocation failed!");
    setErrorState();
  }

  initI2S();

  // Load model
  model = tflite::GetModel(model_ambulance_siren_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("❌ Model version mismatch. Expected: %d, Got: %d\n", 
                  TFLITE_SCHEMA_VERSION, model->version());
    setErrorState();
  }

  // Resolver
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddRelu();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Tensor allocation failed");
    setErrorState();
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.printf("💾 Free Heap: %d bytes\n", ESP.getFreeHeap());
  
  // Mulai dengan semua merah, lalu jalur 1 hijau
  setAllRed();
  lastStateChange = millis();
  lastAudioCheck = millis();
  currentPhase = PHASE_JALUR1_GREEN;
  phaseBeforeEmergency = PHASE_JALUR1_GREEN; // Inisialisasi

  Serial.println("\n✅ System ready - Continuous cycle with emergency memory");
  Serial.println("🔄 Siklus normal + kembali ke fase sebelumnya setelah emergency");
  Serial.println("==============================================");
}

// ===========================
//  LOOP
// ===========================
void loop() {
  unsigned long now = millis();

  switch (currentState) {
    case NORMAL_OPERATION:
      controlNormalTraffic();
      if (now - lastAudioCheck >= AUDIO_CHECK_INTERVAL) {
        lastAudioCheck = now;
        classifyAudio();
      }
      break;

    case AMBULANCE_EMERGENCY_JALUR_1:
    case AMBULANCE_EMERGENCY_JALUR_2:
      if (now - lastStateChange >= EMERGENCY_TIME) {
        returnToNormalPhase();
      }
      break;

    case SYSTEM_ERROR:
      break;
  }

  static unsigned long lastStatusPrint = 0;
  if (now - lastStatusPrint >= 15000) {
    lastStatusPrint = now;
    Serial.printf("📈 Stats - Detections: %d, Inferences: %d\n", 
                  ambulanceDetectionCount, inferenceCount);
  }

  delay(10);
}