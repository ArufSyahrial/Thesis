#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model_ambulance_siren.h"

#include <I2S.h>

// Configuration
#define SAMPLE_RATE 16000
#define RECORD_MS 1000
#define NUM_SAMPLES (SAMPLE_RATE * RECORD_MS / 1000)
#define NUM_FEATURES 80

// ESP32-S3 Traffic Light LED Pins
#define RED_LED    4
#define YELLOW_LED 5
#define GREEN_LED  6

// ESP32-S3 I2S Pin Configuration for INMP441
#define I2S_BCK    8
#define I2S_WS     9
#define I2S_DATA   10

// TensorFlow Lite setup
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 16 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// Buffers
int32_t audio_buffer[NUM_SAMPLES];
float features[NUM_FEATURES];

// Traffic Light States
enum TrafficLightState {
  NORMAL_OPERATION,
  AMBULANCE_EMERGENCY,
  SYSTEM_ERROR
};

TrafficLightState currentState = NORMAL_OPERATION;
unsigned long lastStateChange = 0;
unsigned long lastAudioCheck = 0;

const unsigned long NORMAL_CYCLE_TIME = 10000;
const unsigned long EMERGENCY_TIME = 15000;
const unsigned long AUDIO_CHECK_INTERVAL = 3000;

unsigned long totalInferenceTime = 0;
unsigned int inferenceCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("==========================================");
  Serial.println("🚨 Smart Traffic Light Controller - ESP32-S3");
  Serial.println("🎯 Ambulance Detection System");
  Serial.println("📊 Model: CNN 1D (80 features)");
  Serial.println("🎯 Classes: Ambulance vs Traffic");
  Serial.println("==========================================");
  
  // Initialize Traffic Light LEDs
  pinMode(RED_LED, OUTPUT);
  pinMode(YELLOW_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  
  // Start with blinking yellow (system initializing)
  blinkYellow(3);
  
  // Initialize I2S for INMP441 with ESP32-S3 pins
  Serial.println("🎤 Initializing INMP441 I2S...");
  I2S.setPins(I2S_BCK, I2S_WS, I2S_DATA);
  if (!I2S.begin(I2S_PHILIPS_MODE, SAMPLE_RATE, 32)) {
    Serial.println("❌ Failed to initialize I2S!");
    setErrorState();
    return;
  }
  Serial.println("✅ I2S initialized successfully");
  
  // Load TFLite model
  Serial.println("🧠 Loading TensorFlow Lite model...");
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  model = tflite::GetModel(model_ambulance_siren_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model schema mismatch!");
    setErrorState();
    return;
  }
  
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ AllocateTensors() failed");
    setErrorState();
    return;
  }
  
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("✅ TensorFlow Lite model loaded");
  Serial.print("📥 Input shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
  
  // Start with RED light
  setTrafficLight(RED_LED);
  lastStateChange = millis();
  
  Serial.println("🚦 System initialized - RED Light Active");
  Serial.println("🎤 Monitoring for ambulance sirens...");
  Serial.println("==========================================");
}

void blinkYellow(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(YELLOW_LED, HIGH);
    delay(300);
    digitalWrite(YELLOW_LED, LOW);
    delay(300);
  }
}

void setErrorState() {
  currentState = SYSTEM_ERROR;
  // Blink red LED to indicate error
  while (true) {
    digitalWrite(RED_LED, HIGH);
    delay(500);
    digitalWrite(RED_LED, LOW);
    delay(500);
    Serial.println("❌ SYSTEM ERROR - Check connections");
  }
}

void setTrafficLight(int led) {
  // Turn off all LEDs
  digitalWrite(RED_LED, LOW);
  digitalWrite(YELLOW_LED, LOW);
  digitalWrite(GREEN_LED, LOW);
  
  // Turn on selected LED
  digitalWrite(led, HIGH);
  
  // Print state change
  switch (led) {
    case RED_LED:
      Serial.println("🚦 State: RED Light");
      break;
    case YELLOW_LED:
      Serial.println("🚦 State: YELLOW Light");
      break;
    case GREEN_LED:
      Serial.println("🚦 State: GREEN Light");
      break;
  }
}

void controlNormalTraffic() {
  unsigned long currentTime = millis();
  unsigned long cycleTime = currentTime - lastStateChange;
  
  if (cycleTime < NORMAL_CYCLE_TIME) {
    // RED phase
    setTrafficLight(RED_LED);
  } else if (cycleTime < NORMAL_CYCLE_TIME * 2) {
    // GREEN phase  
    setTrafficLight(GREEN_LED);
  } else if (cycleTime < NORMAL_CYCLE_TIME * 2 + 3000) {
    // YELLOW phase (3 seconds)
    setTrafficLight(YELLOW_LED);
  } else {
    // Reset cycle
    lastStateChange = currentTime;
    Serial.println("🔄 Traffic cycle reset");
  }
}

void handleAmbulanceDetection() {
  Serial.println("==========================================");
  Serial.println("🚨🚨🚨 EMERGENCY: AMBULANCE DETECTED! 🚨🚨🚨");
  Serial.println("🟢 Setting GREEN light for emergency vehicle");
  Serial.println("==========================================");
  
  setTrafficLight(GREEN_LED);
  lastStateChange = millis();
  currentState = AMBULANCE_EMERGENCY;
}

void extractFeatures(int32_t* audio, float* features) {
  // Normalize audio to [-1, 1]
  float normalized[NUM_SAMPLES];
  for (int i = 0; i < NUM_SAMPLES; i++) {
    normalized[i] = audio[i] / 2147483648.0;
  }
  
  // Pre-emphasis filter
  float emphasized[NUM_SAMPLES];
  emphasized[0] = normalized[0];
  for (int i = 1; i < NUM_SAMPLES; i++) {
    emphasized[i] = normalized[i] - 0.95 * normalized[i-1];
  }
  
  // Frame blocking - divide into 80 frames
  int frame_size = NUM_SAMPLES / NUM_FEATURES;
  for (int frame = 0; frame < NUM_FEATURES; frame++) {
    float frame_energy = 0.0;
    int start = frame * frame_size;
    int end = start + frame_size;
    
    // Calculate RMS energy for each frame
    for (int i = start; i < end && i < NUM_SAMPLES; i++) {
      frame_energy += emphasized[i] * emphasized[i];
    }
    features[frame] = sqrt(frame_energy / frame_size);
  }
  
  // Normalize features (zero mean, unit variance)
  float mean = 0.0, std = 0.0;
  for (int i = 0; i < NUM_FEATURES; i++) {
    mean += features[i];
  }
  mean /= NUM_FEATURES;
  
  for (int i = 0; i < NUM_FEATURES; i++) {
    std += (features[i] - mean) * (features[i] - mean);
  }
  std = sqrt(std / NUM_FEATURES);
  
  for (int i = 0; i < NUM_FEATURES; i++) {
    features[i] = (features[i] - mean) / (std + 1e-8);
  }
}

void loop() {
  unsigned long currentTime = millis();
  
  // Handle traffic light state machine
  switch (currentState) {
    case NORMAL_OPERATION:
      controlNormalTraffic();
      break;
      
    case AMBULANCE_EMERGENCY:
      if (currentTime - lastStateChange >= EMERGENCY_TIME) {
        Serial.println("🟡 Emergency over - Returning to normal operation");
        setTrafficLight(YELLOW_LED);
        delay(2000); // Yellow for 2 seconds
        currentState = NORMAL_OPERATION;
        lastStateChange = currentTime;
      }
      break;
      
    case SYSTEM_ERROR:
      // Error state - already handled in setErrorState()
      break;
  }
  
  // Audio classification (only in normal operation)
  if (currentState == NORMAL_OPERATION && 
      currentTime - lastAudioCheck >= AUDIO_CHECK_INTERVAL) {
    lastAudioCheck = currentTime;
    classifyAudio();
  }
  
  // Performance monitoring (every minute)
  static unsigned long lastPerformanceLog = 0;
  if (currentTime - lastPerformanceLog >= 60000) {
    if (inferenceCount > 0) {
      Serial.printf("📊 Performance: Avg inference time: %.1f ms\n", 
                    (float)totalInferenceTime / inferenceCount);
    }
    totalInferenceTime = 0;
    inferenceCount = 0;
    lastPerformanceLog = currentTime;
  }
}

void classifyAudio() {
  // Record audio from INMP441
  int samples_read = 0;
  unsigned long record_start = millis();
  
  while (samples_read < NUM_SAMPLES) {
    int32_t sample = 0;
    if (I2S.read((void*)&sample, sizeof(sample)) == sizeof(sample)) {
      audio_buffer[samples_read] = sample;
      samples_read++;
    }
  }
  
  unsigned long record_time = millis() - record_start;
  
  // Extract features
  extractFeatures(audio_buffer, features);
  
  // Prepare input tensor (reshape to 1, 80, 1)
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features[i];
  }
  
  // Run inference
  unsigned long inference_start = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long inference_time = millis() - inference_start;
  
  // Update performance metrics
  totalInferenceTime += inference_time;
  inferenceCount++;
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("❌ Inference failed!");
    return;
  }
  
  // Get results
  float ambulance_prob = output->data.f[0];
  float traffic_prob = output->data.f[1];
  
  // Decision with confidence threshold
  float confidence_threshold = 0.8f; // 80% confidence
  
  if (ambulance_prob > confidence_threshold && ambulance_prob > traffic_prob) {
    handleAmbulanceDetection();
  } else {
    // Optional: Log low-confidence detections for debugging
    if (ambulance_prob > 0.3) {
      Serial.printf("🔊 Amb: %.1f%%, Traffic: %.1f%% (Below threshold)\n", 
                    ambulance_prob * 100, traffic_prob * 100);
    }
  }
}

// Test function for hardware verification
void testHardware() {
  Serial.println("🧪 Testing hardware components...");
  
  // Test LEDs
  Serial.println("💡 Testing LEDs...");
  setTrafficLight(RED_LED);
  delay(1000);
  setTrafficLight(YELLOW_LED);
  delay(1000);
  setTrafficLight(GREEN_LED);
  delay(1000);
  setTrafficLight(RED_LED);
  
  // Test microphone
  Serial.println("🎤 Testing microphone...");
  testMicrophone();
  
  Serial.println("✅ Hardware test completed");
}

void testMicrophone() {
  Serial.println("Recording test sample...");
  int32_t test_sample;
  for (int i = 0; i < 100; i++) {
    if (I2S.read((void*)&test_sample, sizeof(test_sample)) == sizeof(test_sample)) {
      if (i % 20 == 0) {
        Serial.printf("Sample %d: %d\n", i, test_sample);
      }
    }
    delay(10);
  }
}