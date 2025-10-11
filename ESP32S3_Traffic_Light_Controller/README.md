ESP32-S3 Traffic Light Controller with Audio Classification
===========================================================

Overview
--------
Smart traffic light system that detects ambulance sirens using machine learning and automatically controls traffic lights to give priority to emergency vehicles.

Features
--------
- Real-time ambulance siren detection using CNN
- Automatic traffic light control (Red, Yellow, Green LEDs)
- INMP441 I2S microphone for high-quality audio input
- TensorFlow Lite for efficient embedded AI inference
- Performance monitoring and real-time logging
- High accuracy (100% validated on test dataset)
- Optimized for ESP32-S3 with dual-core processing

Hardware Requirements
---------------------
Main Components:
- ESP32-S3 Development Board
- INMP441 I2S Microphone Module
- LED Traffic Lights (Red, Yellow, Green) x3
- Resistors 220Ω x3
- Breadboard and jumper wires
- USB Cable for programming

Software Requirements
---------------------
Python Environment:
- Python 3.8+
- TensorFlow 2.13+
- Jupyter Notebook
- scikit-learn
- librosa
- matplotlib
- numpy

Embedded Development:
- PlatformIO (recommended) or Arduino IDE
- TensorFlow Lite for Microcontrollers
- Arduino I2S library

Project Structure
-----------------
ESP32S3_Traffic_Light_Controller/
│
├── notebooks/                          # Jupyter notebooks
│   └── Ambulance Siren and Traffic Noise Audio Classification.ipynb
│
├── data/                               # Dataset and features
│   ├── Extracted_features.pkl          # Pre-processed features
│   ├── test_audio/                     # Audio samples
│   └── performance_logs/               # Performance logs
│
├── src/                                # ESP32 source code
│   └── ESP32S3_Traffic_Light_Controller.ino
│
├── include/                            # Header files
│   └── model_ambulance_siren.h         # TFLite model as C array
│
├── models/                             # Machine learning models
│   ├── my_model.h5                     # Original Keras model
│   ├── model_ambulance_siren.tflite    # Converted TFLite model
│   └── model_conversion_scripts/       # Conversion utilities
│
├── scripts/                            # Python utilities
│   ├── model_validation.py
│   ├── feature_extraction.py
│   └── data_analysis.py
│
├── docs/                               # Documentation
│   ├── wiring_diagram.png
│   └── troubleshooting.md
│
├── platformio.ini                      # PlatformIO configuration
├── partitions.csv                      # ESP32-S3 memory partitioning
├── config.py                           # Path configuration
└── README.txt                          # This file

Installation & Setup
--------------------
Step 1: Clone and Setup Directory Structure
git clone <repository-url>
cd ESP32S3_Traffic_Light_Controller
python config.py

Step 2: Install Python Dependencies
pip install tensorflow scikit-learn librosa matplotlib numpy jupyter

Step 3: Validate Existing Model and Data
python scripts/model_validation.py

Step 4: Convert Model for ESP32-S3
python models/model_conversion_scripts/convert_to_tflite.py
python models/model_conversion_scripts/convert_to_c_array.py

Step 5: Setup PlatformIO
pio project init --board esp32-s3-dev
pio run
pio run -t upload
pio device monitor

Wiring Diagram
--------------
ESP32-S3 to INMP441 Microphone:
3.3V  -> VDD
GND   -> GND
GPIO8 -> SCK (BCLK)
GPIO9 -> WS (LRCLK)
GPIO10 -> SD (DATA)
GND   -> L/R (select left channel)

ESP32-S3 to Traffic Light LEDs:
GPIO4 -> RED LED (+ 220Ω resistor)
GPIO5 -> YELLOW LED (+ 220Ω resistor)
GPIO6 -> GREEN LED (+ 220Ω resistor)
GND   -> Common GND

Model Information
-----------------
Architecture: 1D Convolutional Neural Network (CNN)
Input: 80 features from 1-second audio @ 16kHz
Output: 2 classes (ambulance, traffic)
Test Accuracy: 100%
Test Loss: 0.0002
Model Size: 47.12 KB (H5), ~25 KB (TFLite)
Inference Time: ~50-100ms on ESP32-S3

Usage
-----
Normal Operation:
1. System initializes with blinking yellow LED
2. Normal traffic light cycle: RED (10s) -> GREEN (10s) -> YELLOW (3s)
3. Audio monitoring every 3 seconds

Emergency Mode:
1. When ambulance detected, immediately switch to GREEN light
2. Green light stays for 15 seconds
3. Returns to normal cycle via YELLOW light (2 seconds)

Testing
-------
Hardware Testing:
Call testHardware() function in setup() to verify all components

Audio Testing:
python scripts/feature_extraction.py
python scripts/feature_extraction.py --analyze "data/test_audio/ambulance_sample.wav"

Performance Testing:
python scripts/data_analysis.py
python scripts/model_validation.py

Troubleshooting
---------------
Common Issues:
1. Microphone not working - Check I2S wiring and power
2. Model inference failing - Verify model files and memory allocation
3. LEDs not lighting - Check resistors and GPIO connections
4. Poor detection accuracy - Adjust confidence threshold or retrain model

Serial Error Messages:
- "Failed to initialize I2S!" - Check microphone wiring
- "AllocateTensors() failed" - Check model file and memory
- "Model schema mismatch!" - Update TensorFlow Lite version
- "Inference failed!" - Check input data format

Performance Metrics
-------------------
Real-time Performance (ESP32-S3):
- Audio Recording: ~1000ms
- Feature Extraction: ~15-25ms
- Model Inference: ~50-100ms
- Total Processing: ~1100-1125ms
- Memory Usage: ~16KB tensor arena

Accuracy Metrics:
- Precision: 100% (both classes)
- Recall: 100% (both classes)
- F1-Score: 1.0 (both classes)

Future Enhancements
-------------------
Planned Features:
1. Multiple intersection coordination
2. Advanced audio processing with multiple microphones
3. WiFi and MQTT integration for cloud monitoring
4. Enhanced safety mechanisms and battery backup

License
-------
MIT License

Contributing
------------
We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

Support
-------
If you encounter any issues:
1. Check the troubleshooting section
2. Review closed GitHub issues for similar problems
3. Create a new issue with detailed information

Acknowledgments
---------------
- TensorFlow team for TensorFlow Lite Micro
- ESP32 community for hardware support
- Contributors and testers

Note: This system is designed for educational and prototype purposes. For real-world traffic applications, additional safety measures and certifications would be required.

Last updated: October 2024