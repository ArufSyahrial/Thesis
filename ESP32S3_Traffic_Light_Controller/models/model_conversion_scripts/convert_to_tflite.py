import tensorflow as tf
import numpy as np
import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_path

def convert_to_tflite():
    print("🔄 Converting model to TensorFlow Lite...")
    
    try:
        # Load the trained model
        model_path = get_path('model_h5')
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
        # Display model architecture
        print("\n📋 Model Architecture:")
        model.summary()
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimizations for ESP32
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Reduce size
        
        # Convert model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_path = get_path('model_tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TFLite model saved to: {tflite_path}")
        print(f"📦 TFLite model size: {len(tflite_model) / 1024:.2f} KB")
        
        # Validate TFLite model
        validate_tflite_model(tflite_model)
        
        return tflite_model
        
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return None

def validate_tflite_model(tflite_model):
    """Validate the converted TFLite model"""
    print("\n🧪 Validating TFLite model...")
    
    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("📋 TFLite Model Details:")
        print(f"📥 Input: {input_details[0]['shape']} - {input_details[0]['dtype']}")
        print(f"📤 Output: {output_details[0]['shape']} - {output_details[0]['dtype']}")
        
        # Test with random input
        input_shape = input_details[0]['shape']
        test_input = np.random.random(input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✅ TFLite model validation successful!")
        print(f"🧪 Test output shape: {output_data.shape}")
        
    except Exception as e:
        print(f"❌ TFLite model validation failed: {e}")

if __name__ == "__main__":
    convert_to_tflite()