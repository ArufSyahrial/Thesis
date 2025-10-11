import os
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_path

def convert_to_c_array():
    print("🔄 Converting TFLite model to C array...")
    
    try:
        tflite_path = get_path('model_tflite')
        header_path = get_path('model_header')
        
        # Read TFLite model
        with open(tflite_path, 'rb') as f:
            model_data = f.read()
        
        # Generate C header file
        with open(header_path, 'w') as f:
            f.write('#ifndef MODEL_AMBULANCE_SIREN_H\n')
            f.write('#define MODEL_AMBULANCE_SIREN_H\n\n')
            f.write('#include <cstdint>\n\n')
            f.write('// TensorFlow Lite model for ambulance siren detection\n')
            f.write('// Input: 80 features, Output: 2 classes (ambulance, traffic)\n')
            f.write('// Model size: {} bytes\n\n'.format(len(model_data)))
            
            f.write('alignas(8) const unsigned char model_ambulance_siren_tflite[] = {\n')
            
            # Write bytes as hex array
            for i, byte in enumerate(model_data):
                if i % 12 == 0:
                    f.write('    ')
                f.write(f'0x{byte:02x}')
                if i < len(model_data) - 1:
                    f.write(', ')
                if (i + 1) % 12 == 0:
                    f.write('\n')
            
            f.write('\n};\n\n')
            f.write(f'const int model_ambulance_siren_tflite_len = {len(model_data)};\n\n')
            f.write('#endif  // MODEL_AMBULANCE_SIREN_H\n')
        
        print(f"✅ C header file saved to: {header_path}")
        print(f"📦 Model size: {len(model_data)} bytes ({len(model_data) / 1024:.2f} KB)")
        
        # Verify file was created
        if os.path.exists(header_path):
            file_size = os.path.getsize(header_path)
            print(f"📁 Header file size: {file_size / 1024:.2f} KB")
        else:
            print("❌ Header file was not created!")
            
    except Exception as e:
        print(f"❌ Error converting to C array: {e}")

if __name__ == "__main__":
    convert_to_c_array()