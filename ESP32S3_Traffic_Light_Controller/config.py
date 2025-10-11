import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path configurations
PATHS = {
    'notebooks': os.path.join(BASE_DIR, 'notebooks'),
    'data': os.path.join(BASE_DIR, 'data'),
    'models': os.path.join(BASE_DIR, 'models'),
    'scripts': os.path.join(BASE_DIR, 'scripts'),
    'src': os.path.join(BASE_DIR, 'src'),
    'include': os.path.join(BASE_DIR, 'include'),
    'docs': os.path.join(BASE_DIR, 'docs')
}

# File paths
FILES = {
    'features_pkl': os.path.join(PATHS['data'], 'Extracted_features.pkl'),
    'model_h5': os.path.join(PATHS['models'], 'my_model.h5'),
    'model_tflite': os.path.join(PATHS['models'], 'model_ambulance_siren.tflite'),
    'model_header': os.path.join(PATHS['include'], 'model_ambulance_siren.h'),
    'platformio_ini': os.path.join(BASE_DIR, 'platformio.ini')
}

def get_path(key):
    """Get absolute path for file or directory"""
    return FILES.get(key, PATHS.get(key))

def setup_directories():
    """Create all necessary directories"""
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
        print(f"✅ Created: {path}")

if __name__ == "__main__":
    setup_directories()
    print("🎯 Directory structure setup complete!")