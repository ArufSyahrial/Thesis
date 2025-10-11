import os
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import sys

# Add config to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import get_path

def extract_features_from_audio(file_path, n_features=80, sr=16000):
    """
    Extract features from audio file (compatible with ESP32 implementation)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sr)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Pre-emphasis filter (same as ESP32)
        emphasized_audio = np.append(audio[0], audio[1:] - 0.95 * audio[:-1])
        
        # Frame blocking - divide into n_features frames
        frame_size = len(emphasized_audio) // n_features
        features = np.zeros(n_features)
        
        for frame in range(n_features):
            start = frame * frame_size
            end = start + frame_size
            
            if end > len(emphasized_audio):
                end = len(emphasized_audio)
            
            # Calculate RMS energy for each frame (same as ESP32)
            if end > start:
                frame_energy = np.sum(emphasized_audio[start:end] ** 2)
                features[frame] = np.sqrt(frame_energy / (end - start))
        
        # Normalize features (zero mean, unit variance) - same as ESP32
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_dataset_from_folder(data_folder, output_file=None, n_features=80):
    """
    Create dataset from folder containing audio files
    Folder structure should be: data_folder/class_name/audio_files
    """
    if output_file is None:
        output_file = get_path('features_pkl')
    
    features = []
    labels = []
    
    # Find all subfolders (each subfolder is one class)
    classes = [d for d in os.listdir(data_folder) 
               if os.path.isdir(os.path.join(data_folder, d))]
    
    print(f"🔍 Found classes: {classes}")
    
    for class_name in classes:
        class_path = os.path.join(data_folder, class_name)
        audio_files = [f for f in os.listdir(class_path) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        print(f"📁 Processing {len(audio_files)} files in '{class_name}'")
        
        for i, audio_file in enumerate(audio_files):
            file_path = os.path.join(class_path, audio_file)
            
            # Extract features
            feature = extract_features_from_audio(file_path, n_features)
            
            if feature is not None:
                features.append(feature)
                labels.append(class_name)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(audio_files)} files")
    
    # Convert to numpy arrays
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    # Save to pickle file
    data_dict = {
        'features': features_array,
        'labels': labels_array
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"\n✅ Dataset created successfully!")
    print(f"📊 Total samples: {len(features)}")
    print(f"📊 Feature shape: {features_array.shape}")
    print(f"🎯 Classes distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")
    print(f"💾 Saved to: {output_file}")
    
    return features_array, labels_array

def analyze_audio_file(file_path):
    """
    Analyze audio file and display detailed information
    """
    print(f"\n🔍 Analyzing: {file_path}")
    
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=None)
        
        print(f"📊 Sample rate: {sr} Hz")
        print(f"📊 Duration: {len(audio)/sr:.2f} seconds")
        print(f"📊 Samples: {len(audio)}")
        print(f"📊 Min/Max amplitude: {np.min(audio):.4f} / {np.max(audio):.4f}")
        
        # Extract features
        features = extract_features_from_audio(file_path)
        
        if features is not None:
            print(f"📊 Extracted features: {len(features)}")
            print(f"📊 Features range: {np.min(features):.4f} to {np.max(features):.4f}")
            print(f"📊 Features mean: {np.mean(features):.4f}")
            print(f"📊 Features std: {np.std(features):.4f}")
            
            # Plot audio signal and features
            plt.figure(figsize=(12, 8))
            
            # Plot audio signal
            plt.subplot(2, 1, 1)
            time_axis = np.linspace(0, len(audio)/sr, len(audio))
            plt.plot(time_axis, audio)
            plt.title(f'Audio Signal: {os.path.basename(file_path)}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Plot features
            plt.subplot(2, 1, 2)
            plt.plot(features, 'o-', linewidth=2, markersize=4)
            plt.title('Extracted Features (RMS Energy per Frame)')
            plt.xlabel('Frame Index')
            plt.ylabel('Normalized Energy')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return features
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return None

def test_feature_extraction():
    """
    Test feature extraction with dummy data
    """
    print("🧪 Testing feature extraction...")
    
    # Generate dummy audio data (sine wave + noise)
    duration = 1.0  # seconds
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    
    # Test with sine wave (simulating ambulance siren)
    test_audio = 0.5 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.normal(0, 1, len(t))
    
    features = extract_features_from_raw_audio(test_audio, n_features=80)
    
    if features is not None:
        print(f"✅ Feature extraction successful!")
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 First 10 features: {features[:10]}")
        
        return True
    else:
        print("❌ Feature extraction failed!")
        return False

def extract_features_from_raw_audio(audio_data, n_features=80, sr=16000):
    """
    Extract features from raw audio data (for real-time processing)
    """
    try:
        # Normalize audio
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Pre-emphasis filter
        emphasized_audio = np.append(audio_data[0], audio_data[1:] - 0.95 * audio_data[:-1])
        
        # Frame blocking
        frame_size = len(emphasized_audio) // n_features
        features = np.zeros(n_features)
        
        for frame in range(n_features):
            start = frame * frame_size
            end = start + frame_size
            
            if end > len(emphasized_audio):
                end = len(emphasized_audio)
            
            # Calculate RMS energy
            if end > start:
                frame_energy = np.sum(emphasized_audio[start:end] ** 2)
                features[frame] = np.sqrt(frame_energy / (end - start))
        
        # Normalize features
        if np.std(features) > 0:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
        
    except Exception as e:
        print(f"Error processing audio data: {e}")
        return None

if __name__ == "__main__":
    print("🎵 Audio Feature Extraction Tool")
    print("=" * 50)
    
    # Test feature extraction
    test_feature_extraction()
    
    # Example usage:
    # create_dataset_from_folder('sounds/', 'data/Extracted_features.pkl')
    # analyze_audio_file('path/to/your/audio.wav')