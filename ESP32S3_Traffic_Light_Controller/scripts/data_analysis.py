import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

def analyze_dataset(pickle_file='Extracted_features.pkl'):
    """
    Analisis dataset yang sudah diekstrak
    """
    print("🔍 Analyzing dataset...")
    
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"📊 Data type: {type(data)}")
        
        if isinstance(data, dict):
            features = data['features']
            labels = data['labels']
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            features, labels = data
        else:
            print("❌ Unknown data structure")
            return
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Features shape: {features.shape}")
        print(f"📊 Labels shape: {labels.shape}")
        print(f"🎯 Unique labels: {np.unique(labels)}")
        
        # Distribusi kelas
        unique, counts = np.unique(labels, return_counts=True)
        print(f"📈 Class distribution: {dict(zip(unique, counts))}")
        
        # Statistik fitur
        print(f"📊 Feature statistics:")
        print(f"   Min: {np.min(features):.4f}")
        print(f"   Max: {np.max(features):.4f}")
        print(f"   Mean: {np.mean(features):.4f}")
        print(f"   Std: {np.std(features):.4f}")
        
        # Visualisasi
        plt.figure(figsize=(15, 10))
        
        # Plot distribusi kelas
        plt.subplot(2, 3, 1)
        plt.bar(unique, counts)
        plt.title('Class Distribution')
        plt.xticks(rotation=45)
        
        # Plot histogram fitur
        plt.subplot(2, 3, 2)
        plt.hist(features.flatten(), bins=50)
        plt.title('Feature Value Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        
        # Plot beberapa sample fitur per kelas
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        
        for i, class_name in enumerate(le.classes_):
            plt.subplot(2, 3, 3 + i)
            class_indices = np.where(labels_encoded == i)[0]
            
            # Plot 5 sample pertama
            for j in range(min(5, len(class_indices))):
                plt.plot(features[class_indices[j]], alpha=0.7, label=f'Sample {j+1}')
            
            plt.title(f'Features - {class_name}')
            plt.xlabel('Feature Index')
            plt.ylabel('Value')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return features, labels
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return None, None

def check_feature_compatibility(model_input_shape, features_shape):
    """
    Cek kompatibilitas fitur dengan model
    """
    print("\n🔧 Checking feature compatibility with model...")
    
    print(f"📥 Model expects input: {model_input_shape}")
    print(f"📊 Features shape: {features_shape}")
    
    # Sesuaikan dengan model CNN 1D
    if len(model_input_shape) == 3:
        expected_features = model_input_shape[1]  # timesteps
        if features_shape[1] == expected_features:
            print(f"✅ Features compatible with model!")
            print(f"🔧 Reshape needed: (samples, {expected_features}) -> (samples, {expected_features}, 1)")
            return True
        else:
            print(f"❌ Feature dimension mismatch!")
            print(f"   Expected: {expected_features}, Got: {features_shape[1]}")
            return False
    else:
        print("⚠️ Unknown model input shape")
        return True

def main():
    """
    Analisis dataset utama
    """
    print("📊 Dataset Analysis Tool")
    print("=" * 50)
    
    # Analisis dataset yang ada
    features, labels = analyze_dataset('Extracted_features.pkl')
    
    if features is not None:
        # Cek kompatibilitas dengan model (asumsi model input shape)
        model_input_shape = (None, 80, 1)  # Sesuai dengan model Anda
        check_feature_compatibility(model_input_shape, features.shape)

if __name__ == "__main__":
    main()