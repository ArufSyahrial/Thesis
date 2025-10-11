import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sys

# Add config to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import get_path

def validate_model_and_data():
    """Validate model and dataset compatibility"""
    print("🔍 Validating Model and Dataset...")
    
    try:
        # Check if files exist
        model_path = get_path('model_h5')
        features_path = get_path('features_pkl')
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        if not os.path.exists(features_path):
            print(f"❌ Features file not found: {features_path}")
            return False
        
        # Load model
        print("📥 Loading model...")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Display model architecture
        print("\n📋 Model Architecture:")
        model.summary()
        
        # Load features data
        print("\n📥 Loading features data...")
        with open(features_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different data structures
        if isinstance(data, dict):
            features = np.array(data['features'])
            labels = np.array(data['labels'])
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            features, labels = data
        else:
            print("❌ Unknown data structure in pickle file")
            return False
        
        print(f"✅ Data loaded successfully")
        print(f"📊 Features shape: {features.shape}")
        print(f"🎯 Labels shape: {labels.shape}")
        print(f"🔤 Unique labels: {np.unique(labels)}")
        
        # Check compatibility with model
        print(f"\n🔧 Model input shape: {model.input_shape}")
        print(f"🔧 Features shape: {features.shape}")
        
        # Reshape features if needed for CNN
        if len(model.input_shape) == 3:  # CNN expects 3D input
            expected_timesteps = model.input_shape[1]
            if features.shape[1] == expected_timesteps:
                features_reshaped = features.reshape(features.shape[0], features.shape[1], 1)
                print(f"✅ Features compatible - reshaped to: {features_reshaped.shape}")
            else:
                print(f"❌ Feature dimension mismatch!")
                print(f"   Expected: {expected_timesteps}, Got: {features.shape[1]}")
                return False
        else:
            features_reshaped = features
        
        # Evaluate model performance
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Preprocess data
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        labels_categorical = tf.keras.utils.to_categorical(labels_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_reshaped, labels_categorical, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Data split:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Testing samples: {X_test.shape[0]}")
        
        # Evaluate model
        print("\n🧪 Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"🎯 Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"📉 Test Loss: {test_loss:.4f}")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print("📊 Confusion Matrix:")
        print(cm)
        
        # Check ESP32 compatibility
        print("\n🔍 ESP32 Compatibility Check:")
        print(f"📦 Model parameters: {model.count_params()}")
        print(f"📦 Model file size: {os.path.getsize(model_path) / 1024:.2f} KB")
        print(f"📥 Input shape: {model.input_shape}")
        print(f"📤 Output shape: {model.output_shape}")
        
        # Test single prediction
        print(f"\n🧪 Testing single prediction...")
        sample_idx = 0
        sample_input = X_test[sample_idx:sample_idx+1]
        sample_pred = model.predict(sample_input)
        predicted_class = le.inverse_transform([np.argmax(sample_pred)])[0]
        true_class = le.inverse_transform([y_true_classes[sample_idx]])[0]
        
        print(f"📊 Prediction probabilities: {sample_pred[0]}")
        print(f"🎯 Predicted class: {predicted_class}")
        print(f"✅ True class: {true_class}")
        
        # Create visualization
        create_validation_plots(model, features, labels, le, cm, test_accuracy)
        
        print("\n✅ Validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_validation_plots(model, features, labels, label_encoder, confusion_matrix, accuracy):
    """Create validation plots and charts"""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Confusion Matrix
    plt.subplot(2, 3, 1)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")
    
    # Plot 2: Class Distribution
    plt.subplot(2, 3, 2)
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Plot 3: Feature Distribution
    plt.subplot(2, 3, 3)
    plt.hist(features.flatten(), bins=50, alpha=0.7)
    plt.title('Feature Value Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    
    # Plot 4: Sample Features
    plt.subplot(2, 3, 4)
    for i in range(min(3, features.shape[0])):
        plt.plot(features[i], alpha=0.7, label=f'Sample {i+1}')
    plt.title('Sample Feature Vectors')
    plt.xlabel('Feature Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    
    # Plot 5: Model Info
    plt.subplot(2, 3, 5)
    plt.axis('off')
    info_text = f"""
    Model Information:
    - Input Shape: {model.input_shape}
    - Output Shape: {model.output_shape}
    - Parameters: {model.count_params():,}
    - Test Accuracy: {accuracy*100:.2f}%
    - Classes: {', '.join(label_encoder.classes_)}
    """
    plt.text(0.1, 0.9, info_text, fontsize=10, verticalalignment='top')
    plt.title('Model Summary')
    
    # Plot 6: Feature Statistics
    plt.subplot(2, 3, 6)
    stats_text = f"""
    Feature Statistics:
    - Shape: {features.shape}
    - Min: {np.min(features):.4f}
    - Max: {np.max(features):.4f}
    - Mean: {np.mean(features):.4f}
    - Std: {np.std(features):.4f}
    - Samples: {features.shape[0]}
    """
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top')
    plt.axis('off')
    plt.title('Feature Statistics')
    
    plt.tight_layout()
    plt.savefig('model_validation_report.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    validate_model_and_data()