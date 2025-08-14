import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "sensor_data"
SAMPLE_RATE = 50  # Hz
WINDOW_SIZE_SECONDS = 4
STRIDE_SECONDS = 1
WINDOW_SIZE = WINDOW_SIZE_SECONDS * SAMPLE_RATE  # 200 samples
STRIDE = STRIDE_SECONDS * SAMPLE_RATE  # 50 samples

def create_positioned_windows(gesture_data, target_length=200, num_positions=5):
    """
    Create multiple windows by placing gesture at different positions
    within the target window length, padding with noise/idle data
    """
    gesture_len = len(gesture_data)
    if gesture_len >= target_length:
        return [gesture_data[:target_length]]  # Just truncate if too long
    
    windows = []
    
    # Generate low-level noise (simulate idle state)
    idle_noise = np.random.normal(0, 0.1, (target_length, gesture_data.shape[1]))
    
    for i in range(num_positions):
        window = idle_noise.copy()
        
        # Place gesture at different positions
        if i == 0:  # Beginning
            start_pos = 0
        elif i == num_positions - 1:  # End
            start_pos = target_length - gesture_len
        else:  # Distributed in middle
            start_pos = int((target_length - gesture_len) * i / (num_positions - 1))
        
        window[start_pos:start_pos + gesture_len] = gesture_data
        windows.append(window)
    
    return windows

def load_and_preprocess():
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    for filename in files:
        # Extract label: files starting with 'circle' -> 'circle', everything else -> 'misc'
        filename_without_ext = filename.replace('.csv', '')
        if filename_without_ext.startswith('circle'):
            base_label = 'circle'
        else:
            base_label = 'misc'
        
        df = pd.read_csv(os.path.join(DATA_DIR, filename))
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        
        sensors = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        data = df[sensors].values
        
        # Normalize data (z-score normalization)
        def normalize_window(window):
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            return (window - mean) / std
        
        # Handle files shorter than WINDOW_SIZE
        if len(data) < WINDOW_SIZE:
            # Create positioned windows for training
            positioned_windows = create_positioned_windows(data, WINDOW_SIZE)
            
            for window in positioned_windows:
                normalized_window = normalize_window(window)
                X_train.append(normalized_window)
                y_train.append(base_label)
            
            # Use one positioned window for test (middle position)
            test_window = positioned_windows[len(positioned_windows)//2]
            normalized_test_window = normalize_window(test_window)
            X_test.append(normalized_test_window)
            y_test.append(base_label)
            
        else:
            # Existing logic for files >= WINDOW_SIZE
            # Extract test window from middle
            mid_start = (len(data) - WINDOW_SIZE) // 2
            test_window = data[mid_start:mid_start + WINDOW_SIZE]
            normalized_test_window = normalize_window(test_window)
            X_test.append(normalized_test_window)
            y_test.append(base_label)
            
            # Add remaining windows to training (excluding test window)
            start = 0
            while start + WINDOW_SIZE <= len(data):
                if start == mid_start:
                    start += STRIDE
                    continue
                window = data[start:start + WINDOW_SIZE]
                normalized_window = normalize_window(window)
                X_train.append(normalized_window)
                y_train.append(base_label)
                start += STRIDE
    
    # Convert to numpy arrays - now all windows should be same size
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Encode string labels to integers
    label_encoder = LabelEncoder()
    all_labels = y_train + y_test
    label_encoder.fit(all_labels)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    return (X_train, y_train_enc), (X_test, y_test_enc), label_encoder

# Add function to create idle data samples
def create_idle_samples(num_samples=50, window_size=200, sensor_count=6):
    """Create realistic idle/stationary samples"""
    idle_samples = []
    for _ in range(num_samples):
        # Simulate device sitting on table with minimal movement
        sample = np.random.normal(0, 0.05, (window_size, sensor_count))
        # Add slight gravity component to accelerometer
        sample[:, 2] += 9.8 + np.random.normal(0, 0.1, window_size)  # Z-axis gravity
        idle_samples.append(sample)
    return np.array(idle_samples)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), label_encoder = load_and_preprocess()
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Train labels (encoded): {set(y_train)}")
    print(f"Test labels (encoded): {set(y_test)}")
    print(f"Label classes: {list(label_encoder.classes_)}")