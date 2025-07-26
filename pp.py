import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "raw_data"
SAMPLE_RATE = 50  # Hz
WINDOW_SIZE_SECONDS = 4
STRIDE_SECONDS = 1

WINDOW_SIZE = WINDOW_SIZE_SECONDS * SAMPLE_RATE
STRIDE = STRIDE_SECONDS * SAMPLE_RATE

def load_and_preprocess():
    X_train, y_train = [], []
    X_test, y_test = [], []

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    for filename in files:
        base_label = filename.replace('.csv', '')

        df = pd.read_csv(os.path.join(DATA_DIR, filename))
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp'])

        df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        sensors = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        data = df[sensors].values

        if len(data) < WINDOW_SIZE:
            continue  # skip short files

        # Calculate middle index to start the test window
        mid_start = (len(data) - WINDOW_SIZE) // 2

        # Extract test window from middle
        test_window = data[mid_start:mid_start + WINDOW_SIZE]
        mean = test_window.mean(axis=0)
        std = test_window.std(axis=0)
        std[std == 0] = 1
        test_window = (test_window - mean) / std

        X_test.append(test_window)
        y_test.append(base_label)

        # Add remaining windows (excluding the middle test window) to training
        start = 0
        while start + WINDOW_SIZE <= len(data):
            # Skip the window used for testing
            if start == mid_start:
                start += STRIDE
                continue

            window = data[start:start + WINDOW_SIZE]
            mean = window.mean(axis=0)
            std = window.std(axis=0)
            std[std == 0] = 1
            window = (window - mean) / std

            X_train.append(window)
            y_train.append(base_label)

            start += STRIDE

    # Convert to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    all_labels = y_train + y_test
    label_encoder.fit(all_labels)

    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    return (X_train, y_train_enc), (X_test, y_test_enc), label_encoder



if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), label_encoder = load_and_preprocess()
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Train labels (encoded): {set(y_train)}")
    print(f"Test labels (encoded): {set(y_test)}")
    print(f"Label classes: {list(label_encoder.classes_)}")
