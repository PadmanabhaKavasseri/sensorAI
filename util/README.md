Description of files in current directory

viz_data.py
Plots all CSVs in sensor_data directory
python viz_data.py


convert_aihub.py
Converts PyTorch model specified in model_path to tflite using AI HUB. 
Will need to go to AI HUB https://app.aihub.qualcomm.com/jobs to download.
More Help at
https://aihub.qualcomm.com/get-started

If RB3 name changed use this to find the name to use
qai-hub list-devices


dataset.py
Used in main.py
Makes the data compatible with PyTorch's DataLoader
Not something you run itself


pp.py
Only load_and_preprocess used in main.
Can run the file standalone. 
This script loads raw gesture .csv files from config.DATA_DIR, applies normalization, and generates fixed-length sliding windows for training and testing.
It supports short gesture padding with synthetic idle noise, label encoding, and optional idle sample generation for balancing datasets.
Outputs NumPy arrays (X_train, y_train), (X_test, y_test) and a fitted LabelEncoder for model training.

This script loads CSV files of accelerometer and gyroscope data, normalizes them, and splits them into training and testing sets for gesture classification.
Features
Handles variable-length recordings:
If shorter than WINDOW_SIZE (default 4 seconds at 50 Hz), the gesture is placed at multiple positions within a fixed-size window for training.
If longer, overlapping sliding windows are extracted.
Z-score normalization for each window.
Automatic label extraction:
Filenames starting with circle → label circle.
Others → label misc.
Train/Test split:
One middle-position window is reserved for testing.
Remaining windows go to training.
Label encoding into integers.
Configuration
SAMPLE_RATE: 50 Hz
WINDOW_SIZE_SECONDS: 4
STRIDE_SECONDS: 1
CSV files must be in config.DATA_DIR.

flowchart TD
    A[CSV Files in DATA_DIR] --> B[Read with Pandas]
    B --> C[Extract accel/gyro columns]
    C --> D[Z-score normalization per window]
    D --> E{Data length < WINDOW_SIZE?}
    E -->|Yes| F[Create positioned windows]
    E -->|No| G[Sliding window extraction]
    F --> H[Assign train/test split]
    G --> H
    H --> I[Encode labels with LabelEncoder]
    I --> J[(X_train, y_train), (X_test, y_test)]


