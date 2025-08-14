import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Adds ~/Projects/sensor_ai
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



# Get list of all .csv files in the raw_data directory
files = [f for f in os.listdir(config.DATA_DIR) if f.endswith('.csv')]

for filename in files:
    # Use filename without .csv as the label
    label = os.path.splitext(filename)[0]
    df = pd.read_csv(os.path.join(config.DATA_DIR, filename))

    # Convert timestamp strings to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Drop any rows where timestamp couldn't be parsed
    df = df.dropna(subset=['timestamp'])

    # Convert to seconds since start
    df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    # Plot both accel and gyro in same figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Accelerometer
    ax[0].plot(df['time_sec'], df['accel_x'], label='accel_x')
    ax[0].plot(df['time_sec'], df['accel_y'], label='accel_y')
    ax[0].plot(df['time_sec'], df['accel_z'], label='accel_z')
    ax[0].set_ylabel('Acceleration')
    ax[0].set_title(label)
    ax[0].legend()
    ax[0].grid(True)

    # Gyroscope
    ax[1].plot(df['time_sec'], df['gyro_x'], label='gyro_x')
    ax[1].plot(df['time_sec'], df['gyro_y'], label='gyro_y')
    ax[1].plot(df['time_sec'], df['gyro_z'], label='gyro_z')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Angular velocity')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()