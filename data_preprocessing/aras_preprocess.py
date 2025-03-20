from glob import glob
import numpy as np
import pandas as pd
from collections import Counter
import logging

class TSDataSet:
    def __init__(self, data, label, length):
        self.data = data
        self.label = int(label)
        self.length = int(length)

# ARAS loader with downsampling (averaging over timespan)
# ARAS data format: sensor[0:19], activity1[20], activity2[21]
# Example: 0 0 0 0 ... 0 13 17
def arasLoader(file_name_pattern, timespan, min_seq):
    logging.info("Loading ARAS Dataset --------------------------------------")

    file_list = sorted(glob(file_name_pattern))
    dataset_list = []
    total_data_pointers = 0  # After downsampling

    for file_path in file_list:
        df = pd.read_csv(file_path, sep=' ', header=None).to_numpy()
        print(file_path)

        if len(df) == 0:
            continue

        current_label = [df[0, 20], df[0, 21]]  # Initial labels
        temp_segment = [df[0, 0:20]]           # Sensor data accumulation
        current_data = df[0, 0:20]
        activity_segments = []  # Store each activity segment (before downsampling)

        for i in range(1, len(df)):
            label_r1, label_r2 = df[i, 20], df[i, 21]
            sensor_data = df[i, 0:20]

            if (current_label[0] == label_r1) and (current_label[1] == label_r2):
                if (current_data != sensor_data).any():
                    temp_segment.append(sensor_data)
                    current_data = sensor_data
            else:
                # Store the segment if valid length
                if len(temp_segment) >= min_seq:
                    activity_segments.append((np.array(temp_segment), current_label))
                # Reset for new activity
                temp_segment = [sensor_data]
                current_data = sensor_data
                current_label = [label_r1, label_r2]

        # Final segment
        if len(temp_segment) >= min_seq:
            activity_segments.append((np.array(temp_segment), current_label))

        # Downsample each activity segment
        window_size = int(timespan / 1000)  # rows per timespan 
        for segment_data, labels in activity_segments:
            downsampled_sequence = []
            for idx in range(0, len(segment_data), window_size):
                window = segment_data[idx:idx+window_size]
                if len(window) > 0:
                    avg_vector = np.mean(window, axis=0)
                    downsampled_sequence.append(avg_vector)

            downsampled_sequence = np.stack(downsampled_sequence)
            downsampled_length = len(downsampled_sequence)
            total_data_pointers += downsampled_length

            # Determine which label changed (resident 1 or 2)
            label_to_use = labels[0] if labels[0] != 0 else labels[1]
            dataset_list.append(TSDataSet(downsampled_sequence, label_to_use, downsampled_length))

    # Summary
    sensor_channels = 20  # Fixed
    label_list = [ds.label for ds in dataset_list]
    activity_counts = Counter(label_list)
    num_activity_types = len(activity_counts)
    total_activities = len(dataset_list)

    logging.info("Loading ARAS Dataset Finished --------------------------------------")
    logging.info("====== Dataset Summary ======")
    logging.info(f"Sensor channels: {sensor_channels}")
    logging.info(f"Total data points (after downsampling): {total_data_pointers}")
    logging.info(f"Total activities (sequences): {total_activities}")
    logging.info(f"Number of activity types: {num_activity_types}")
    logging.info("Activity sequence counts and data points:")
    for label in sorted(activity_counts.keys()):
        count = activity_counts[label]
        # Sum lengths of all sequences with this label
        total_points = sum(ds.length for ds in dataset_list if ds.label == label)
        logging.info(f"  Activity {label}: {count} sequences, {total_points} data points")
    
    return dataset_list
