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

        current_label = [df[0, 20], df[0, 21]]
        temp_segment = [df[0, 0:20]]
        current_data = df[0, 0:20]
        activity_segments = []

        for i in range(1, len(df)):
            label_r1, label_r2 = df[i, 20], df[i, 21]
            sensor_data = df[i, 0:20]
            new_label = [label_r1, label_r2]

            if current_label == new_label:
                if (current_data != sensor_data).any():
                    temp_segment.append(sensor_data)
                    current_data = sensor_data
            else:
                # Identify which label changed
                changed_label = None
                if current_label[0] != new_label[0]:
                    changed_label = new_label[0]
                elif current_label[1] != new_label[1]:
                    changed_label = new_label[1]

                if len(temp_segment) >= min_seq and changed_label is not None:
                    activity_segments.append((np.array(temp_segment), changed_label))

                # Reset
                temp_segment = [sensor_data]
                current_data = sensor_data
                current_label = new_label

        # Final segment
        if len(temp_segment) >= min_seq:
            # No new label, but still need to check what changed
            final_label = current_label[0] if current_label[0] != 0 else current_label[1]
            if final_label != 0:
                activity_segments.append((np.array(temp_segment), final_label))

        # Downsampling
        window_size = int(timespan / 1000)
        for segment_data, label in activity_segments:
            downsampled_sequence = []
            for idx in range(0, len(segment_data), window_size):
                window = segment_data[idx:idx+window_size]
                if len(window) > 0:
                    avg_vector = np.mean(window, axis=0)
                    downsampled_sequence.append(avg_vector)

            if len(downsampled_sequence) == 0:
                continue

            downsampled_sequence = np.stack(downsampled_sequence)
            downsampled_length = len(downsampled_sequence)
            total_data_pointers += downsampled_length

            dataset_list.append(TSDataSet(downsampled_sequence, label, downsampled_length))

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
        total_points = sum(ds.length for ds in dataset_list if ds.label == label)
        logging.info(f"  Activity {label}: {count} sequences, {total_points} data points")

    return dataset_list
