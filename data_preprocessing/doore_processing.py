from glob import glob
import numpy as np
import pandas as pd
from collections import Counter

class TSDataSet:
    def __init__(self, data, label, length):
        self.data = data
        self.label = int(label)
        self.length = int(length)

# Map file name to activity label number
# Activity label mapping: [1: 'Chatting', 2: 'Discussion', 3: 'GroupStudy', 4: 'Presentation', 5: 'NULL']
def label_num(filename):
    label_candidate = ['Chatting', 'Discussion', 'GroupStudy', 'Presentation', 'NULL']
    for i, label in enumerate(label_candidate):
        if label in filename:
            return i + 1  # Activity labels: 1 ~ 5
    return 0  # Default label if no match

# Loader for Doore/Lapras dataset with summary output
# Lapras data format: sensor type, sensor state, start time, end time, duration
# File name indicates activity label
# Example (csv): Seat Occupy,1,1.490317862115E12,1.490319250294E12,23.13...
# Activity numbers: [1, 2, 3, 5, 4] = [Chatting, Discussion, GroupStudy, NULL, Presentation]
# Activity counts:  [119, 52, 40, 116, 129]
def dooreLoader(file_name_pattern, timespan, min_seq):
    print("Loading Doore Dataset --------------------------------------")

    file_list = sorted(glob(file_name_pattern))
    time_list = []  # Store [start_time, end_time] for each file

    # Determine time spans for each file
    for file_path in file_list:
        df = pd.read_csv(file_path, sep=',', header=None).to_numpy()
        if df.size == 0:
            continue

        start_time = np.min(df[:, 2])  # Start time of first instance
        end_time = np.max(df[:, 3])    # End time of last instance
        time_list.append([start_time, end_time])

    # Fixed sensor list used for indexing
    sensor_list = ['Seat Occupy', 'Sound', 'Brightness', 'Light', 'Existence', 'Projector', 'Presentation']

    # Output dataset list and total timestamp count
    dataset_list = []
    total_timestamps = 0  # Total number of timestamps (time slots) across all sequences

    # For each file
    for file_idx, file_path in enumerate(file_list):
        print(f"Processing: {file_path}")
        df = pd.read_csv(file_path, sep=',', header=None).to_numpy()
        if df.size == 0:
            continue

        start_time, end_time = time_list[file_idx]
        seq_length = int((end_time - start_time) / timespan)  # Total number of timestamps for this instance
        temp_dataset = np.zeros((seq_length, len(sensor_list)))  # Initialize empty time-series array

        # If activity duration is sufficiently long
        if seq_length > min_seq:
            # For each row in the file
            for row in df:
                sensor_type, state, row_start, row_end = row[0], int(row[1]), row[2], row[3]
                if sensor_type not in sensor_list:
                    continue
                sensor_idx = sensor_list.index(sensor_type)

                # Determine time index range for this sensor event
                start_idx = int((row_start - start_time) / timespan)
                end_idx = int((row_end - start_time) / timespan)

                # Fill in sensor data over the corresponding timestamps
                for t in range(start_idx, end_idx):
                    if 0 <= t < seq_length:
                        if sensor_type in ['Sound', 'Brightness']:  # Environment-driven sensors
                            temp_dataset[t][sensor_idx] = state % 10
                        else:  # User-driven or actuator-driven events
                            temp_dataset[t][sensor_idx] += 1

            # Append this instance as a TSDataSet object
            label = label_num(file_path)
            dataset_list.append(TSDataSet(temp_dataset, label, seq_length))
            total_timestamps += seq_length  # Count all timestamps, regardless of filled or not

    # Summarize dataset statistics
    sensor_channels = len(sensor_list)
    label_list = [ds.label for ds in dataset_list]
    activity_counts = Counter(label_list)
    num_activity_types = len(activity_counts)
    
    print("Loading Doore Dataset Finished --------------------------------------")
    print("====== Dataset Summary ======")
    print(f"Sensor channels: {sensor_channels}")
    print(f"Total timestamps (data points): {total_timestamps}")
    print(f"Number of activity types: {num_activity_types}")
    print("Activity sequence counts and data points:")
    for label in sorted(activity_counts.keys()):
        count = activity_counts[label]
        # Sum lengths of all sequences with this label
        total_points = sum(ds.length for ds in dataset_list if ds.label == label)
        print(f"  Activity {label}: {count} sequences, {total_points} data points")

    count_num = 20  # Minimum number of sequences for an activity
    filtered_labels = [label for label, count in activity_counts.items() if count >= count_num]

    total_sequences_filtered = sum(activity_counts[label] for label in filtered_labels)
    total_pointers_filtered = sum(ds.length for ds in dataset_list if ds.label in filtered_labels)

    print(f"====== Activities with ≥ {count_num} sequences ======")
    print(f"Number of such activities: {len(filtered_labels)}")
    print(f"Total sequences in these activities: {total_sequences_filtered}")
    print(f"Total data points in these activities: {total_pointers_filtered}")
    for label in sorted(filtered_labels):
        count = activity_counts[label]
        total_points = sum(ds.length for ds in dataset_list if ds.label == label)
        print(f"  Activity {label}: {count} sequences, {total_points} data points")


    return dataset_list


    return dataset_list

# # Example usage
# dataset = dooreLoader('../data/doore/*.csv', timespan=10000, min_seq=10)
# # Inspect the first sequence
# first_sequence = dataset[0]
# print(f"First sequence shape: {first_sequence.data.shape}")
# print(f"First sequence label: {first_sequence.label}")
# print(f"First sequence length: {first_sequence.length}")