from glob import glob
import numpy as np
import pandas as pd
from collections import Counter

class TSDataSet:
    def __init__(self, data, label, length):
        self.data = data
        self.label = int(label)
        self.length = int(length)

# Opportunity data format: sensor type + context name ..., activity label (ADL)
# File name: each user (4) * 5
# Example (txt): 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 13 17
# The number of examples: 101(Relaxing)-40, 102(Coffee time)-20, 103(Early morning)-20, 104(Cleanup)-20, 105(Sandwich time)-20
# Activity numbers: [1, 3, 2, 5, 4]
# Activity counts:  [40, 20, 20, 20, 20]
def opportunityLoader(file_name_pattern, timespan, min_seq):
    print("Loading Opportunity Dataset (Averaging over timespan) ---------------------")

    # Initialization
    file_list = []           # List to store file names
    dataset_list = []        # Output list of TSDataSet objects
    total_raw_pointers = 0   # Total number of raw data rows before averaging

    # Extract and sort file names
    for x in glob(file_name_pattern):
        file_list.append(x)
    file_list.sort()

    # For each file
    for file_path in file_list:
        print(f"Processing: {file_path}")
        temp_df = pd.read_csv(file_path, sep=' ', header=None)

        # Extract rows with target ADLs (column 244 => 101~105), convert to numpy array
        temp_df = temp_df[temp_df[244] > 100].to_numpy()
        total_raw_pointers += len(temp_df)

        # If at least one ADL exists in the file
        if len(temp_df) > 0:
            current_label = temp_df[0, 244]     # Column 244 is the activity label
            current_time = temp_df[0, 0]        # Column 0 is the timestamp
            temp_buffer = [temp_df[0, 1:243]]   # Columns 1–242 are sensor data
            averaged_sequence = []

            # For each row
            for i in range(1, len(temp_df)):
                row_time = temp_df[i, 0]
                row_label = temp_df[i, 244]
                row_sensor = temp_df[i, 1:243]

                # Within the same timespan
                if (row_time - current_time) < timespan and row_label == current_label:
                    # Accumulate sensor data in buffer
                    temp_buffer.append(row_sensor)
                else:
                    # Compute average for current timespan window
                    avg_vector = np.mean(temp_buffer, axis=0)
                    averaged_sequence.append(avg_vector)

                    # Reset buffer
                    temp_buffer = [row_sensor]
                    current_time = row_time

                    # If the activity label has changed
                    if row_label != current_label:
                        # Construct TSDataSet object for the previous activity
                        if len(averaged_sequence) >= min_seq:
                            seq_array = np.stack(averaged_sequence)
                            dataset_list.append(TSDataSet(seq_array, current_label - 100, len(averaged_sequence)))
                        averaged_sequence = []
                        current_label = row_label

            # For the last activity segment
            if temp_buffer:
                avg_vector = np.mean(temp_buffer, axis=0)
                averaged_sequence.append(avg_vector)

            if len(averaged_sequence) >= min_seq:
                seq_array = np.stack(averaged_sequence)
                dataset_list.append(TSDataSet(seq_array, current_label - 100, len(averaged_sequence)))

    # After processing all files, compute and print summary
    total_averaged_pointers = sum(seq.length for seq in dataset_list)
    sensor_channels = dataset_list[0].data.shape[1] if dataset_list else 0
    label_list = [seq.label for seq in dataset_list]
    activity_counts = Counter(label_list)
    num_activity_types = len(activity_counts)

    print("Loading Opportunity Dataset Finished --------------------------------------")
    print("====== Dataset Summary ======")
    print(f"Sensor channels: {sensor_channels}")
    print(f"Raw data points (before averaging): {total_raw_pointers}")
    print(f"Averaged data points (after averaging): {total_averaged_pointers}")
    print(f"Number of activity types: {num_activity_types}")
    print("Activity sequence counts:")
    for label, count in sorted(activity_counts.items()):
        print(f"  Activity {label}: {count} sequences")

    return dataset_list

        
dataset = opportunityLoader('../data/opportunity/*.dat', timespan=1000, min_seq=5)

print(f"Total sequences: {len(dataset)}")
print(f"First sequence shape: {dataset[0].data.shape}")
print(f"First sequence label: {dataset[0].label}")
