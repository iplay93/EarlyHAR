from glob import glob
import numpy as np
import pandas as pd
from collections import Counter

class TSDataSet:
    def __init__(self, data, label, length):
        self.data = data
        self.label = int(label)
        self.length = int(length)

# Loader for CASAS dataset (no timespan used)
# CASAS data format: timestamp, sensor type+context name, state, user #, activity label [1–15]
# File name = day (e.g., 2008-11-10)
# Example: 2008-11-10 14:28:17.986759 M22 ON 2 2
def casasLoader(file_name_pattern, min_seq):
    print("Loading CASAS Dataset --------------------------------------")

    file_list = sorted(glob(file_name_pattern))

    # Fixed sensor list (from CASAS metadata)
    sensor_list = ['M19', 'M23', 'M18', 'M01', 'M17', 'D07', 'M21', 'M22', 'M03', 'I04',
                   'D12', 'I06', 'M26', 'M04', 'M02', 'M07', 'M08', 'M09', 'M14', 'M15', 
                   'M16', 'M06', 'M10', 'M11', 'M51', 'D11', 'M13', 'M12', 'D14', 'D13', 
                   'D10', 'M05', 'D09', 'D15', 'M20', 'M25', 'M24']

    dataset_list = []
    total_data_pointers = 0  # Total rows across all sequences

    # Process each resident (0: resident 1, 1: resident 2)
    for rid in range(2):
        for file_path in file_list:
            df = pd.read_csv(file_path, sep='\t', header=None).to_numpy()
            print(file_path)

            if len(df) == 0:
                continue

            current_label = 0
            activity_list = np.zeros(len(sensor_list))
            temp_dataset = np.array([activity_list])

            for i in range(len(df)):
                temp_list = df[i, 3].split(" ")

                if (len(temp_list) == 3 and int(temp_list[1]) - 1 == rid) or \
                   (len(temp_list) > 3 and (int(temp_list[1]) - 1 == rid or int(temp_list[3]) - 1 == rid)):

                    # First row for this activity
                    if current_label == 0:
                        if int(temp_list[1]) - 1 == rid:
                            current_label = int(temp_list[2])
                        else:
                            current_label = int(temp_list[4])

                        activity_list[sensor_list.index(df[i, 2])] = 1 if temp_list[0] in ['ON', 'OPEN', 'PRESENT'] else 0
                        temp_dataset = np.array([activity_list])

                    # Continuing the same activity
                    if (int(temp_list[2]) == current_label) or \
                       (len(temp_list) > 3 and int(temp_list[4]) == current_label):
                        activity_list[sensor_list.index(df[i, 2])] = 1 if temp_list[0] in ['ON', 'OPEN', 'PRESENT'] else 0
                        temp_dataset = np.concatenate((temp_dataset, [activity_list]), axis=0)

                    # Activity changed
                    else:
                        if len(temp_dataset) > min_seq:
                            dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                            total_data_pointers += len(temp_dataset)

                        activity_list = np.zeros(len(sensor_list))
                        activity_list[sensor_list.index(df[i, 2])] = 1 if temp_list[0] in ['ON', 'OPEN', 'PRESENT'] else 0
                        temp_dataset = np.array([activity_list])

                        if int(temp_list[1]) - 1 == rid:
                            current_label = int(temp_list[2])
                        else:
                            current_label = int(temp_list[4])

            # Final activity segment
            if len(temp_dataset) > min_seq:
                dataset_list.append(TSDataSet(temp_dataset, current_label, len(temp_dataset)))
                total_data_pointers += len(temp_dataset)

    # Summary
    sensor_channels = len(sensor_list)
    label_list = [ds.label for ds in dataset_list]
    activity_counts = Counter(label_list)
    num_activity_types = len(activity_counts)
    total_activities = len(dataset_list)
    
    print("Loading CASAS Dataset Finished --------------------------------------")
    print("====== Dataset Summary ======")
    print(f"Sensor channels: {sensor_channels}")
    print(f"Total data points: {total_data_pointers}")
    print(f"Number of activity types: {num_activity_types}")
    print(f"Total activities (sequences): {total_activities}")
    print("Activity sequence counts:")
    for label, count in sorted(activity_counts.items()):
        print(f"  Activity {label}: {count} sequences")

    return dataset_list


dataset = casasLoader('../data/casas/*.txt', min_seq=5)