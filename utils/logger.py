import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import logging
import os
from logging.handlers import RotatingFileHandler

import logging
import os
from datetime import datetime

import logging
import os
from datetime import datetime

def setup_logging(base_log_dir='logs', dataset_name='default', mode='none', max_logs=5):
    """
    Set up logging in a per-dataset/mode directory.
    - Directory: logs/{dataset_name}/{mode}/
    - Deletes oldest logs if more than max_logs exist in that directory.
    - Filename: {dataset}_{mode}_YYYYMMDD_HHMMSS.log
    """
    # Create dataset/mode-specific directory
    log_dir = os.path.join(base_log_dir, dataset_name, mode)
    os.makedirs(log_dir, exist_ok=True)

    # Clear previous handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Gather existing logs in this specific dir
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.log')]

    # Sort by modification time and remove oldest if exceeding max_logs
    log_files.sort(key=os.path.getmtime)
    while len(log_files) >= max_logs:
        oldest = log_files.pop(0)
        os.remove(oldest)

    # Create new log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{dataset_name}_{mode}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',  # No timestamp in log lines
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized → {log_path}")
    return log_path



# def setup_logging(log_dir='logs', dataset_name='default'):
#     os.makedirs(log_dir, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     log_filename = f"{dataset_name}_{timestamp}.log"
#     log_path = os.path.join(log_dir, log_filename)

#     logging.basicConfig(
#         level=logging.INFO,
#         #format='[%(asctime)s] %(levelname)s: %(message)s',
#         format='%(levelname)s: %(message)s',
#         handlers=[
#             logging.FileHandler(log_path),
#             logging.StreamHandler()
#         ]
#     )
#     logging.info(f"Logging initialized → {log_path}")
#     return log_path
