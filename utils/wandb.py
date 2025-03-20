import os
import shutil

def manage_wandb_logs(wandb_dir='wandb', max_runs=5):
    if not os.path.exists(wandb_dir):
        return

    # Get list of run directories, sorted by modification time (oldest first)
    run_dirs = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d)) and d.startswith('run-')]
    run_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(wandb_dir, d)))

    # If exceeding max_runs, delete oldest ones
    if len(run_dirs) > max_runs:
        dirs_to_delete = run_dirs[:len(run_dirs) - max_runs]
        for dir_name in dirs_to_delete:
            full_path = os.path.join(wandb_dir, dir_name)
            shutil.rmtree(full_path)
            print(f"[wandb cleanup] Removed: {full_path}")