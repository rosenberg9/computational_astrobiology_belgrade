import os
import time
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

def simulated_work(task_name):
    """Worker function: simulates a PSLS run."""
    duration = random.uniform(0.5, 1.5)
    time.sleep(duration)
    # Simulate a 10% failure rate
    success = random.random() > 0.1
    return task_name, success

if __name__ == '__main__':
    TOTAL_TASKS = 40
    WORKERS = 4
    
    all_tasks = [f"sys_{i:04d}" for i in range(TOTAL_TASKS)]
    
    print(f"Main Process: Dispatching {TOTAL_TASKS} tasks across {WORKERS} workers.")
    print("Monitoring progress manually (No tqdm)...\n")

    completed = 0
    success_count = 0
    failures = []

    # Using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        # submit all tasks
        futures = {executor.submit(simulated_work, task): task for task in all_tasks}

        # As each task finishes, update the manual bar
        for future in as_completed(futures):
            task_name, ok = future.result()
            completed += 1
            
            if ok:
                success_count += 1
            else:
                failures.append(task_name)

            # --- MANUAL PROGRESS BAR LOGIC ---
            percent = (completed / TOTAL_TASKS) * 100
            bar_length = 40
            filled_length = int(bar_length * completed // TOTAL_TASKS)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # \r moves the cursor back to the start of the line
            # sys.stdout.write ensures immediate printing without buffering
            sys.stdout.write(f'\rProgress: |{bar}| {percent:.1f}% ({completed}/{TOTAL_TASKS})')
            sys.stdout.flush()

    # Final summary
    print(f"\n\nTest Complete!")
    print(f"Successful: {success_count}/{TOTAL_TASKS}")
    if failures:
        print(f"Failures: {', '.join(failures)}")