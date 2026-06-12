import os
import subprocess
import pandas as pd
import sys
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
PSLS_EXEC   = os.path.abspath("../../psls-1.9/psls.py")
CONFIG_DIR  = os.path.abspath("configs")
OUTPUT_ROOT = os.path.abspath("outputs")
NUM_SYSTEMS = 1_500

# --- Load ---
#df = pd.read_csv("paired_simulation_labels.csv")
df = pd.read_csv("../../data/input_labels/paired_simulation_labels_v2.csv")

system_ids = df['system_id'].head(NUM_SYSTEMS).tolist()

print(f"Running {len(system_ids)} systems sequentially...")
print(f"Python: {sys.executable}")
print(f"PSLS:   {PSLS_EXEC}\n")

success_count = 0

with tqdm(total=len(system_ids), desc="Simulating", unit="sys") as pbar:
    for sys_id in system_ids:
        yaml_name = f"{sys_id}.yaml"
        yaml_path = os.path.join(CONFIG_DIR, yaml_name)
        out_dir   = os.path.join(OUTPUT_ROOT, sys_id)

        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(yaml_path):
            tqdm.write(f"  ✗  {yaml_name} — config missing")
            pbar.update(1)
            continue

        try:
            subprocess.run(
                [sys.executable, PSLS_EXEC, yaml_path],
                cwd=out_dir,
                capture_output=True,
                text=True,
                check=True
            )
            tqdm.write(f"  ✓  {yaml_name}")
            success_count += 1

        except subprocess.CalledProcessError as e:
            stdout = (e.stdout or "").strip()
            stderr = (e.stderr or "").strip()
            full_log = "\n".join(part for part in (stdout, stderr) if part)
            log_path = Path(out_dir) / "psls_error.log"
            log_path.write_text(full_log + "\n")

            lines = [line for line in full_log.splitlines() if line.strip()]
            summary = lines[-1] if lines else f"exit code {e.returncode}"
            tqdm.write(f"  ✗  {yaml_name} — {summary} (full log: {log_path})")
        except Exception as e:
            tqdm.write(f"  ✗  {yaml_name} — {e}")

        pbar.update(1)

print("-" * 40)
print(f"Done. {success_count}/{len(system_ids)} successful.")
print(f"Outputs in {OUTPUT_ROOT}")
