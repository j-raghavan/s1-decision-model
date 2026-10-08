"""Notebook cell for a long labeling run that Colab will not reclaim.

Colab reclaims a runtime about 11 minutes after its kernel goes idle (measured;
see COMPUTE_LOG.md), even while a detached process is still working. This cell
therefore keeps the kernel busy: it runs the job in the foreground, then holds
the VM until the Mac signals that the final labels are downloaded by uploading
/content/downloaded.flag (or 30 minutes pass).

    colab exec -s <session> -f colab/launch_label.py

The local exec client may time out on a long run; the cell keeps executing on
the server regardless.
"""

import subprocess
import sys
import time
from pathlib import Path

ROWS = "/content/s1/data/train/all_v1.jsonl"
FLAG = Path("/content/downloaded.flag")

print(subprocess.run("mkdir -p /content/s1 && tar xzf /content/s1_full.tgz -C /content/s1 && wc -l " + ROWS,
                     shell=True, capture_output=True, text=True).stdout, flush=True)
with open("/content/job.log", "w") as log:
    job = subprocess.Popen([sys.executable, "/content/s1/colab/label_job.py", "--rows", ROWS, "--max-minutes", "90"],
                           stdout=log, stderr=subprocess.STDOUT)
    while job.poll() is None:
        time.sleep(30)
print(f"job exited with code {job.returncode}", flush=True)

t0 = time.time()
while not FLAG.exists() and time.time() - t0 < 1800:
    time.sleep(15)
print("download confirmed" if FLAG.exists() else "no download confirmation after 30 min; releasing", flush=True)
