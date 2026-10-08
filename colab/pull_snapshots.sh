#!/usr/bin/env bash
# Pull full-run adapter snapshots off the VM as they appear, so a lost VM can resume from the latest one
# (python -m s1.train_decoder --init-adapter <snapshot> ...). Run from the Mac while the session trains.
# A download is kept only if the safetensors file is complete (header length + data size == file size), so a
# pull that races the trainer's write is retried rather than kept truncated.
#
#   colab/pull_snapshots.sh <session> <local_dir> [interval_seconds] [max_hours]
set -u
SESSION=$1; DEST=$2; INTERVAL=${3:-120}; MAX_HOURS=${4:-7}
mkdir -p "$DEST"
trap 'rm -rf "$DEST"/*.part' EXIT
complete() {  # exit 0 when the safetensors file is whole
  python3 - "$1" <<'PY'
import json, os, struct, sys
p = sys.argv[1]
with open(p, "rb") as f:
    head = f.read(8)
    if len(head) < 8: sys.exit(1)
    n = struct.unpack("<Q", head)[0]
    meta = json.loads(f.read(n))
end = max(v["data_offsets"][1] for k, v in meta.items() if k != "__metadata__")
sys.exit(0 if os.path.getsize(p) == 8 + n + end else 1)
PY
}
START=$(date +%s)
while true; do
  for name in step-500 step-1000 step-1500 step-2000 step-2500 step-3000 step-3500 step-4000 step-4500 last; do
    [ -s "$DEST/$name/adapter_model.safetensors" ] && continue
    mkdir -p "$DEST/$name.part"
    if colab download -s "$SESSION" "/content/adapterF/$name/adapter_config.json" "$DEST/$name.part/adapter_config.json" < /dev/null > /dev/null 2>&1 \
       && colab download -s "$SESSION" "/content/adapterF/$name/adapter_model.safetensors" "$DEST/$name.part/adapter_model.safetensors" < /dev/null > /dev/null 2>&1 \
       && complete "$DEST/$name.part/adapter_model.safetensors" 2>/dev/null; then
      mv "$DEST/$name.part" "$DEST/$name"
      echo "$(date -u +%H:%M:%S) pulled $name ($(du -h "$DEST/$name/adapter_model.safetensors" | cut -f1), verified complete)"
    else
      rm -rf "$DEST/$name.part"
    fi
  done
  [ -s "$DEST/last/adapter_model.safetensors" ] && { echo "all snapshots pulled"; break; }
  [ $(( $(date +%s) - START )) -gt $(( MAX_HOURS * 3600 )) ] && { echo "stopping after ${MAX_HOURS}h without the final adapter"; break; }
  sleep "$INTERVAL"
done
