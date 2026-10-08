#!/usr/bin/env bash
# Upload files of any size to a Colab session and verify them.
#
# The CLI's upload goes through the Jupyter contents API, which returns HTTP 500 for files above
# roughly 60 MB and for destinations whose folder does not exist. This splits each file into 40 MB
# parts in /content, reassembles them on the VM and checks SHA-256 before deleting the parts.
#
#   scripts/colab_upload.sh <session> <local_file>[:<remote_name>] ...
set -euo pipefail
session=$1; shift
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
: > "$work/sums.txt"
names=()
for spec in "$@"; do
  src=${spec%%:*}; name=${spec#*:}; [ "$name" = "$spec" ] && name=$(basename "$src")
  (cd "$work" && split -b 40m "$OLDPWD/$src" "$name.part.")
  (cd "$(dirname "$src")" && shasum -a 256 "$(basename "$src")") | awk -v n="$name" '{print $1"  "n}' >> "$work/sums.txt"
  names+=("$name")
done
for f in "$work"/*.part.* "$work/sums.txt"; do
  for attempt in 1 2 3; do
    colab upload -s "$session" "$f" "/content/$(basename "$f")" < /dev/null 2>&1 | grep -q Uploaded && break
    [ "$attempt" = 3 ] && { echo "upload failed: $(basename "$f")" >&2; exit 1; }
    sleep 3
  done
done
cmd="cd /content"
for n in "${names[@]}"; do cmd="$cmd && cat $n.part.* > $n"; done
cmd="$cmd && sha256sum -c sums.txt && rm -f *.part.* sums.txt"
CMD="$cmd" python3 -c 'import json, os
print("import subprocess\nr = subprocess.run(" + json.dumps(os.environ["CMD"]) + ", shell=True, capture_output=True, text=True)\nprint(r.stdout, r.stderr)\nraise SystemExit(r.returncode)")' > "$work/assemble.py"
colab exec -s "$session" -f "$work/assemble.py" < /dev/null
