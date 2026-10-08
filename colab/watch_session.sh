#!/usr/bin/env bash
# Stream a session's /content/session.log: new lines (routine "exit 0" scoring lines dropped), a heartbeat every
# HEARTBEAT seconds even when nothing changes, a warning when the log cannot be fetched repeatedly, and exit as
# soon as the session reports SESSION DONE so the VM can be released at once.
#
#   colab/watch_session.sh <session> <local_copy> [poll_seconds] [heartbeat_seconds]
set -u
SESSION=$1; LOCAL=$2; POLL=${3:-60}; HEARTBEAT=${4:-600}
seen=0; fails=0; last_beat=$(date +%s)
while true; do
  if colab download -s "$SESSION" /content/session.log "$LOCAL" < /dev/null > /dev/null 2>&1; then
    fails=0
    n=$(wc -l < "$LOCAL" | tr -d ' ')
    if [ "$n" -gt "$seen" ]; then
      sed -n "$((seen + 1)),${n}p" "$LOCAL" | grep -v -E ' dev (jevbench_dev|custom_dev2)( \(HF\))?: .* exit 0$' || true
      seen=$n
    fi
    grep -q "SESSION DONE" "$LOCAL" && { echo "WATCH END: session done"; exit 0; }
  else
    fails=$((fails + 1))
    [ $((fails % 5)) -eq 0 ] && echo "WARNING: session log not fetched $fails times in a row"
  fi
  now=$(date +%s)
  if [ $((now - last_beat)) -ge "$HEARTBEAT" ]; then
    train=""
    if colab download -s "$SESSION" /content/trainF.log "$LOCAL.train" < /dev/null > /dev/null 2>&1; then
      train=" | training: $(grep '^{' "$LOCAL.train" | tail -1 | cut -c1-120)"
    fi
    echo "heartbeat $(date -u +%H:%M) | last: $(tail -1 "$LOCAL" 2>/dev/null | cut -c1-120)$train"
    last_beat=$now
  fi
  sleep "$POLL"
done
