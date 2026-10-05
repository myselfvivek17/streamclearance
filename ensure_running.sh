#!/usr/bin/env bash
# Start StreamClearance if it isn't answering. Used by cron (@reboot + every 5 min).
cd "$(dirname "$0")"
PORT="${SC_PORT:-8430}"
if ! curl -sf -m 5 "http://127.0.0.1:${PORT}/api/status" >/dev/null 2>&1; then
  mkdir -p jobs
  setsid nohup ./run.sh >>jobs/server.log 2>&1 < /dev/null &
  echo "$(date -Is) started" >> jobs/watchdog.log
fi
