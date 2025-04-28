#!/bin/bash

pid=2999
echo "Watching for PID $pid to exit…"

# 프로세스가 살아 있는 동안 루프
while kill -0 "$pid" 2>/dev/null; do
  sleep 360    # 360초마다 체크
done

echo "Process $pid has exited."
bash next_pre.sh
