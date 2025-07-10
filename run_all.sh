#!/usr/bin/env bash
# run_all.sh

for SEED in $(seq 0 99); do
  echo "=== Running seed=$SEED ==="
  python AlexNet_SEatt.py --seed ${SEED}
done
