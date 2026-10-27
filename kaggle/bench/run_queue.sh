#!/usr/bin/env bash
# Push queued GPU kernels as slots free up (Kaggle allows 2 concurrent GPU batch sessions),
# wait for every prediction kernel, then run the analysis kernel and download its output.
# Usage: ./run_queue.sh [dataset ...]   (datasets whose GPU kernel still needs pushing)
cd "$(dirname "$0")"
# every prediction kernel the analysis reads
SOURCES=$(python3 -c "import json; print(' '.join(s.split('/')[1].removeprefix('churn-bench-') for s in json.load(open('analysis-kernel/kernel-metadata.json'))['kernel_sources']))")
queue=("$@")

state() { kaggle kernels status "ipolas/churn-bench-$1" 2>&1 | grep -v outdated; }
done_state() { grep -q -i -E 'complete|error|cancel'; }

while ((${#queue[@]})); do
  out=$(kaggle kernels push -p "kernels/${queue[0]}-gpu" 2>&1 | grep -v outdated)
  if grep -q "successfully pushed" <<< "$out"; then echo "pushed ${queue[0]}-gpu"; queue=("${queue[@]:1}"); sleep 60
  else sleep 120; fi
done

for k in $SOURCES; do
  until s=$(state "$k"); done_state <<< "$s"; do sleep 120; done
  echo "$s"
done

cp analyze.py analysis-kernel/analyze.py
kaggle kernels push -p analysis-kernel 2>&1 | grep -v outdated
sleep 60
until s=$(state analysis); done_state <<< "$s"; do sleep 60; done
echo "$s"
out=${OUT_DIR:-out/analysis}
rm -rf "$out" && mkdir -p "$out" && kaggle kernels output ipolas/churn-bench-analysis -p "$out" 2>&1 | grep -v outdated | tail -2
