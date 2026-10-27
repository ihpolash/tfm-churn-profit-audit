#!/usr/bin/env bash
# Generate one Kaggle kernel folder per (dataset, group) from predict.py, then optionally push them.
# Usage: ./make_kernels.sh [push] [dataset ...]   (also: timing, <dataset>-context)
set -euo pipefail
cd "$(dirname "$0")"

source_for() {
  case "$1" in
    telco) echo '"blastchar/telco-customer-churn"' ;;
    maven) echo '"shilongzhuang/telecom-customer-churn-by-maven-analytics"' ;;
    cell2cell) echo '"jpacse/datasets-for-churn-telecom"' ;;
    iranian) echo '"royjafari/customer-churn"' ;;
    ecommerce) echo '"ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"' ;;  # replication check only
    orange) echo '' ;;  # fetched from OpenML (data_id 1112)
    all) for d in telco maven cell2cell iranian; do source_for $d; done | paste -sd, - ;;
  esac
}

push=""; [[ "${1:-}" == "push" ]] && { push=1; shift; }
datasets=("${@:-telco maven cell2cell iranian orange}")
read -r -a datasets <<< "${datasets[*]}"

for ds in "${datasets[@]}"; do
  groups="cpu gpu"
  if [[ $ds == timing ]]; then  # one GPU kernel that benchmarks scoring throughput on all datasets
    groups="timing"; ds="all"
  fi
  if [[ $ds == *-context ]]; then  # context-construction ablation, e.g. orange-context
    groups="context"; ds="${ds%-context}"
  fi
  for group in $groups; do
    dir="kernels/$ds-$group"; mkdir -p "$dir"
    sed -e "s/__DATASET__/$ds/" -e "s/__GROUP__/$group/" predict.py > "$dir/predict.py"
    src=$(source_for "$ds")
    if [[ $group != cpu ]]; then
      src="${src:+$src, }\"ipolas/tabpfn-token\""; gpu=true; shape='"machine_shape": "NvidiaTeslaT4",'
    else
      gpu=false; shape=''
    fi
    cat > "$dir/kernel-metadata.json" <<EOF
{
  "id": "ipolas/churn-bench-$ds-$group",
  "title": "churn-bench-$ds-$group",
  "code_file": "predict.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": $gpu,
  "enable_tpu": false,
  "enable_internet": true,
  $shape
  "keywords": [],
  "dataset_sources": [$src],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
EOF
    if [[ -n $push ]]; then kaggle kernels push -p "$dir" 2>&1 | grep -v outdated; fi
  done
done
