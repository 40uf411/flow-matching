#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="/export/home/aaouf/workspace/images/datasets_3d"   # <-- parent folder containing subfolders
OUT_ROOT="${CODE_ROOT}/outputs_3d"
LOG_ROOT="${CODE_ROOT}/logs_3d"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

DATASETS=(
  #"boom_clay_bin"
  #"illite_bin"
  #"illite_3phases"
  #"bentheimer_bin"
  #"banderabrown_bin"
  "savonnieres_bin"
)

for ds in "${DATASETS[@]}"; do
  log="${LOG_ROOT}/${ds}.out"

  echo "▶ Training ${ds}"
  "${PYTHON_BIN}" "${CODE_ROOT}/train_flow_matching_on_image_3D.py" \
    --dataset "${ds}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUT_ROOT}" \
    --exp "fm_${ds}" \
    --batch_size 2 \
    --n_epochs 300 \
    --image_size 64 \
    --eval_every 50 \
    --save_every 50 \
    2>&1 | tee "${log}"
done

