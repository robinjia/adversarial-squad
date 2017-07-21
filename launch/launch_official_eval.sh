#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 dataset.json predictions.json" 1>&2
  exit 1
fi
cl work "$(cat cl_worksheet.txt)"
cl run :evaluate-v1.1.py dataset.json:$1 predictions.json:$2 'python evaluate-v1.1.py dataset.json predictions.json > eval.json' -n eval -d "Eval $2 on $1"
