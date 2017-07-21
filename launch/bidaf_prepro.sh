#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 train-v1.1.json dev-v1.1.json" 1>&2
  exit 1
fi
cl work "$(cat cl_worksheet.txt)"
train="$1"
dev="$2"
cl run train-v1.1.json:${train} dev-v1.1.json:${dev} :nltk_data :bi-att-flow-dev :glove.6B.100d.txt 'export PYTHONPATH="bi-att-flow-dev:$PYTHONPATH" ; python3 -m squad.prepro --source_dir . -pm --target_dir out --glove_dir .' --request-docker-image minjoon/tf4:latest -n prepro -d "BiDAF prepro, ${train}, ${dev}" --request-cpus 1

