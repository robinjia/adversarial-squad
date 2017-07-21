#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 preproBundle machine gpuid [flags]" 1>&2
  exit 1
fi
preproBundle="$1"
host="$2"
gpuid="$3"
shift
shift
shift
flags="$@"
desc="BiDAF train, ${preproBundle}"
if [ -n "${flags}" ]; then
  desc="${desc}, ${flags}"
fi
cl work "$(cat cl_worksheet.txt)"
cl run prepro:${preproBundle} :nltk_data :bi-att-flow-dev :glove.6B.100d.txt 'export PYTHONPATH="bi-att-flow-dev:$PYTHONPATH"; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 -m basic.cli --mode train --noload --len_opt --cluster --data_dir prepro/out --device /gpu:'"${gpuid}"' --device_type gpu --debug '"${flags}" --request-docker-image robinjia/tf-gpu-py3:1.2 -n train -d "${desc}" --request-queue host=${host}

