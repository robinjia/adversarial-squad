#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 trainBundle preproBundle machine gpuid [flags]" 1>&2
  exit 1
fi
trainBundle="$1"
preproBundle="$2"
host="$3"
gpuid="$4"
shift
shift
shift
shift
flags="$@"
desc="BiDAF retrain, ${trainBundle}, ${preproBundle}"
if [ -n "${flags}" ]; then
  desc="${desc}, ${flags}"
fi
cl work "$(cat cl_worksheet.txt)"
cl run init:${trainBundle} prepro:${preproBundle} :nltk_data :bi-att-flow-dev :glove.6B.100d.txt 'cp -r init/out out; export PYTHONPATH="bi-att-flow-dev:$PYTHONPATH"; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 -m basic.cli --mode train --len_opt --cluster --data_dir prepro/out --device /gpu:'"${gpuid}"' --device_type gpu --debug '"${flags}" --request-docker-image robinjia/tf-gpu-py3:1.2 -n train -d "${desc}" --request-queue host=${host}

