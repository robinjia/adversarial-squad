#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 trainBundle dataBundle.json" 1>&2
  exit 1
fi
cl work "$(cat cl_worksheet.txt)"
train="$1"
dataset="$2"
#preproBundle="$2"
cl run traindir:${train} data.json:${dataset} :eval_squad.py :nltk_data :bi-att-flow-dev :glove.6B.100d.txt 'ln -s traindir/out; export PYTHONPATH="bi-att-flow-dev:$PYTHONPATH" ; python3 -m squad.prepro --mode single --single_path data.json -pm --target_dir inter_single --glove_dir . ; python3 -m basic.cli --data_dir inter_single --eval_path inter_single/eval.json --nodump_answer --shared_path out/basic/00/shared.json --eval_num_batches 0 --mode forward --batch_size 1 --len_opt --cluster --cpu_opt --load_ema --device_type cpu; python3 -m basic.ensemble --data_path inter_single/data_single.json --shared_path inter_single/shared_single.json -o single-server.json inter_single/eval.json; python2.7 eval_squad.py data.json single-server.json -o eval.json' --request-docker-image robinjia/tf-gpu-py3:1.3 -n run -d "BiDAF Retrained, ${train}, ${dataset}" --request-cpus 4
#cl run prepro:${preproBundle} traindir:${train} :eval_squad.py :nltk_data :save :bi-att-flow-dev :glove.6B.100d.txt 'mkdir data; cp -r prepro/out data/squad; cp -r traindir/out .; export PYTHONPATH="bi-att-flow-dev:$PYTHONPATH" ; python3 -m basic.cli --len_opt --cluster' --request-docker-image minjoon/tf4:latest -n run -d "BiDAF Retrained, ${train}, ${dataset}" --request-cpus 4
