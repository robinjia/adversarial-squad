#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 <mode> dataBundle.json <num_additions> <sample_num> <num_particles> <num_nearby> <num_epochs>" 1>&2
  exit 1
fi
cl work "$(cat cl_worksheet.txt)"
mode="$1"
dataset="$2"
num_additions="$3"
sample_num="$4"
num_particles="$5"
num_nearby="$6"
num_epochs="$7"
cl run data.json:${dataset} nearby.json:nearby_n100_glove_6B_100d.json :src-addwords :nltk_data english.txt:brown_english_1k.txt :glove_shortlists :eval_squad.py :save :bi-att-flow-dev :glove.6B.100d.txt "python src-addwords/adversarial_squad.py ${mode} bidaf-ensemble data.json english.txt -n ${num_additions} -k ${sample_num} -p ${num_particles} -q ${num_nearby} -T ${num_epochs} --nearby-file nearby.json -o adversarial_data.json --plot-file eval.png;"' export PYTHONPATH="bi-att-flow-dev:$PYTHONPATH" ; bi-att-flow-dev/basic/run_ensemble.sh adversarial_data.json ensemble-server.json; python eval_squad.py adversarial_data.json ensemble-server.json -o eval.json' --request-docker-image robinjia/minjoon-tf4:v1 -n adv -d "${mode}, BiDAF ensemble, n = ${num_additions}, k = ${sample_num}, p = ${num_particles}, q = ${num_nearby}, T = ${num_epochs}, ${dataset}" --request-cpus 4 --request-queue john
# ; rm -rf adversarial_squad_out
