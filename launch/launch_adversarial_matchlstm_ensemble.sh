#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 <mode> dataBundle.json <num_additions> <sample_num> <num_particles> <num_nearby> <num_epochs> (johnN)" 1>&2
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
if [ -z "${8-}" ]
then
  queue="john"
else
  queue="host=john${8}"
fi
nearby_file="nearby_n100_glove_6B_100d.json"  # Smaller vectors, used by BiDAF
#nearby_file="nearby_n100_glove_840B_300d.json"  # Larger vectors, used by Match-LSTM
cl run data.json:${dataset} nearby.json:${nearby_file} :src-addwords :nltk_data english.txt:brown_english_1k.txt :glove_shortlists :eval_squad.py :matchlstm :ivocab_t.t7 :vocab_t.t7 :initEmb_t.t7 :js2tokens.py :txt2js.py :model_BE_1 :model_BE_2 :model_BE_3 :model_BE_4 :model_BE_5 :model_bpoint :ensemble.lua :utils.lua :Embedding.lua :CAddRepTable.lua :LSTM2.lua :LSTMwwatten2.lua :pointNet4.lua :pointBEMlstm2.lua :bpointBEMlstm.lua :main4.lua :bash.sh "python src-addwords/adversarial_squad.py ${mode} matchlstm-ensemble data.json english.txt -n ${num_additions} -k ${sample_num} -p ${num_particles} -q ${num_nearby} -T ${num_epochs} --nearby-file nearby.json -o adversarial_data.json --plot-file eval.png; sh bash.sh adversarial_data.json; python eval_squad.py adversarial_data.json prediction.json -o eval.json; rm -rf adversarial_squad_out" --request-docker-image robinjia/shuohang-torch-1.8:v1.1 -n adv -d "${mode}, MatchLSTM Ensemble, n = ${num_additions}, k = ${sample_num}, p = ${num_particles}, q = ${num_nearby}, T = ${num_epochs}, ${dataset}" --request-cpus 1 --request-queue ${queue}
