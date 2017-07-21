#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 dataBundle.json (johnN)" 1>&2
  exit 1
fi
dataset="$1"
if [ -z "${2-}" ]
then
  queue="john"
else
  queue="host=john${2}"
fi
cl work "$(cat cl_worksheet.txt)"
cl run data.json:${dataset} :eval_squad.py :ivocab_t.t7 :vocab_t.t7 :initEmb_t.t7 :js2tokens.py :txt2js.py :model_BE_1 :model_BE_2 :model_BE_3 :model_BE_4 :model_BE_5 :model_bpoint :ensemble.lua :utils.lua :Embedding.lua :CAddRepTable.lua :LSTM2.lua :LSTMwwatten2.lua :pointNet4.lua :pointBEMlstm2.lua :bpointBEMlstm.lua :main4.lua :bash.sh 'th main4.lua -input data.json; python eval_squad.py data.json prediction.json -o eval.json'  --request-docker-image shuohang/torch:1.8 -n run -d "Match-LSTM w/ Bi-Ans-Ptr Boundary Single, ${dataset}" --request-cpus 1 --request-queue ${queue}
