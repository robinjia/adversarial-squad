python js2tokens.py $1
wait
th main4.lua -model pointBEMlstm -modelSaved model_BE_1 -input $1&
th main4.lua -model pointBEMlstm -modelSaved model_BE_2 -input $1&
th main4.lua -model pointBEMlstm -modelSaved model_BE_3 -input $1&
th main4.lua -model pointBEMlstm -modelSaved model_BE_4 -input $1&
th main4.lua -model pointBEMlstm -modelSaved model_BE_5 -input $1&
wait
th ensemble.lua
python txt2js.py $1 test_output.txt
