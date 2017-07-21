require 'torch'
require 'nngraph'
require 'optim'
require 'debug'
require 'sys'
require 'nn'

transition = {}
include 'utils.lua'

torch.setdefaulttensortype('torch.FloatTensor')
include 'Embedding.lua'
include 'LSTM2.lua'
include 'LSTMwwatten2.lua'
include 'pointNet4.lua'
include 'pointBEMlstm2.lua'
include 'bpointBEMlstm.lua'

include 'CAddRepTable.lua'

print ("require done !")

cmd = torch.CmdLine()

cmd:option('-model','bpointBEMlstm','model')
cmd:option('-input','dev-v1.1.json1','input dataset')
cmd:option('-modelSaved', 'model_bpoint','saved model name')

local opt = cmd:parse(arg)
torch.setnumthreads(1)

ivocab = torch.load("ivocab_t.t7")
opt.numWords = #ivocab

local model_class = transition[opt.model]
local model = model_class(opt)
model:load(opt.modelSaved)
if opt.model == 'bpointBEMlstm' then
    sys.execute('python js2tokens.py '..opt.input)
end
local data_test, vocab_new, ivocab_new, emb_new = unpack(load_data())
ivocab = ivocab_new
model.emb_vecs.weight = emb_new
if opt.model == 'pointBEMlstm' then
    model.idx = tonumber(opt.modelSaved:sub(-1))
end
local output_str = model:predict_dataset(data_test)
if opt.model == 'bpointBEMlstm' then
    sys.execute('python txt2js.py '..opt.input..' test_output.txt')
end
--local res = sys.execute('python evaluator.py '..opt.input.. ' prediction.json')
--print(res)
