local pointBEMlstm = torch.class('transition.pointBEMlstm')

function pointBEMlstm:__init(config)
    self.mem_dim       = config.mem_dim       or 100
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 25
    self.num_layers    = config.num_layers    or 1
    self.reg           = config.reg           or 1e-4
    self.lstmModel     = config.lstmModel     or 'lstm'
    self.sim_nhidden   = config.sim_nhidden   or 50
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'squad'
    self.numWords      = config.numWords
    self.maxsenLen     = config.maxsenLen     or 50
    self.dropoutP      = config.dropoutP      or 0
    self.grad          = config.grad          or 'adagrad'
    self.visualize     = false

    self.emb_vecs = Embedding(self.numWords, self.emb_dim)


    self.dropoutl = nn.Dropout(self.dropoutP)
    self.dropoutr = nn.Dropout(self.dropoutP)

    self.optim_state = { learningRate = self.learning_rate }


    local lstm_config = {in_dim = self.emb_dim,mem_dim = self.mem_dim}


    self.llstm = transition.LSTM(lstm_config)
    self.rlstm = transition.LSTM(lstm_config)
    local wwatten_config = {att_dim = self.att_dim, mem_dim = self.mem_dim}
    self.att_module = transition.LSTMwwatten(wwatten_config)
    self.att_module_b = transition.LSTMwwatten(wwatten_config)

    self.point_module = transition.pointNet({mem_dim = self.mem_dim})

    local modules = nn.Parallel()
        :add(self.llstm)
        :add(self.att_module)
        :add(self.point_module)
        self.params, self.grad_params = modules:getParameters()

    share_params(self.rlstm, self.llstm)
    share_params(self.att_module_b, self.att_module)

end


function pointBEMlstm:predict(lsent, rsent)
    self.llstm:evaluate()
    self.rlstm:evaluate()
    self.dropoutl:evaluate()
    self.dropoutr:evaluate()
    local linputs = self.emb_vecs:forward(lsent)
    local rinputs = self.emb_vecs:forward(rsent)

    linputs = self.dropoutl:forward(linputs)
    rinputs = self.dropoutr:forward(rinputs)


    local rep_inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}
    local lHinputs = self.llstm.hOutput
    local rHinputs = self.rlstm.hOutput


    self.att_module:forward({rHinputs, lHinputs})
    self.att_module_b:forward({rHinputs, lHinputs}, true)

    local point_out = self.point_module:predict_prob({self.att_module.hOutput, self.att_module_b.hOutput}, 2)


    self.point_module:forget()
    self.att_module:forget()
    self.att_module_b:forget()

    self.llstm:forget()
    self.rlstm:forget()

    return point_out
end

function pointBEMlstm:predict_dataset(dataset)
    local logScoreFile = io.open('model_'..self.idx..'_scores.txt', "w")
    dataset.size = #dataset--/10
    local pred_probs = {}
    for i = 1, dataset.size do
        --xlua.progress(i, dataset.size)
        local lsent, rsent, labels, labels_dict, labels_ans = unpack(dataset[i])
        local pred = self:predict(lsent, rsent)

        -- Write probabilities
        logScoreFile:write(table.concat(torch.totable(pred[1]), ' '))
        logScoreFile:write('\t')
        logScoreFile:write(table.concat(torch.totable(pred[2]), ' '))
        logScoreFile:write('\n')

        pred_probs[i] = pred
    end
    logScoreFile.close()
    torch.save('prob_tensor'..self.idx, pred_probs)
end


function pointBEMlstm:load(path)
    local state = torch.load(path)
    self:__init(state.config)
    self.params:copy(state.params)
end
