
local bpointBEMlstm = torch.class('transition.bpointBEMlstm')

function bpointBEMlstm:__init(config)
    self.mem_dim       = config.mem_dim       or 150
    self.att_dim       = config.att_dim       or self.mem_dim
    self.fih_dim       = config.fih_dim       or self.mem_dim
    self.learning_rate = config.learning_rate or 0.001
    self.batch_size    = config.batch_size    or 25
    self.num_layers    = config.num_layers    or 1
    self.reg           = config.reg           or 1e-4
    self.lstmModel     = config.lstmModel     or 'bilstm' -- {lstm, bilstm}
    self.sim_nhidden   = config.sim_nhidden   or 50
    self.emb_dim       = config.wvecDim       or 300
    self.task          = config.task          or 'paraphrase'
    self.numWords      = config.numWords
    self.maxsenLen     = config.maxsenLen     or 50
    self.dropoutP      = config.dropoutP      or 0
    self.grad          = config.grad          or 'adamax'
    self.visualize     = false
    self.emb_lr        = config.emb_lr        or 0.001
    self.emb_partial   = config.emb_partial   or true
    self.best_res      = 0
    -- word embedding
    self.emb_vecs = Embedding(self.numWords, self.emb_dim)

    self.dropoutl = nn.Dropout(self.dropoutP)
    self.dropoutr = nn.Dropout(self.dropoutP)


    self.optim_state = { learningRate = self.learning_rate }


    self.criterion = nn.ClassNLLCriterion()
    self.bcriterion = nn.ClassNLLCriterion()

    local lstm_config = {in_dim = self.emb_dim,mem_dim = self.mem_dim}
    self.llstm = transition.LSTM(lstm_config)
    self.llstm_b = transition.LSTM(lstm_config)
    self.rlstm = transition.LSTM(lstm_config)
    self.rlstm_b = transition.LSTM(lstm_config)

    local wwatten_config = {in_dim  = self.mem_dim * 2, att_dim = self.att_dim, mem_dim = self.mem_dim}
    self.att_module = transition.LSTMwwatten(wwatten_config)
    self.att_module_b = transition.LSTMwwatten(wwatten_config)

    self.point_module = transition.pointNet({in_dim = 2*self.mem_dim, mem_dim = self.mem_dim})
    self.bpoint_module = transition.pointNet({in_dim = 2*self.mem_dim, mem_dim = self.mem_dim})

    local modules = nn.Parallel()
      :add(self.llstm)
      :add(self.att_module)
      :add(self.point_module)
      :add(self.bpoint_module)
    self.params, self.grad_params = modules:getParameters()

    share_params(self.rlstm, self.llstm)
    share_params(self.att_module_b, self.att_module)
    share_params(self.llstm_b, self.llstm)
    share_params(self.rlstm_b, self.llstm)

end


function bpointBEMlstm:predict(lsent, rsent)

    local linputs = self.emb_vecs:forward(lsent)
    local rinputs = self.emb_vecs:forward(rsent)

    linputs = self.dropoutl:forward(linputs)
    rinputs = self.dropoutr:forward(rinputs)

    self.llstm:forward(linputs)
    self.llstm_b:forward(linputs, true)
    self.rlstm:forward(rinputs)
    self.rlstm_b:forward(rinputs, true)

    local lHinputs = torch.Tensor(lsent:size(1), self.mem_dim * 2)
    local rHinputs = torch.Tensor(rsent:size(1), self.mem_dim * 2)

    lHinputs[{{},{1, self.mem_dim}}] = self.llstm.hOutput
    lHinputs[{{},{self.mem_dim+1, -1}}] = self.llstm_b.hOutput
    rHinputs[{{},{1, self.mem_dim}}] = self.rlstm.hOutput
    rHinputs[{{},{self.mem_dim+1, -1}}] = self.rlstm_b.hOutput


    self.att_module:forward({rHinputs, lHinputs})
    self.att_module_b:forward({rHinputs, lHinputs}, true)

    local point_out = self.point_module:predict_prob({self.att_module.hOutput, self.att_module_b.hOutput}, 2)
    local bpoint_out = self.bpoint_module:predict_prob({self.att_module.hOutput, self.att_module_b.hOutput}, 2)
    point_out[1]:add(bpoint_out[2])
    point_out[2]:add(bpoint_out[1])
    local pre_labels
    local max_score = -999999
    for i = 1, point_out:size(2) do
        for j = 0, 15 do
            if i + j > point_out:size(2) then break end
            local score = point_out[1][i] + point_out[2][i+j]

            if score > max_score then
                max_score = score
                pre_labels = {i, i+j}
            end
        end
    end

    self.point_module:forget()
    self.bpoint_module:forget()
    self.att_module:forget()
    self.att_module_b:forget()

    self.llstm:forget()
    self.rlstm:forget()
    assert(self.lstmModel == 'bilstm')
    self.llstm_b:forget()
    self.rlstm_b:forget()

    return pre_labels, torch.totable(point_out)
end

function bpointBEMlstm:predict_dataset(dataset)
    self.llstm:evaluate()
    self.rlstm:evaluate()
    self.llstm_b:evaluate()
    self.rlstm_b:evaluate()

    self.dropoutl:evaluate()
    self.dropoutr:evaluate()
    local fileL = io.open('test_output.txt', "w")
    local logScoreFile = io.open('model_scores.txt', "w")
    dataset.size = #dataset--/10
    for i = 1, dataset.size do
        xlua.progress(i, dataset.size)
        local lsent, rsent, labels, labels_dict, labels_ans = unpack(dataset[i])
        local pred, point_out = self:predict(lsent, rsent)
        -- Write probabilities
        logScoreFile:write(table.concat(point_out[1], ' '))
        logScoreFile:write('\t')
        logScoreFile:write(table.concat(point_out[2], ' '))
        logScoreFile:write('\n')

        local pred_str = {}
        local ps, pe
        if pred[1] <= pred[2] then
            ps, pe = pred[1], pred[2]
        else
            ps, pe = pred[2], pred[1]
        end
        while ps <= pe do

            fileL:write(ivocab[lsent[ps] ])
            if ps ~= pe then fileL:write(' ') end
            ps = ps + 1
        end
        fileL:write('\n')
    end

    fileL:close()
    logScoreFile.close()
    return res
end


function bpointBEMlstm:load(path)
    local state = torch.load(path)
    if self.visualize then
        state.config.visualize = true
    end
    self:__init(state.config)
    self.params:copy(state.params)
end
