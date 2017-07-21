

local LSTMwwatten, parent  = torch.class('transition.LSTMwwatten', 'nn.Module')

function LSTMwwatten:__init(config)
    parent.__init(self)
    self.mem_dim       = config.mem_dim
    self.att_dim       = config.att_dim       or self.mem_dim
    self.in_dim        = config.in_dim        or self.mem_dim
    self.maxsenLen     = config.maxsenLen     or 50

    self.master_ww = self:new_ww()
    self.depth = 0
    self.wws = {}

    self.initial_values = {torch.zeros(self.mem_dim), torch.zeros(self.mem_dim)}
    self.gradInput = {
        torch.zeros(self.maxsenLen, self.mem_dim),
        torch.zeros(self.mem_dim),
        torch.zeros(self.mem_dim),
        torch.zeros(self.mem_dim)
    }
    self.hOutput = torch.Tensor()

end

function LSTMwwatten:new_ww()

    local linput, rinput, ainput, ctable_p = nn.Identity()(), nn.Identity()(), nn.Identity()(), nn.Identity()()
    --padding
    local lPad = nn.Padding(1,1)(linput)
    local M_l = nn.Linear(self.in_dim, self.att_dim)(lPad)
    --local M_l = nn.Linear(self.mem_dim, self.att_dim)(linput)
    local M_r = nn.Linear(self.in_dim, self.att_dim)(rinput)
    local M_a = nn.Linear(self.mem_dim, self.att_dim)(ainput)

    local M_ra =  nn.CAddTable(){M_r, M_a}
    local M = nn.Tanh()(nn.CAddRepTable(){M_l, M_ra})

    local wM = nn.Linear(self.att_dim, 1)(M)
    local alpha = nn.SoftMax()( nn.Transpose({1,2})(wM) )

    local Yl =  nn.Select(1, 1)(nn.MM(){alpha, lPad})
    --local Yl =  nn.Select(1, 1)(nn.MM(){alpha, linput})


    local new_gate = function()
      return nn.CAddTable(){
        nn.Linear(self.mem_dim, self.mem_dim)(ainput),
        nn.Linear(self.in_dim, self.mem_dim)(rinput),
        nn.Linear(self.in_dim, self.mem_dim)(Yl)
      }
    end

    -- input, forget, and output gates
    local i = nn.Sigmoid()(new_gate())
    local f = nn.Sigmoid()(new_gate())
    local update = nn.Tanh()(new_gate())

    -- update the state of the LSTM cell
    local ctable = nn.CAddTable(){
      nn.CMulTable(){f, ctable_p},
      nn.CMulTable(){i, update}
    }
    local o = nn.Sigmoid()(new_gate())
    local a_next = nn.CMulTable(){o, nn.Tanh()(ctable)}


    local ww = nn.gModule({linput, rinput, ainput, ctable_p}, {a_next, ctable})

    if self.master_ww then
        share_params(ww, self.master_ww)
    end

    return ww

end

function LSTMwwatten:forward(inputs, reverse)
    local lHinputs, rHinputs = unpack(inputs)
    local size = rHinputs:size(1)
    self.raRpt = lHinputs:size(1)
    self.hOutput:resize(size, self.mem_dim)
    for t = 1, size do
        local idx = reverse and size-t+1 or t
        --print (size, t)
        self.depth = self.depth + 1
        local ww = self.wws[self.depth]
        if ww == nil then
            ww = self:new_ww()
            self.wws[self.depth] = ww
        end
        local prev_output
        if t > 1 then
          prev_output = self.wws[self.depth - 1].output
        else
          prev_output = self.initial_values
        end
        local output = ww:forward({lHinputs, rHinputs[idx], unpack(prev_output)})
        self.hOutput[idx] = output[1]
        self.output = output
    end
    return self.output
end




function LSTMwwatten:share(LSTMwwatten)
    assert( self.att_dim == LSTMwwatten.att_dim)
    assert( self.mem_dim == LSTMwwatten.mem_dim)
    share_params(self.master_ww, LSTMwwatten.master_ww)
end

function LSTMwwatten:zeroGradParameters()
    self.master_ww:zeroGradParameters()
end

function LSTMwwatten:parameters()
    return self.master_ww:parameters()
end

function LSTMwwatten:forget()
    self.depth = 0
    for i = 1, #self.gradInput do
        local gradInput = self.gradInput[i]
        if type(gradInput) == 'table' then
            for _, t in pairs(gradInput) do t:zero() end
        else
            self.gradInput[i]:zero()
        end
    end
end
function LSTMwwatten:forgetgrad()
    for i = 1, #self.gradInput do
        local gradInput = self.gradInput[i]
        if type(gradInput) == 'table' then
            for _, t in pairs(gradInput) do t:zero() end
        else
            self.gradInput[i]:zero()
        end
    end
end
