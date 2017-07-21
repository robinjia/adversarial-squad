
local Embedding, parent = torch.class('Embedding', 'nn.Module')

function Embedding:__init(inputSize, outputSize, unUpdateVocab)
    parent.__init(self)
    self.outputSize = outputSize
    self.weight = torch.randn(inputSize, outputSize)
    self.gradWeight = {}
    if type(unUpdateVocab) == 'table' then
        self.unUpdateVocab = unUpdateVocab
    else
        self.unUpdateVocab = {}
    end
end

function Embedding:updateOutput(input)
    self.output = torch.Tensor(input:size(1), self.outputSize)
    for i = 1, input:size(1) do
        self.output[i]:copy(self.weight[input[i]])
    end
    return self.output
end

function Embedding:updateGradInput(input, gradOutput)
    if self.gradInput then
        self.gradInput:resize(input:size())
        return self.gradInput
    end
end

function Embedding:accGradParameters(input, gradOutput, scale)
    for i = 1, input:size(1) do
        local word = input[i]
        if not self.unUpdateVocab[word] then
            if self.gradWeight[word] == nil then
                self.gradWeight[word] = torch.zeros(self.outputSize)
            end
            self.gradWeight[word]:add(gradOutput[i])
        end
    end
end

function Embedding:updateParameters(learningRate)
    local params, gradParams = self.weight, self.gradWeight
    if params then
        for idx, gvec in pairs(gradParams) do
            params[idx]:add(-learningRate, gvec)
        end
    end
    gradParams = {}
end

Embedding.sharedAccUpdateGradParameters = Embedding.accUpdateGradParameters
