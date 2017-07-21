require 'torch'
require 'nngraph'
require 'optim'
require 'debug'
require 'sys'
require 'nn'
include 'utils.lua'
cmd = torch.CmdLine()
cmd:option('-input','dev-v1.1.json','input dataset')
local prob_tensors = {}
for i = 1, 5 do
	prob_tensors[i] = torch.load('prob_tensor'..i)
end

local fileL = io.open('test_output.txt', "w")
local j = 1
for line in io.lines('data_token.txt') do
	local divs = stringx.split(line, '\t')
	local words = stringx.split(stringx.strip(divs[1]), ' ')

	for i = 2, 5 do
		prob_tensors[1][j]:add(prob_tensors[i][j])
	end

	local point_out = prob_tensors[1][j]
	local pred
	local max_score = -999999
    for i = 1, point_out:size(2) do
        for j = 0, 15 do
            if i + j > point_out:size(2) then break end
            local score = point_out[1][i] + point_out[2][i+j]

            if score > max_score then
                max_score = score
                pred = {i, i+j}
            end
        end
    end
	local ps, pe
	if pred[1] <= pred[2] then
		ps, pe = pred[1], pred[2]
	else
		ps, pe = pred[2], pred[1]
	end
	while ps <= pe do

		fileL:write(words[ps])
		if ps ~= pe then fileL:write(' ') end
		ps = ps + 1
	end
	fileL:write('\n')
	j = j + 1
end
