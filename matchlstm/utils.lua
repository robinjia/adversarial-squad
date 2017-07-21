
function share_params(cell, src)
    if torch.type(cell) == 'nn.gModule' then
        for i = 1, #cell.forwardnodes do
            local node = cell.forwardnodes[i]
            if node.data.module then
                node.data.module:share(src.forwardnodes[i].data.module,
                                    'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif torch.isTypeOf(cell, 'nn.Module') then
        cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('parameters cannot be shared for this input')
    end
end

function load_data(org_data)
    local vocab = torch.load("vocab_t.t7")
    local ivocab = torch.load("ivocab_t.t7")
    local emb = torch.load('initEmb_t.t7')
    local unSeenVocab = {}
    local unSeenVocabNum = 0
    print (#ivocab)
    local filename = 'test.txt'
    local data = {}
    --for _, line in pairs(stringx.split(org_data, '\n')) do
    for line in io.lines('data_token.txt') do
        local divs = stringx.split(line, '\t')

        assert(#divs==2)
        local instance = {}
        for j = 1, #divs do
            local words = stringx.split(stringx.strip(divs[j]), ' ')
            instance[j] = torch.IntTensor(#words)
            for i = 1, #words do
                if vocab[ words[i] ] ~= nil then
                    instance[j][i] = vocab[ words[i] ]
                else
                    vocab[words[i]] = #ivocab + 1
                    unSeenVocab[words[i]] = #ivocab + 1
                    instance[j][i] = #ivocab + 1
                    ivocab[#ivocab + 1] = words[i]
                    unSeenVocabNum = unSeenVocabNum + 1
                end
            end
        end
        data[#data+1] = instance
    end
    local emb_new = torch.zeros(#ivocab, 300)
    emb_new[{{1,emb:size(1)}}]:copy(emb)

    --reload glove

    local count = 0

    if unSeenVocabNum ~= 0 then
        local pwdpath = sys.execute('echo $Glove_DATA')
        pwdpath = stringx.strip(pwdpath)
        local file = io.open(pwdpath.."/glove.840B.300d.txt", 'r')
        while true do
            local line = file:read()
            if line == nil then break end
            vals = stringx.split(line, ' ')
            if unSeenVocab[vals[1] ] ~= nil then
                for i = 2, #vals do
                    emb_new[unSeenVocab[vals[1] ] ][i-1] = tonumber(vals[i])
                end
                count = count + 1
                if count == unSeenVocabNum then
                    break
                end
            end
        end
    end

    print('unSeenVocabNum: '..unSeenVocabNum)
    print('count: '..count)
    print (#ivocab)
    return {data, vocab, ivocab, emb_new}
end
