--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 4/08/2016
-- Time: 10:45 PM
-- To change this template use File | Settings | File Templates.
--
--[[

A vocabulary object. Initialized from a file with one vocabulary token per line.
Maps between vocabulary tokens and indices. If an UNK token is defined in the
vocabulary, returns the index to this token if queried for an out-of-vocabulary
token.

--]]

local Vocab = torch.class('sentenceSim.Vocab')

function Vocab:__init(path)
    self.size = 0
    self._index = {}
    self._tokens = {}
    self._frequent = {}
    --local file = io.open(path)
    local count = 1
    while true do
        local line = path[count]
        --print (line)
        if line == nil then break end
        self.size = self.size + 1

        --local token_line = stringx.split(line, ' ')
        --print (token_line)
        self._tokens[self.size] = line
        self._index[line] = self.size
        self._frequent[line] = 0

        count = count + 1
    end
    --file:close()

    local unks = {'<UNK>', '<unk>', 'UUUNKKK'}
    for _, tok in pairs(unks) do
        self.unk_index = self.unk_index or self._index[tok]
        if self.unk_index ~= nil then
            self.unk_token = tok
            break
        end
    end

    local starts = {'<s>', '<S>'}
    for _, tok in pairs(starts) do
        self.start_index = self.start_index or self._index[tok]
        if self.start_index ~= nil then
            self.start_token = tok
            break
        end
    end

    local ends = {'</s>', '</S>'}
    for _, tok in pairs(ends) do
        self.end_index = self.end_index or self._index[tok]
        if self.end_index ~= nil then
            self.end_token = tok
            break
        end
    end

    local pads = {'<PAD>','#PAD#'}
    for _, tok in pairs(pads) do
        self.pad_index = self.pad_index or self._index[tok]
        if self.pad_index ~= nil then
            self.pad_token = tok
            break
        end
    end

end

function Vocab:contains(w)
    if not self._index[w] then return false end
    return true
end

function Vocab:add(w)
    if self._index[w] ~= nil then
        return self._index[w]
    end
    self.size = self.size + 1
    self._tokens[self.size] = w
    self._index[w] = self.size
    self._frequent[w] = 0
    return self.size
end

function Vocab:index(w)
    local index = self._index[w]
    if index == nil then
        --    if self.unk_index == nil then
        --      error('Token not in vocabulary and no UNK token defined: ' .. w)
        --    end
        return self.unk_index
    end
    return index
end

function Vocab:frequent(w)
    local freq = self._frequent[w]
    --print (freq)
    if freq == nil then
        --    if self.unk_index == nil then
        --      error('Token not in vocabulary and no UNK token defined: ' .. w)
        --    end
        return 0
    end
    return freq
end

function Vocab:addFrequent(w)
    local freq = self._frequent[w]
    --print (freq)
    if freq ~= nil then
        --print ('baba')
        self._frequent[w] = freq + 1
        --print (self._frequent[w])
    end
    --print (self._frequent)
end

function Vocab:token(i)
    if i < 1 or i > self.size then
        error('Index ' .. i .. ' out of bounds')
    end
    return self._tokens[i]
end



function Vocab:map(tokens)
    local len = #tokens
    local output = torch.IntTensor(len)
    for i = 1, len do
        output[i] = self:index(tokens[i])
    end
    return output
end

function Vocab:add_unk_token()
    if self.unk_token ~= nil then return end
    self.unk_index = self:add('<UNK>')
end

function Vocab:add_pad_token()
    if self.pad_token ~= nil then return end
    self.pad_index = self:add('<PAD>')
end

function Vocab:add_start_token()
    if self.start_token ~= nil then return end
    self.start_index = self:add('<s>')
end

function Vocab:add_end_token()
    if self.end_token ~= nil then return end
    self.end_index = self:add('</s>')
end


