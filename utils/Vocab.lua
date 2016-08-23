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
    local file = io.open(path)
    while true do
        local line = file:read()
        if line == nil then break end
        self.size = self.size + 1

        local token_line = stringx.split(line, ' ')
        if #token_line ~= 1 then
            self._tokens[self.size] = token_line[1]
            self._index[token_line[1]] = self.size
            self._frequent[token_line[1]] = 0
        else
            self._tokens[self.size] = line
            self._index[line] = self.size
            self._frequent[line] = 0
        end
    end
    file:close()

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
        return self.unk_index
    end
    return index
end

function Vocab:frequent(w)
    local freq = self._frequent[w]
    --print (freq)
    if freq == nil then
        return 0
    end
    return freq
end

function Vocab:addFrequent(w)
    local freq = self._frequent[w]
    if freq ~= nil then
        self._frequent[w] = freq + 1
    end
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


