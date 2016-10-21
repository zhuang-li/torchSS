
--[[
-- User: zhuangli
-- Date: 4/08/2016
-- Time: 10:45 PM
-- Vocabulary file, object used to store the word vocabulary
--]]

local vocab = torch.class('sentenceSim.vocab')

function vocab:__init(path)
    self.size = 0
    self._index = {}
    self._tokens = {}
    self._frequent = {}
    local file = io.open(path)
    while true do
        local line = file:read()
        if line == nil then break end
        self.size = self.size + 1
        self._tokens[self.size] = line
        self._index[line] = self.size
        self._frequent[line] = 0
    end
    file:close()

end

-- store frequency of words

function vocab:frequent(w)
    local freq = self._frequent[w]
    if freq == nil then
        return 0
    end
    return freq
end

-- increment frequency of words

function vocab:addFrequent(w)
    local freq = self._frequent[w]
    if freq ~= nil then
        self._frequent[w] = freq + 1
    end
end

-- get tokens

function vocab:token(i)
    if i < 1 or i > self.size then
        error('Index ' .. i .. ' out of bounds')
    end
    return self._tokens[i]
end

-- check whether word is contained in vocabulary

function vocab:contains(w)
    if not self._index[w] then return false end
    return true
end

-- add new token to vocabulart

function vocab:add(w)
    if self._index[w] ~= nil then
        return self._index[w]
    end
    self.size = self.size + 1
    self._tokens[self.size] = w
    self._index[w] = self.size
    self._frequent[w] = 0
    return self.size
end

-- get the index of token

function vocab:index(w)
    local index = self._index[w]
    if index == nil then
        return self.unk_index
    end
    return index
end

-- add unkknown token

function vocab:add_unk_token()
    if self.unk_token ~= nil then return end
    self.unk_index = self:add('<UNK>')
end

-- add padding token

function vocab:add_pad_token()
    if self.pad_token ~= nil then return end
    self.pad_index = self:add('<PAD>')
end

-- add start token

function vocab:add_start_token()
    if self.start_token ~= nil then return end
    self.start_index = self:add('<s>')
end

-- add end token

function vocab:add_end_token()
    if self.end_token ~= nil then return end
    self.end_index = self:add('</s>')
end


