--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 3/08/2016
-- Time: 11:49 PM
-- To change this template use File | Settings | File Templates.
--

function sentenceSim.read_embedding(vocab_path, emb_path)
    local vocab = sentenceSim.read_vocab(vocab_path)
    local embedding = torch.load(emb_path):double()
    return vocab, embedding
end

function sentenceSim.read_vocab(vocab_path)
    local vocab = {}
    vocab.size = 0
    vocab._index = {}
    vocab._tokens = {}

    local file = io.open(vocab_path)
    while true do
        local line = file:read()
        if line == nil then break end
        vocab.size = vocab.size + 1

        local token_line = stringx.split(line, ' ')
        if #token_line ~= 1 then
            vocab._tokens[vocab.size] = token_line[1]
            vocab._index[token_line[1]] = vocab.size
        else
            vocab._tokens[vocab.size] = line
            vocab._index[line] = vocab.size
        end
    end
    file:close()
    return vocab
end


function sentenceSim.load_data(data_path,vocab,batch_size,seq_length)
    local data_set = {}
    data_set.max_batchs = 0
    data_set.size = 0
    local file = io.open(data_path)
    while true do

        local output = sentenceSim.read_batch(file,vocab,batch_size,seq_length)
        if output == nil then break end
        data_set.labels[data_set.max_batchs + 1] = output[1]
        data_set.lsents[data_set.max_batchs + 1] = output[2]
        data_set.rsents[data_set.max_batchs + 1] = output[3]
        data_set.max_batchs = data_set.max_batchs + 1
        data_set.size = (data_set.size + 1)*batch_size
    end
end

function sentenceSim.read_batch(file,vocab,batch_size,seq_length)
    local label_tensor = torch.Tensor(batch_size)
    local ldata_matrix = torch.Tensor(batch_size,seq_length)
    local rdata_matrix = torch.Tensor(batch_size,seq_length)
    local idx = 0
    while idx < batch_size do
        local line = file:read()
        if line == nil then break end
        local items = stringx.split(line, '\t')
        local sent1_tensor = sentenceSim.read_tokens_tensor_and_padding(items[3],vocab,seq_length)
        local sent2_tensor = sentenceSim.read_tokens_tensor_and_padding(items[4],vocab,seq_length)
        if sent1_tensor ~= nil and sent2_tensor ~= nil then
            ldata_matrix[idx + 1] =  sent1_tensor
            rdata_matrix[idx + 1] =  sent2_tensor
            idx = idx + 1
        end
    end
    if idx ~= batch_size then
        return nil
    else
        return {label_tensor,ldata_matrix:t(),rdata_matrix:t()}
    end
end

function sentenceSim.read_tokens_tensor_and_padding(sent,vocab,seq_length)
    local tokens = stringx.split(sent, ' ')
    local sent_tensor = torch.Tensor(seq_length)
    if #tokens > seq_length then
        return nil
    else
        for i = 1, #tokens do
            if vocab._index[tokens[i]] == nil then
                sent_tensor[i] = vocab._index['#UNKNOWN#']
            else
                sent_tensor[i] = vocab._index[tokens[i]]
            end
        end
    end
    sent_tensor = nn.Padding(1,#tokens - seq_length,1,1)(sent_tensor)
    return sent_tensor
end