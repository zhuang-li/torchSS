--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 3/08/2016
-- Time: 11:49 PM
-- To change this template use File | Settings | File Templates.
--



function sentenceSim.read_embedding(vocab_path, emb_path)
    local vocab = sentenceSim.Vocab(vocab_path)
    local embedding = torch.load(emb_path):double()
    return vocab, embedding
end


function sentenceSim.load_pretrain_data(data_path,vocab,batch_size,seq_length)
    local data_set = {}
    data_set.max_batchs = 0
    data_set.size = 0
    data_set.dump_data_size = 0
    data_set.sum_length = 0
    data_set.labels = {}
    data_set.lsents = {}
    data_set.rsents = {}
    local file = io.open(data_path)
    while true do
        local output = sentenceSim.read_pretrain_batch(file,vocab,batch_size,seq_length)
        if output == nil then break end
        data_set.labels[data_set.max_batchs + 1] = output[1]
        data_set.lsents[data_set.max_batchs + 1] = output[2]
        data_set.rsents[data_set.max_batchs + 1] = output[3]
        data_set.max_batchs = data_set.max_batchs + 1
        data_set.size = data_set.size + batch_size
        data_set.dump_data_size = data_set.dump_data_size + output[4]
        --print (output[5])
        data_set.sum_length = data_set.sum_length + output[5] + output[6]
        --print (data_set.sum_length)
    end

    return data_set
end

function sentenceSim.read_pretrain_batch(file,vocab,batch_size,seq_length)
    local label_matrix = torch.LongTensor(batch_size,seq_length)
    local ldata_matrix = torch.LongTensor(batch_size,seq_length)
    local rdata_matrix = torch.LongTensor(batch_size,seq_length)
    local idx = 0
    local abadon_data = 0
    local sent1_length = 0
    local sent2_length = 0
    while idx < batch_size do
        local line = file:read()
        if line == nil then break end
        local items = stringx.split(line, '\t')
        local sent1_tensor,sent1_len = sentenceSim.read_tokens_tensor_and_padding(items[3],vocab,seq_length,true)
        local sent2_tensor,sent2_len = sentenceSim.read_tokens_tensor_and_padding(items[4],vocab,seq_length,false)

        local label_tensor = sentenceSim.process_pretrain_label(sent2_tensor,vocab)
        if sent1_tensor ~= nil and sent2_tensor ~= nil and label_tensor ~=nil then
            ldata_matrix[idx + 1] =  sent1_tensor
            rdata_matrix[idx + 1] =  sent2_tensor
            label_matrix[idx + 1] = label_tensor
            sent1_length = sent1_length + sent1_len
            sent2_length = sent2_length + sent2_len
            idx = idx + 1
        else
            abadon_data = abadon_data + 1
        end
    end
    if idx ~= batch_size then
        return nil
    else
        return {label_matrix:t(),ldata_matrix:t(),rdata_matrix:t(),abadon_data,sent1_length,sent2_length}
    end
end

function sentenceSim.process_pretrain_label(sent2_tensor,vocab)
    if sent2_tensor == nil then
        return nil
    end

    local length = sent2_tensor:size(1)
    --print (length)
    local label_tensor = torch.LongTensor(length)

    for i = 1, length do
        if i == length or sent2_tensor[i+1] == vocab.pad_index then
            label_tensor[i] = vocab:index('</s>')
        else
            label_tensor[i] = sent2_tensor[i+1]
        end
    end
    return label_tensor
end

function sentenceSim.load_data(data_path,vocab,batch_size,seq_length)
    local data_set = {}
    data_set.max_batchs = 0
    data_set.size = 0
    data_set.dump_data_size = 0
    data_set.sum_length = 0
    data_set.labels = {}
    data_set.lsents = {}
    data_set.rsents = {}
    local file = io.open(data_path)
    while true do
        local output = sentenceSim.read_batch(file,vocab,batch_size,seq_length)
        if output == nil then break end
        data_set.labels[data_set.max_batchs + 1] = output[1]
        data_set.lsents[data_set.max_batchs + 1] = output[2]
        data_set.rsents[data_set.max_batchs + 1] = output[3]
        data_set.max_batchs = data_set.max_batchs + 1
        data_set.size = data_set.size + batch_size
        data_set.dump_data_size = data_set.dump_data_size + output[4]
        --print (output[5])
        data_set.sum_length = data_set.sum_length + output[5] + output[6]
        --print (data_set.sum_length)
    end

    return data_set
end

function sentenceSim.read_batch(file,vocab,batch_size,seq_length)
    local label_tensor = torch.Tensor(batch_size)
    local ldata_matrix = torch.Tensor(batch_size,seq_length)
    local rdata_matrix = torch.Tensor(batch_size,seq_length)
    local idx = 0
    local abadon_data = 0
    local sent1_length = 0
    local sent2_length = 0
    while idx < batch_size do
        local line = file:read()
        if line == nil then break end
        local items = stringx.split(line, '\t')
        local sent1_tensor,sent1_len = sentenceSim.read_tokens_tensor_and_padding(items[3],vocab,seq_length,true)
        local sent2_tensor,sent2_len = sentenceSim.read_tokens_tensor_and_padding(items[4],vocab,seq_length,true)
        local label = sentenceSim.process_label(items[5])
        if sent1_tensor ~= nil and sent2_tensor ~= nil and label ~=nil then
            ldata_matrix[idx + 1] =  sent1_tensor
            rdata_matrix[idx + 1] =  sent2_tensor
            label_tensor[idx + 1] = label
            sent1_length = sent1_length + sent1_len
            sent2_length = sent2_length + sent2_len
            idx = idx + 1
        else
            abadon_data = abadon_data + 1
        end
    end
    if idx ~= batch_size then
        return nil
    else
        return {label_tensor,ldata_matrix:t(),rdata_matrix:t(),abadon_data,sent1_length,sent2_length}
    end
end

function sentenceSim.process_label(label_tuple)
    --print (label_tuple)
    local c = label_tuple:sub(2,2)
    --print (c)
    local v = tonumber(c)
    if v >= 3 then
        return 1
    elseif v == 2 then
        return nil
    else
        return 0
    end
end

function sentenceSim.read_tokens_tensor_and_padding(sent,vocab,seq_length,reverse)
    local tokens = stringx.split(sent, ' ')
    local sent_tensor = torch.Tensor(seq_length):fill(vocab.pad_index)
    local token_length = #tokens
    if token_length > seq_length then
        return nil
    else
        for i = 1, token_length do
            if reverse then
                if vocab:index(tokens[i]) == nil then
                    sent_tensor[seq_length- token_length + i] = vocab.unk_index
                else
                    sent_tensor[seq_length- token_length + i] = vocab:index(tokens[i])
                end
            else
                if vocab:index(tokens[i]) == nil then
                    sent_tensor[i] = vocab.unk_index
                else
                    sent_tensor[i] = vocab:index(tokens[i])
                end
            end
        end
    end
    --print ('Padding index is '..vocab.pad_index)
    --print (sent_tensor)
    return sent_tensor,token_length
end