--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 3/08/2016
-- Time: 11:49 PM
-- Pre-processing code, pre-process the data to adapt our model's needs
--


-- read word embedding and vocabulary

function sentenceSim.read_embedding(vocab_path, emb_path)
    local vocab = sentenceSim.vocab(vocab_path)
    local embedding = torch.load(emb_path):double()
    return vocab, embedding
end


-- load pre-training data

function sentenceSim.load_pretrain_data(data_path,vocab,batch_size,seq_length)
    local data_set = {}
    data_set.max_batchs = 0
    data_set.size = 0
    data_set.dump_data_size = 0
    data_set.sum_length = 0
    data_set.labels = {}
    data_set.lsents = {}
    data_set.rsents = {}
    data_set.unk_words = 0
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
        data_set.unk_words = data_set.unk_words + output[7]
        data_set.sum_length = data_set.sum_length + output[5] + output[6]
    end
    file:close()
    return data_set
end

-- load a batch of pre-training data

function sentenceSim.read_pretrain_batch(file,vocab,batch_size,seq_length)
    local label_matrix = torch.LongTensor(batch_size,seq_length)
    local ldata_matrix = torch.LongTensor(batch_size,seq_length)
    local rdata_matrix = torch.LongTensor(batch_size,seq_length)
    local idx = 0
    local abadon_data = 0
    local sent1_length = 0
    local sent2_length = 0
    local unk_count = 0
    while idx < batch_size do
        local line = file:read()
        if line == nil then break end
        local items = stringx.split(line, '\t')
        local sent1_tensor,sent1_len,unk_count1 = sentenceSim.read_tokens_tensor_and_padding(items[3],vocab,seq_length,true,false)
        local sent2_tensor,sent2_len,unk_count2 = sentenceSim.read_tokens_tensor_and_padding(items[3],vocab,seq_length,false,true)

        local label_tensor = sentenceSim.process_pretrain_label(sent2_tensor,vocab)
        if sent1_tensor ~= nil and sent2_tensor ~= nil and label_tensor ~=nil then
            ldata_matrix[idx + 1] =  sent1_tensor
            rdata_matrix[idx + 1] =  sent2_tensor
            label_matrix[idx + 1] = label_tensor
            sent1_length = sent1_length + sent1_len
            sent2_length = sent2_length + sent2_len
            unk_count = unk_count + unk_count1 + unk_count2
            idx = idx + 1
        else
            abadon_data = abadon_data + 1
        end
    end
    if idx ~= batch_size then
        return nil
    else
        return {label_matrix:t(),ldata_matrix:t(),rdata_matrix:t(),abadon_data,sent1_length,sent2_length,unk_count}
    end
end

-- process labels for pre-training data

function sentenceSim.process_pretrain_label(sent2_tensor,vocab)
    if sent2_tensor == nil then
        return nil
    end
    local length = sent2_tensor:size(1)

    local label_tensor = torch.zeros(length)
    for i = 1, length do
        if i == length or sent2_tensor[i+1] == vocab.pad_index then
            label_tensor[i] = vocab:index('</s>')
            break
        else
            label_tensor[i] = sent2_tensor[i+1]
        end
    end
    return label_tensor
end

-- load finetuning data

function sentenceSim.load_data(data_path,vocab,batch_size,seq_length,label_path)
    local data_set = {}
    data_set.max_batchs = 0
    data_set.size = 0
    data_set.dump_data_size = 0
    data_set.sum_length = 0
    data_set.unk_words = 0
    data_set.labels = {}
    data_set.lsents = {}
    data_set.rsents = {}
    local file = io.open(data_path)
    local label_file
    if label_path ~= nil then
        label_file = io.open(label_path)
    end
    while true do
        local output = sentenceSim.read_batch(file,vocab,batch_size,seq_length,label_file)
        if output == nil then break end
        data_set.labels[data_set.max_batchs + 1] = output[1]
        data_set.lsents[data_set.max_batchs + 1] = output[2]
        data_set.rsents[data_set.max_batchs + 1] = output[3]
        data_set.max_batchs = data_set.max_batchs + 1
        data_set.size = data_set.size + output[1]:size(1)
        data_set.dump_data_size = data_set.dump_data_size + output[4]
        data_set.sum_length = data_set.sum_length + output[5] + output[6]
        data_set.unk_words = data_set.unk_words + output[7]
    end
    file:close()
    if label_path ~= nil then
        label_file:close()
    end
    return data_set
end

-- read a batch of finetuning data

function sentenceSim.read_batch(file,vocab,batch_size,seq_length,label_file)
    local label_tensor = torch.Tensor(batch_size)
    local ldata_matrix = torch.Tensor(batch_size,seq_length)
    local rdata_matrix = torch.Tensor(batch_size,seq_length)
    local idx = 0
    local abadon_data = 0
    local sent1_length = 0
    local sent2_length = 0
    local unk_count = 0
    while idx < batch_size do
        local line = file:read()
        if line == nil then break end
        local items = stringx.split(line, '\t')
        local sent1_tensor,sent1_len,unk_count1 = sentenceSim.read_tokens_tensor_and_padding(items[3],vocab,seq_length,true,false)
        local sent2_tensor,sent2_len,unk_count2 = sentenceSim.read_tokens_tensor_and_padding(items[4],vocab,seq_length,true,false)
        local label
        if label_file == nil then
            label = sentenceSim.process_label(items[5])
        else
            label = sentenceSim.process_test_label(label_file:read())
        end
        if sent1_tensor ~= nil and sent2_tensor ~= nil and label ~=nil then
            ldata_matrix[idx + 1] =  sent1_tensor
            rdata_matrix[idx + 1] =  sent2_tensor
            label_tensor[idx + 1] = label
            sent1_length = sent1_length + sent1_len
            sent2_length = sent2_length + sent2_len
            unk_count = unk_count + unk_count1 + unk_count2
            idx = idx + 1
        else
            abadon_data = abadon_data + 1
        end
    end
    if idx ~= batch_size then
        if idx == 0 then
            return nil
        else
            return {label_tensor:narrow(1,1,idx),ldata_matrix:narrow(1,1,idx):t(),rdata_matrix:narrow(1,1,idx):t(),abadon_data,sent1_length,sent2_length,unk_count }
        end
    else
        return {label_tensor,ldata_matrix:t(),rdata_matrix:t(),abadon_data,sent1_length,sent2_length,unk_count}
    end
end

-- process labels of test data in finetuning

function sentenceSim.process_test_label(line)
    local items = stringx.split(line, '\t')
    if items[1] == 'false' then
        return 0
    elseif items[1] == '----' then
        return nil
    else
        return 1
    end
end

-- process labels of development data in finetuning

function sentenceSim.process_label(label_tuple)
    local c = label_tuple:sub(2,2)
    local v = tonumber(c)
    if v == 2 then
        return nil
    elseif v >= 3 then
        return 1
    else
        return 0
    end
end

-- read frequency of words

function sentenceSim.read_unigram(vocab)
    local tensor = torch.zeros(vocab.size)
    for i = 1,vocab.size do
        tensor[i] = vocab:frequent(vocab:token(i))
    end
    return tensor
end

-- read tensors for sentences

function sentenceSim.read_tokens_tensor_and_padding(sent,vocab,seq_length,reverse,dec)
    local tokens_ori = stringx.split(sent, ' ')
    local tokens = {}
    if dec then
        tokens[1] = '<s>'
        for i = 1, #tokens_ori do
            tokens[i+1] = tokens_ori[i]
        end
    else
        tokens = tokens_ori
    end
    local sent_tensor = torch.Tensor(seq_length):fill(vocab.pad_index)
    local token_length = #tokens
    local count = 0
    if token_length > seq_length then
        return nil
    else
        for i = 1, token_length do
            local idx = 0
            if reverse then
                idx = seq_length - token_length + i
            else
                idx = i
            end
            local token_idx = 0
            if not vocab:contains(tokens[i]) then
                count = count + 1
                token_idx = vocab.unk_index
            else
                token_idx = vocab:index(tokens[i])
                vocab:addFrequent(tokens[i])

            end
            sent_tensor[idx] = token_idx
        end
    end

    return sent_tensor,token_length,count
end