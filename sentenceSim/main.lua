--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 4/08/2016
-- Time: 3:39 AM
-- To change this template use File | Settings | File Templates.
--

require('..')
function accuracy(pred,gold,size)
    local count = 0
    for i = 1,#pred do
        count = count + torch.eq(pred[i], gold[i]):sum()
        --print (pred[i])
        --print (gold[i])
    end
    --print (count)
    return count / size
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train sentence similarity model')
cmd:text()
cmd:text('Options')
cmd:option('-data_dir','data','data directory. Should contain the training data and embedding')
cmd:option('-hidden_size', 200, 'size of LSTM internal state')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
cmd:option('-learning_rate',0.05,'learning rate')
cmd:option('-reg',0.0001,'regularization rate')
cmd:option('-max_epochs',15,'max training epoch')
cmd:option('-seq_length',10,'number of timesteps to unroll for')
cmd:option('-batch_size',100,'number of sequences to train on in parallel')
cmd:option('-gpuidx',0,'which gpu to use. 0 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

local embedding_path = opt.data_dir..'/embedding/glove.840B.300d.th'
local vocab_path = opt.data_dir..'/embedding/glove.840B.vocab'
local train_path = opt.data_dir..'/train/pit_train.txt'
local dev_path = opt.data_dir..'/dev/pit_dev.txt'
local batch_size = opt.batch_size
local seq_length = opt.seq_length
local epochs = opt.max_epochs

local vocab,emb_vecs = sentenceSim.read_embedding(vocab_path, embedding_path)
local emb_dim = emb_vecs:size(2)
local ori_vocab = sentenceSim.Vocab(vocab_path)

vocab:add_unk_token()
vocab:add_pad_token()
local num_unk = 0
local vecs = torch.Tensor(vocab.size, emb_dim)
for i = 1, vocab.size do
    local w = vocab:token(i)
    if ori_vocab:contains(w) then
        vecs[i] = emb_vecs[ori_vocab:index(w)]
    else
        num_unk = num_unk + 1
        if i == vocab.pad_index then
            vecs[i]:zero()
        else
            vecs[i]:uniform(-0.05, 0.05)
        end
    end
end
--vecs = torch.zeros(vecs:size())
ori_vocab = nil
print('unk count = ' .. num_unk)

local train_data = sentenceSim.load_data(train_path,vocab,batch_size,seq_length)
--[[
local count = 0
for i=1 ,train_data.labels[1]:size(1) do
    if train_data.labels[1][i] == 1 then
        count =count +1
    end
end
print (count)
--]]
local dev_data = sentenceSim.load_data(dev_path,vocab,batch_size,seq_length)
local train_avg_length = train_data.sum_length/(train_data.size*2)
local dev_avg_length = dev_data.sum_length/(dev_data.size*2)
printf('max epochs = %d\n', epochs)
printf('training data size = %d\n', train_data.size)
printf('development data size = %d\n', dev_data.size)
printf('dumped training data size = %d\n', train_data.dump_data_size)
printf('dumped development data size = %d\n', dev_data.dump_data_size)
printf('average training data length = %d\n', train_avg_length)
printf('average development data length = %d\n', dev_avg_length)
vocab = nil
emb_vecs = nil
collectgarbage()
-- initialize model
local model = sentenceSim.LSTMSim{
    emb_vecs   = vecs,
    structure  = opt.model,
    mem_dim    = opt.hidden_size,
    gpuidx = opt.gpuidx,
    batch_size = opt.batch_size,
    reg = opt.reg,
    seq_length = opt.seq_length,
    learning_rate = opt.learning_rate
}


-- print information
header('model configuration')

model:print_config()

-- train
local train_start = sys.clock()
header('Training model')
printf('-- epoch %d\n', 0)
local dev_predictions = model:predict_dataset(dev_data)
local dev_score = accuracy(dev_predictions, dev_data.labels,dev_data.size)
printf('-- dev score: %.4f\n', dev_score)
for i = 1, epochs do
    local start = sys.clock()
    printf('-- epoch %d\n', i)
    local total_loss = model:train(train_data)
    print('Train loss: '..total_loss)
    printf('-- finished epoch in %.2fs\n', sys.clock() - start)


    local dev_predictions = model:predict_dataset(dev_data)
    local dev_score = accuracy(dev_predictions, dev_data.labels,dev_data.size)
    printf('-- dev score: %.4f\n', dev_score)
end
printf('finished training in %.2fs\n', sys.clock() - train_start)