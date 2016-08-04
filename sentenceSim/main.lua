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
    end
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
cmd:option('-seq_length',20,'number of timesteps to unroll for')
cmd:option('-batch_size',100,'number of sequences to train on in parallel')
cmd:option('-gpuidx',0,'which gpu to use. 0 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

local embedding_path = opt.data_dir..'/embedding/vectors.th'
local vocab_path = opt.data_dir..'/embedding/vectors_vocab.txt'
local train_path = opt.data_dir..'/train/pit_train.txt'
local dev_path = opt.data_dir..'/dev/pit_dev.txt'
local batch_size = opt.batch_size
local seq_length = opt.seq_length
local epochs = opt.max_epochs
local vocab,emb_vecs = sentenceSim.read_embedding(vocab_path, embedding_path)

local train_data = sentenceSim.load_data(train_path,vocab,batch_size,seq_length)
local dev_data = sentenceSim.load_data(dev_path,vocab,batch_size,seq_length)
collectgarbage()
-- initialize model
local model = sentenceSim.LSTMSim{
    emb_vecs   = emb_vecs,
    structure  = opt.model,
    mem_dim    = opt.hidden_size,
    gpuidx = opt.gpuidx,
    batch_size = opt.batch_size,
    reg = opt.reg,
    seq_length = opt.seq_length,
    learningRate = opt.learning_rate
}


-- print information
header('model configuration')
printf('max epochs = %d\n', epochs)
model:print_config()

-- train
local train_start = sys.clock()
local best_dev_score = -1.0
local best_dev_model = model
header('Training model')
for i = 1, epochs do
    local start = sys.clock()
    printf('-- epoch %d\n', i)
    model:train(train_data)
    printf('-- finished epoch in %.2fs\n', sys.clock() - start)

    -- uncomment to compute train scores
    --[[
    local train_predictions = model:predict_dataset(train_dataset)
    local train_score = pearson(train_predictions, train_dataset.labels)
    printf('-- train score: %.4f\n', train_score)
    --]]

    local dev_predictions = model:predict_dataset(dev_data)
    local dev_score = accuracy(dev_predictions, dev_data.labels,dev_data.size)
    printf('-- dev score: %.4f\n', dev_score)
end
printf('finished training in %.2fs\n', sys.clock() - train_start)