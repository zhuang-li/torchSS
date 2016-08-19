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
        --print (count)
    end
    return count / size
end


function label_score(pred,gold,size)
    local tp = 0
    local fp = 0
    local fn = 0
    local a = 0
    for i = 1,#pred do
        for j =1, pred[i]:size(1) do
            if pred[i][j] == gold[i][j] then
                a = a + 1
                if pred[i][j] == 1 then
                    tp = tp + 1
                end
            else
                if pred[i][j] == 1 then
                    fp = fp + 1
                else
                    fn = fn + 1
                end
            end
        end
        --print (count)
    end
    print (tp)
    print (fp)
    print (fn)
    local p = tp/(tp + fp)
    local r = tp/(tp + fn)
    local f = 2*(p*r)/(p + r)
    local ac = a / size
    return p,r,f,ac
end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train sentence similarity model')
cmd:text()
cmd:text('Options')
cmd:option('-data_dir','data','data directory. Should contain the training data and embedding')
cmd:option('-hidden_size', 200, 'size of LSTM internal state')
cmd:option('-model', 'lstm', 'lstm,gru or rnn')
cmd:option('-learning_rate',0.01,'learning rate')
cmd:option('-reg',0.0001,'regularization rate')
cmd:option('-max_epochs',15,'max training epoch')
cmd:option('-seq_length',10,'number of timesteps to unroll for')
cmd:option('-batch_size',100,'number of sequences to train on in parallel')
cmd:option('-gpuidx',0,'which gpu to use. 0 = use CPU')
cmd:option('-load','f','load pre-train model or not')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

local embedding_path = opt.data_dir..'/embedding/glove.twitter.27B.200d.th'
local vocab_path = opt.data_dir..'/embedding/glove.twitter.27B.vocab'
local train_path = opt.data_dir..'/train/pit_train.txt'
local dev_path = opt.data_dir..'/dev/pit_dev.txt'
local test_path = opt.data_dir..'/test/pit_test.txt'
local test_label = opt.data_dir..'/test/pit_test_label.txt'
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
--print (vecs[71293])
--vecs = torch.zeros(vecs:size())
ori_vocab = nil
print('unk count = ' .. num_unk)

local train_data = sentenceSim.load_data(train_path,vocab,batch_size,seq_length)
local dev_data = sentenceSim.load_data(dev_path,vocab,batch_size,seq_length)
local test_data = sentenceSim.load_data(test_path,vocab,batch_size,seq_length,test_label)
local train_avg_length = train_data.sum_length/(train_data.size*2)
local dev_avg_length = dev_data.sum_length/(dev_data.size*2)
local test_avg_length = test_data.sum_length/(test_data.size*2)
printf('max epochs = %d\n', epochs)
printf('training data size = %d\n', train_data.size)
printf('development data size = %d\n', dev_data.size)
printf('test data size = %d\n', test_data.size)
printf('dumped training data size = %d\n', train_data.dump_data_size)
printf('dumped development data size = %d\n', dev_data.dump_data_size)
printf('dumped test data size = %d\n', test_data.dump_data_size)
printf('average training data length = %d\n', train_avg_length)
printf('average development data length = %d\n', dev_avg_length)
printf('average test data length = %d\n', test_avg_length)
printf('train set unknown words = %d\n', train_data.unk_words)
printf('dev set unknown words = %d\n', dev_data.unk_words)
printf('test set unknown words = %d\n', test_data.unk_words)
vocab = nil
emb_vecs = nil
collectgarbage()
--load model



-- initialize model
local model = sentenceSim.LSTMSim{
    in_dim = emb_dim,
    emb_vecs   = vecs,
    structure  = opt.model,
    mem_dim    = opt.hidden_size,
    gpuidx = opt.gpuidx,
    batch_size = opt.batch_size,
    reg = opt.reg,
    seq_length = opt.seq_length,
    learning_rate = opt.learning_rate,
    finetune = true
}
if opt.load ~= 'f' then
    print ("Loading model")
    local model_path = opt.data_dir..'/model_ser'..opt.load
    local model_object = torch.load(model_path)
    model.lstm_params:copy(model_object.lstm_params)
end

-- print information
header('model configuration')

model:print_config()

-- train
local last_dev_score = -1
local train_start = sys.clock()
header('Training model')
printf('-- epoch %d\n', 0)
local dev_predictions = model:predict_dataset(dev_data)
local dev_score = accuracy(dev_predictions, dev_data.labels,dev_data.size)
printf('-- dev score: %.4f\n', dev_score)
local test_predictions = model:predict_dataset(test_data)
local p,f,r,a = label_score(test_predictions, test_data.labels,test_data.size)
printf('-- test score: Precision: %.4f Recall: %.4f F-measure: %.4f Accuracy: %.4f\n', p,f,r,a)
for i = 1, epochs do
    local start = sys.clock()
    printf('-- epoch %d\n', i)
    local total_loss = model:fine_tune(train_data)
    print('Train loss: '..total_loss)
    printf('-- finished epoch in %.2fs\n', sys.clock() - start)


    --local dev_predictions = model:predict_dataset(dev_data)
    --local dev_score = accuracy(dev_predictions, dev_data.labels,dev_data.size)
    --printf('-- dev score: %.4f\n', dev_score)
    --if dev_score > last_dev_score then
    local dev_predictions = model:predict_dataset(dev_data)
    local p,f,r,a = label_score(dev_predictions, dev_data.labels,dev_data.size)
    printf('-- test score: Precision: %.4f Recall: %.4f F-measure: %.4f Accuracy: %.4f\n', p,f,r,a)

    local test_predictions = model:predict_dataset(test_data)
    local p,f,r,a = label_score(test_predictions, test_data.labels,test_data.size)
    printf('-- test score: Precision: %.4f Recall: %.4f F-measure: %.4f Accuracy: %.4f\n', p,f,r,a)
    --end
    last_dev_score = dev_score
end
printf('finished training in %.2fs\n', sys.clock() - train_start)