--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 27/07/2016
-- Time: 5:26 PM
-- To change this template use File | Settings | File Templates.
--


local LSTMSim = torch.class('sentenceSim.LSTMSim')

function LSTMSim:__init(config)
    self.mem_dim       = config.mem_dim       or 200
    self.learning_rate = config.learning_rate or 0.05
    self.batch_size    = config.batch_size    or 100
    self.reg           = config.reg           or 1e-4
    self.seq_length    = config.seq_length    or 20
    --print (config.finetune)
    self.finetune      = config.finetune      or false
    --print (self.finetune)
    self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
    self.sim_nhidden   = config.sim_nhidden   or 50

    -- word embedding
    self.emb_vecs = config.emb_vecs
    self.emb_dim = config.emb_vecs:size(2)
    self.vocab_size = config.emb_vecs:size(1)
    self.lemb = nn.LookupTable(self.vocab_size, self.emb_dim)

    self.lemb.weight:copy(config.emb_vecs)
    self.remb = nn.LookupTable(self.vocab_size, self.emb_dim)
    self.remb.weight:copy(config.emb_vecs)
    -- number of similarity rating classes

    -- optimizer configuration
    self.optim_state = { learningRate = self.learning_rate }

    -- KL divergence optimization objective


    -- initialize LSTM model


    self.llstm = nn.Sequential()
    self.llstm.layer = nn.SeqLSTM(self.emb_dim, self.mem_dim)
    self.llstm.layer:maskZero()
    self.llstm:add(self.llstm.layer)
    --llstm:add(nn.Dropout())
    self.llstm:add(nn.Select(1,self.seq_length))
    if not self.finetune then
        self.enc = self.llstm
    end
    self.lstm_params = self.llstm:getParameters()
    self.rlstm = nn.Sequential()
    self.rlstm.layer = nn.SeqLSTM(self.emb_dim, self.mem_dim)
    self.rlstm.layer:maskZero()
    self.rlstm:add(self.rlstm.layer)
    --print (self.finetune)
    if self.finetune then
        self.rlstm:add(nn.Select(1,self.seq_length))
        self.criterion = nn.BCECriterion()
    else
        self.rlstm:add(nn.SplitTable(1))
        local unigram = config.unigram

        local ncemodule = nn.NCEModule(self.mem_dim, self.vocab_size, 25,unigram)
        self.dec = nn.Sequential()
        :add(nn.ParallelTable()
        :add(self.rlstm):add(nn.Identity()))
        :add(nn.ZipTable()) -- {{x1,x2,...}, {t1,t2,...}} -> {{x1,t1},{x2,t2},...}

            -- encapsulate stepmodule into a Sequencer
        self.dec:add(nn.Sequencer(nn.MaskZero(ncemodule, 1)))

            -- remember previous state between batches
        self.dec:remember()



        local crit = nn.MaskZeroCriterion(nn.NCECriterion(), 0)

            -- target is also seqlen x batchsize.
        self.targetmodule = nn.SplitTable(1)

        self.criterion = nn.SequencerCriterion(crit)
    end
    -- similarity model
    if self.finetune then
        local sim_module = self:new_sim_module()
        local siamese_encoder = nn.ParallelTable()
        :add(self.llstm)
        :add(self.rlstm)

        self.model = nn.Sequential()
        :add(siamese_encoder)
        :add(sim_module)

        self.params, self.grad_params = self.model:getParameters()

        -- share must only be called after getParameters, since this changes the
        -- location of the parameters
        if self.finetune then
            share_params(self.rlstm, self.llstm)
        end
    else
        self.model = nn.ParallelTable()
        :add(self.enc)
        :add(self.dec)
        self.params, self.grad_params = self.model:getParameters()
    end

    self.emb_vecs = nil
    collectgarbage()
end

function LSTMSim:forwardConnect(llstm, rlstm)
    rlstm.layer.userPrevOutput = llstm.layer.output[self.seq_length]
    rlstm.layer.userPrevCell = llstm.layer.cell[self.seq_length]
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function LSTMSim:backwardConnect(llstm, rlstm)
    llstm.layer.gradPrevOutput = rlstm.layer.userGradPrevOutput
    llstm.layer.userNextGradCell = rlstm.layer.userGradPrevCell
end

function LSTMSim:new_sim_module()

    local linput, rinput = nn.Identity()(), nn.Identity()()
    local mult_dist = nn.CMulTable(){linput, rinput}
    local add_dist = nn.Abs()(nn.CSubTable(){linput, rinput})
    local vec_dist_feats = nn.JoinTable(2){mult_dist, add_dist}
    local vecs_to_input = nn.gModule({linput,rinput}, {vec_dist_feats})

    -- define similarity model architecture
    local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(2* self.mem_dim, self.sim_nhidden))
    --:add(nn.Dropout())
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, 1))
    :add(nn.Sigmoid())
    return sim_module

end

function LSTMSim:pre_train(dataset)
    self.enc:training()
    self.dec:training()

    local max_batchs = dataset.max_batchs
    local zeros = torch.zeros(self.batch_size,self.mem_dim)
    local indices = torch.randperm(max_batchs)
    local total_loss = 0
    for i = 1, max_batchs do

        xlua.progress((i-1)*self.batch_size, dataset.size)

        local feval = function(x)
            self.grad_params:zero()
            local idx = indices[i]
            local targets = dataset.labels[idx]
            --print (targets)


            local lsent_ids, rsent_ids = dataset.lsents[idx], dataset.rsents[idx]
            --print (rsent_ids)
            --print (targets)
            local linputs = self.lemb:forward(lsent_ids)

            local rinputs = self.remb:forward(rsent_ids)



            local loutput = self.enc:forward(linputs)
            --print (loutput)
            self:forwardConnect(self.llstm,self.rlstm)
            targets = self.targetmodule:forward(targets)
            --print (rinputs)
            local routput = self.dec:forward({rinputs,targets})
            --print (routput)
            local loss = self.criterion:forward(routput, targets)

            assert(loss == loss,'loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cl/cl?' )
            if loss0 == nil then loss0 = loss end
            --assert(loss < loss0 * 3,'loss is exploding, aborting.')
            --print (targets)
            local rgrad = self.criterion:backward(routput, targets)
            self.dec:backward({rinputs,targets},rgrad)

            self:backwardConnect(self.llstm, self.rlstm)
            self.enc:backward(linputs,zeros)
            --local zeros = torch.zeros(self.seq_length)
            --self.lemb:forward(lsent_ids,zeros)
            --self.remb:forward(lsent_ids,zeros)
            self.lemb:clearState()
            self.remb:clearState()


            --self.grad_params:div(self.batch_size)

            -- regularization
            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            total_loss = total_loss + loss
            return loss, self.grad_params
        end

        optim.adagrad(feval, self.params, self.optim_state)
        if i%10 == 0 then
            collectgarbage()
        end
    end

    xlua.progress(dataset.size, dataset.size)
    return total_loss
end


function LSTMSim:fine_tune(dataset)

    self.model:training()
    local max_batchs = dataset.max_batchs
    local indices = torch.randperm(max_batchs)
    local total_loss = 0
    for i = 1, max_batchs do

        xlua.progress((i-1)*self.batch_size, dataset.size)

        local feval = function(x)
            self.grad_params:zero()
            local idx = indices[i]
            local targets = dataset.labels[idx]
            local lsent_ids, rsent_ids = dataset.lsents[idx], dataset.rsents[idx]

            local linputs = self.lemb:forward(lsent_ids)

            local rinputs = self.remb:forward(rsent_ids)

            local output = self.model:forward({linputs,rinputs})

            local loss = self.criterion:forward(output, targets)

            assert(loss == loss,'loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cl/cl?' )
            if loss0 == nil then loss0 = loss end

            local sim_grad = self.criterion:backward(output, targets)

            self.model:backward({linputs,rinputs},sim_grad)

            self.lemb:clearState()
            self.remb:clearState()


            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            total_loss = total_loss + loss
            return loss, self.grad_params
        end

        optim.adagrad(feval, self.params, self.optim_state)
        if i%10 == 0 then
            collectgarbage()
        end
    end

    xlua.progress(dataset.size, dataset.size)
    return total_loss
end



-- Predict the similarity of a sentence pair.
function LSTMSim:predict(lsent_ids, rsent_ids)
    self.model:evaluate()
    self.grad_params:zero()
    local linputs = self.lemb:forward(lsent_ids)

    local rinputs = self.remb:forward(rsent_ids)


    local output = self.model:forward({linputs,rinputs})

    local size = output:size(1)
    local prediction = torch.Tensor(size)

    for i = 1, size do
        --print (i)
        if output[i][1] > 0.5 then prediction[i] = 1 else prediction[i] = 0 end
    end
    self.lemb:clearState()
    self.remb:clearState()
    return prediction
end

-- Produce similarity predictions for each sentence pair in the dataset.
function LSTMSim:predict_dataset(dataset)
    local predictions = {}
    local max_batchs = dataset.max_batchs
    for i = 1, max_batchs do
        xlua.progress((i-1)*self.batch_size, dataset.size)
        local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
        predictions[i] = self:predict(lsent, rsent)
    end
    xlua.progress(dataset.size, dataset.size)
    return predictions
end

function LSTMSim:print_config()
    local num_params = self.params:nElement()
    local num_sim_params = self:new_sim_module():getParameters():nElement()
    printf('%-25s = %d\n',   'num params', num_params)
    printf('%-25s = %d\n',   'num compositional params', num_params - num_sim_params)
    printf('%-25s = %d\n',   'word vector dim', self.emb_dim)
    printf('%-25s = %d\n',   'LSTM memory dim', self.mem_dim)
    printf('%-25s = %.2e\n', 'regularization strength', self.reg)
    printf('%-25s = %d\n',   'minibatch size', self.batch_size)
    printf('%-25s = %.2e\n', 'learning rate', self.learning_rate)
    printf('%-25s = %s\n',   'LSTM structure', self.structure)
    printf('%-25s = %d\n',   'sim module hidden dim', self.sim_nhidden)
    printf('%-25s = %d\n',   'sequence length', self.seq_length)
    --printf('%-25s = %d\n',   'GPU index', self.gpuidx)
end

--
-- Serialization
--

function LSTMSim:save(path)
    torch.save(path,  self.lstm_params)
end


