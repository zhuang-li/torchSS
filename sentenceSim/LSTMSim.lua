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


    local llstm = nn.Sequential()
    llstm.layer = {}
    llstm.layer = nn.SeqLSTM(self.emb_dim, self.mem_dim)
    llstm.layer:maskZero()
    llstm:add(llstm.layer)
    --llstm:add(nn.Dropout())
    llstm:add(nn.Select(1,self.seq_length))
    if not self.finetune then
        --print ("enc")
        self.enc = llstm
    end
    self.lstm_params = llstm:getParameters()
    local rlstm = nn.Sequential()
    rlstm.layer = {}
    rlstm.layer = llstm.layer:clone('weight', 'bias', 'gradWeight', 'gradBias')
    rlstm.layer:maskZero()
    rlstm:add(rlstm.layer)
    --print (self.finetune)
    if self.finetune then
        rlstm:add(nn.Select(1,self.seq_length))
        self.criterion = nn.BCECriterion()
    else
        rlstm:add(nn.SplitTable(1))
        local unigram = config.unigram

        local ncemodule = nn.NCEModule(self.mem_dim, self.vocab_size, 25,unigram)
        --ncemodule.batchnoise = false
            -- NCE requires {input, target} as inputs
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

    local sim_module = self:new_sim_module()
    local siamese_encoder = nn.ParallelTable()
    :add(llstm)
    :add(rlstm)

    self.model = nn.Sequential()
    :add(siamese_encoder)
    :add(sim_module)

    self.params, self.grad_params = self.model:getParameters()
    --self.params:uniform(-0.1, 0.1)
    self.emb_vecs = nil
    collectgarbage()

    -- share must only be called after getParameters, since this changes the
    -- location of the parameters
    share_params(rlstm.layer, llstm.layer)
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
            --print (lsent_ids)
            --local a = nn.LookupTable(71293,200)
            --self.emb:clearState()
            --local b = a:forward(lsent_ids)
            --local b = a:forward(lsent_ids)
            local linputs = self.lemb:forward(lsent_ids)

            local rinputs = self.remb:forward(rsent_ids)

            --print (rinputs:size())
            --print (self.emb.weight:size())
            --print (linputs:size())
            --print (rinputs:size())
            --print (lsent_ids[1][3])
            --print (rsent_ids[1][3])
            --print (linputs)
            --print (rinputs[1][3][10])


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
    --self.llstm:training()
    --self.rlstm:training()
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
            --print (targets)
            local lsent_ids, rsent_ids = dataset.lsents[idx], dataset.rsents[idx]
            --print (lsent_ids)
            --local a = nn.LookupTable(71293,200)
            --self.emb:clearState()
            --local b = a:forward(lsent_ids)
            --local b = a:forward(lsent_ids)
            local linputs = self.lemb:forward(lsent_ids)
            --print (lsent_ids)
            local rinputs = self.remb:forward(rsent_ids)
            --print(rinputs)
            local output = self.model:forward({linputs,rinputs})
            --local linput = self.llstm:forward(linputs)
            --self.llstm.layer:forget()
            --self:forwardConnect(self.llstm,self.rlstm)
            --local rinput = self.rlstm:forward(rinputs)
            --local inputs = {linput,rinput}
                -- compute relatedness
            --inputs[1] = torch.Tensor(inputs[1]:size()):fill(100)
            --inputs[2] = torch.zeros(inputs[1]:size()):fill(1000)
            --print (inputs[1])
            --local output = self.sim_module:forward(inputs)
            --print (targets)
                -- compute loss and backpropagate
            local loss = self.criterion:forward(output, targets)
            --print (loss_tensor)
            --loss = loss_tensor:sum()
            assert(loss == loss,'loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cl/cl?' )
            if loss0 == nil then loss0 = loss end
            --assert(loss < loss0 * 3,'loss is exploding, aborting.')
            --print (targets)
            local sim_grad = self.criterion:backward(output, targets)
            --print (sim_grad)
            --local rep_grad = self.sim_module:backward(inputs, sim_grad)
            --local lgrad,rgrad
                --lgrad = torch.zeros(self.seq_length,self.batch_size,self.mem_dim)
            --lgrad = rep_grad[1]
                --rgrad = torch.zeros(self.seq_length,self.batch_size,self.mem_dim)
            --rgrad = rep_grad[2]
            self.model:backward({linputs,rinputs},sim_grad)
            --self.llstm:backward(linputs,lgrad)

            --self:backwardConnect(self.llstm, self.rlstm,self.finetune)
            --self.rlstm:backward(rinputs,rgrad)
            --self:backwardConnect(self.llstm, self.rlstm)
            --self.llstm.layer:forget()
            --self.llstm:backward(linputs,lgrad)
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



-- Predict the similarity of a sentence pair.
function LSTMSim:predict(lsent_ids, rsent_ids)
    self.model:evaluate()
    --self.rlstm:evaluate()
    --self.sim_module:evaluate()
    self.grad_params:zero()
    local linputs = self.lemb:forward(lsent_ids)

    local rinputs = self.remb:forward(rsent_ids)

    --print (rinputs:size())
    --print (self.emb.weight:size())
    --print (linputs:size())
    --print (rinputs:size())
    --print (lsent_ids[1][3])
    --print (rsent_ids[1][3])
    --print (linputs)
    --print (rinputs[1][3][10])

    local output = self.model:forward({linputs,rinputs})
    --print (output[2][1])
    --print (output)
    local size = output:size(1)
    local prediction = torch.Tensor(size)
    --print (output)
    for i = 1, size do
        --print (i)
        if output[i][1] > 0.5 then prediction[i] = 1 else prediction[i] = 0 end
    end
    self.lemb:clearState()
    self.remb:clearState()
    --self.llstm:forget()
    --self.rlstm:forget()
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
    local config = {
        batch_size    = self.batch_size,
        learning_rate = self.learning_rate,
        mem_dim       = self.mem_dim,
        emb_dim       = self.emb_dim,
        sim_nhidden   = self.sim_nhidden,
        reg           = self.reg,
        structure     = self.structure,
        seq_length    = self.seq_length,
        gpuidx        = self.gpuidx
    }

    torch.save(path, {
        lstm_params = self.lstm_params,
        config = config,
    })
end

function LSTMSim.load(path)
    local state = torch.load(path)
    local model = treelstm.LSTMSim.new(state.config)
    model.lstm_params:copy(state.lstm_params)
    return model
end

