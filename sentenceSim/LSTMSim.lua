--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 27/07/2016
-- Time: 5:26 PM
-- To change this template use File | Settings | File Templates.
--


local LSTMSim = torch.class('treelstm.LSTMSim')

function LSTMSim:__init(config)
    self.mem_dim       = config.mem_dim       or 200
    self.learning_rate = config.learning_rate or 0.05
    self.batch_size    = config.batch_size    or 100
    self.reg           = config.reg           or 1e-4
    self.seq_length     = config.seq_length or 20
    self.gpuidx = config.gpuidx or 0
    if self.gpuidx > 0 then
        local ok, cunn = pcall(require, 'clnn')
        local ok2, cutorch = pcall(require, 'cltorch')
        if not ok then print('package clnn not found!') end
        if not ok2 then print('package cltorch not found!') end
        if ok and ok2 then
            print('using OpenCL on GPU 1 ...')
            cltorch.setDevice(self.gpuidx) -- note +1 to make it 0 indexed! sigh lua
            torch.manualSeed(self.gpuidx)
        end
    end
    self.structure     = config.structure     or 'lstm' -- {lstm, bilstm}
    self.sim_nhidden   = config.sim_nhidden   or 50

    -- word embedding
    self.emb_vecs = config.emb_vecs
    self.emb_dim = config.emb_vecs:size(2)

    -- number of similarity rating classes

    -- optimizer configuration
    self.optim_state = { learningRate = self.learning_rate }

    -- KL divergence optimization objective
    if self.gpuidx > 0 then self.criterion = nn.BCECriterion():cuda() else self.criterion = nn.BCECriterion() end

    -- initialize LSTM model
    local lstm_config = {
        in_dim = self.emb_dim,
        mem_dim = self.mem_dim,
        seq_length = self.seq_length,
        gpuidx = self.gpuidx
    }

    self.llstm = treelstm.LSTM(lstm_config) -- "left" LSTM
    self.rlstm = treelstm.LSTM(lstm_config) -- "right" LSTM

    -- similarity model
    self.sim_module = self:new_sim_module()
    local modules = nn.Parallel()
    :add(self.llstm)
    :add(self.sim_module)
    self.params, self.grad_params = modules:getParameters()

    -- share must only be called after getParameters, since this changes the
    -- location of the parameters
    share_params(self.rlstm, self.llstm)
end

function LSTMSim:new_sim_module()

    local linput, rinput = nn.Identity()(), nn.Identity()()
    local mult_dist = nn.CMulTable(){linput, rinput}
    local add_dist = nn.Abs()(nn.CSubTable(){linput, rinput})
    local vec_dist_feats = nn.JoinTable(1){mult_dist, add_dist}
    local vecs_to_input = nn.gModule({linput,rinput}, {vec_dist_feats})

    -- define similarity model architecture
    local sim_module = nn.Sequential()
    :add(vecs_to_input)
    :add(nn.Linear(2 * self.mem_dim, self.sim_nhidden))
    :add(nn.Sigmoid())    -- does better than tanh
    :add(nn.Linear(self.sim_nhidden, 1))
    :add(nn.Sigmoid())
    if self.gpuidx > 0 then
        return sim_module:cuda()
    else
        return sim_module
    end
end

function LSTMSim:train(dataset)
    self.llstm:training()
    self.rlstm:training()

    local max_batches = dataset.max_batches
    local indices = torch.randperm(max_batches)

    for i = 1, max_batches do

        xlua.progress((i-1)*self.batch_size, dataset.size)

        local feval = function(x)
            self.grad_params:zero()
            local idx = indices[i]
            local targets = dataset.labels[idx]
            local lsent_ids, rsent_ids = dataset.lsents[idx], dataset.rsents[idx]
            local linputs = self.emb_vecs:forward(lsent_ids)
            local rinputs = self.emb_vecs:forward(rsent_ids)


            local inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}

                -- compute relatedness
            local output = self.sim_module:forward(inputs)

                -- compute loss and backpropagate
            local loss = self.criterion:forward(output, targets)
            assert(loss == loss,'loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?' )
            if loss0 == nil then loss0 = loss end
            assert(loss < loss0 * 3,'loss is exploding, aborting.')

            local sim_grad = self.criterion:backward(output, targets)
            local rep_grad = self.sim_module:backward(inputs, sim_grad)
            local lgrad,rgrad
            if self.gpuidx > 0 then
                lgrad = torch.zeros(self.seq_length,self.batch_size,self.in_dim):cuda()
                lgrad[self.seq_length] = rep_grad[1]
            else
                rgrad = torch.zeros(self.seq_length,self.batch_size,self.in_dim)
                rgrad[self.seq_length] = rep_grad[2]
            end
            self.llstm:backward(linputs,lgrad)
            self.rlstm:backward(rinputs,rgrad)



            self.grad_params:div(self.batch_size)

            -- regularization
            loss = loss + 0.5 * self.reg * self.params:norm() ^ 2
            self.grad_params:add(self.reg, self.params)
            return loss, self.grad_params
        end

        optim.adagrad(feval, self.params, self.optim_state)
    end
    xlua.progress(dataset.size, dataset.size)
end



-- Predict the similarity of a sentence pair.
function LSTMSim:predict(lsent_ids, rsent_ids)
    self.llstm:evaluate()
    self.rlstm:evaluate()
    self.grad_params:zero()
    local linputs = self.emb_vecs:forward(lsent_ids)
    local rinputs = self.emb_vecs:forward(rsent_ids)


    local inputs = {self.llstm:forward(linputs), self.rlstm:forward(rinputs)}

    -- compute relatedness
    local output = self.sim_module:forward(inputs)
    local size = output:size(1)
    local prediction = torch.Tensor(size)
    for i = 1, size do
        if output[i] > 0.5 then prediction[i] = 1 else prediction[i] = 0 end
    end
    return prediction
end

-- Produce similarity predictions for each sentence pair in the dataset.
function LSTMSim:predict_dataset(dataset)
    local predictions = {}
    local max_batches = dataset.max_batches

    for i = 1, max_batches do
        xlua.progress((i-1)*self.batch_size, dataset.size)
        local lsent, rsent = dataset.lsents[i], dataset.rsents[i]
        predictions[i] = self:predict(lsent, rsent)
    end
    xlua.progress(self.size, dataset.size)
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
    printf('%-25s = %d\n',   'GPU index', self.gpuidx)
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
        params = self.params,
        config = config,
    })
end

function LSTMSim.load(path)
    local state = torch.load(path)
    local model = treelstm.LSTMSim.new(state.config)
    model.params:copy(state.params)
    return model
end

