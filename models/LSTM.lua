
--[[
--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 25/07/2016
-- Time: 2:25 PM
-- To change this template use File | Settings | File Templates.
--
 Long Short-Term Memory.

--]]

local LSTM, parent = torch.class('sentenceSim.LSTM', 'nn.Module')

function LSTM:__init(config)
    parent.__init(self)

    self.in_dim = config.in_dim or 200
    self.mem_dim = config.mem_dim or 200
    self.batch_size = config.batch_size or 100
    self.seq_length = config.seq_length or 20
    self.gpuidx = config.gpuidx or 0
    self.master_cell = self:new_cell()
    self.cells = self:clone_many_times(self.seq_length) -- table of cells in a roll-out

    -- initial (t = 0) states for forward propagation and initial error signals
    -- for backpropagation
    local c_init = torch.zeros(self.batch_size,self.mem_dim)
    local h_init = torch.zeros(self.batch_size,self.mem_dim)
    local c_grad = torch.zeros(self.batch_size,self.mem_dim)
    local h_grad = torch.zeros(self.batch_size,self.mem_dim)
    self.initial_values = {c_init, h_init}
    self.gradInput = {
        torch.zeros(self.batch_size,self.in_dim),
        c_grad,
        h_grad
    }
    if self.gpuidx > 0 then
        self.initial_values[1] = self.initial_values[1]:cl()
        self.initial_values[2] = self.initial_values[2]:cl()
        self.gradInput[1] = self.gradInput[1]:cl()
        self.gradInput[2] = self.gradInput[2]:cl()
        self.gradInput[3] = self.gradInput[3]:cl()
    end
end

-- share parameters between different LSTM cells

function LSTM:share_params(cell, src)
    if torch.type(cell) == 'nn.gModule' then
        for i = 1, #cell.forwardnodes do
            local node = cell.forwardnodes[i]
            if node.data.module then
                node.data.module:share(src.forwardnodes[i].data.module,
                    'weight', 'bias', 'gradWeight', 'gradBias')
            end
        end
    elseif torch.isTypeOf(cell, 'nn.Module') then
        cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('parameters cannot be shared for this input')
    end
end

-- Instantiate a new LSTM cell.
-- Each cell shares the same parameters, but the activations of their constituent
-- layers differ.
function LSTM:new_cell()
    local input = nn.Identity()()
    local c_p = nn.Identity()()
    local h_p = nn.Identity()()
    local function new_input_sum()
        local x_to_h = nn.Linear(self.in_dim,self.mem_dim)
        local h_to_h = nn.Linear(self.mem_dim, self.mem_dim)
        return nn.CAddTable()({x_to_h(input),h_to_h(h_p)})
    end
    local i = nn.Sigmoid()(new_input_sum())
    local f = nn.Sigmoid()(new_input_sum())
    local c_g = nn.Tanh()(new_input_sum())
    local c = nn.CAddTable()({
        nn.CMulTable()({f,c_p}),
        nn.CMulTable()({i,c_g})
    })
    local o_g = nn.Sigmoid()(new_input_sum())
    local h = nn.CMulTable()({o_g,nn.Tanh()(c)})
    local cell = nn.gModule({input, c_p, h_p}, {c, h})

    if self.gpuidx > 0 then
        return cell:cl()
    else
        return cell
    end

end
-- clone the cell for times of sequence length
function LSTM:clone_many_times(seq_length)
    local cells = {}
    for i = 1 , seq_length do
        cells[i] = self:new_cell()
        if self.master_cell then
            self:share_params(cells[i], self.master_cell)
        end
    end
    return cells
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- Returns the final hidden state of the LSTM.
function LSTM:forward(inputs)
    local output
    for t = 1, self.seq_length do
        local input = inputs[t]
        local cell = self.cells[t]
        local prev_output
        if t > 1 then
            prev_output = self.cells[t - 1].output
        else
            prev_output = self.initial_values
        end
        local outputs = cell:forward({input, prev_output[1], prev_output[2]})
        local c, h = unpack(outputs)
        output = h
    end
    return output
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function LSTM:backward(inputs, grad_outputs)

    local input_grads = {}

    for t = self.seq_length, 1, -1 do
        local input = inputs[t]
        local grad_output = grad_outputs[t]
        local cell = self.cells[t]
        local grads = {self.gradInput[2], self.gradInput[3] }
        grads[2]:add(grad_output)
        local prev_output = (t > 1) and self.cells[t - 1].output or self.initial_values
        self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
        input_grads[t] = self.gradInput[1]
    end
    self:forget()
    return input_grads
end


function LSTM:zeroGradParameters()
    self.master_cell:zeroGradParameters()
end


function LSTM:parameters()
    return self.master_cell:parameters()
end

-- Clear saved gradients
function LSTM:forget()
    for i = 1, #self.gradInput do
        self.gradInput[i]:zero()
    end
end


