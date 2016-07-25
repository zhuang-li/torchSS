--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 25/07/2016
-- Time: 2:25 PM
-- To change this template use File | Settings | File Templates.
--

--[[

 Long Short-Term Memory.

--]]

local LSTM, parent = torch.class('sentenceSim.LSTM', 'nn.Module')

function LSTM:__init(config)
    parent.__init(self)

    self.in_dim = config.in_dim
    self.mem_dim = config.mem_dim or 200
    self.gate_output = config.gate_output
    if self.gate_output == nil then self.gate_output = true end

    self.master_cell = self:new_cell()
    self.depth = 0
    self.cells = {}  -- table of cells in a roll-out

    -- initial (t = 0) states for forward propagation and initial error signals
    -- for backpropagation
    local c_init = torch.zeros(self.mem_dim)
    local h_init = torch.zeros(self.mem_dim)
    local c_grad = torch.zeros(self.mem_dim)
    local h_grad = torch.zeros(self.mem_dim)
    self.initial_values = {c_init, h_init}
    self.gradInput = {
        torch.zeros(self.in_dim),
        c_grad,
        h_grad
    }
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

    -- share parameters
    if self.master_cell then
        self:share_params(cell, self.master_cell)
    end
    return cell
end

-- Forward propagate.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- reverse: if true, read the input from right to left (useful for bidirectional LSTMs).
-- Returns the final hidden state of the LSTM.
function LSTM:forward(inputs, reverse)
    local size = inputs:size(1)
    --print (size)
    local output = torch.zeros(size,self.mem_dim)
    --print (inputs)
    for t = 1, size do
        local input = reverse and inputs[size - t + 1] or inputs[t]
        self.depth = self.depth + 1
        local cell = self.cells[self.depth]
        if cell == nil then
            cell = self:new_cell()
            self.cells[self.depth] = cell
        end
        local prev_output
        if self.depth > 1 then
            prev_output = self.cells[self.depth - 1].output
        else
            prev_output = self.initial_values
        end
        --print (prev_output[1])
        local outputs = cell:forward({input, prev_output[1], prev_output[2]})
        local c, h = unpack(outputs)
        --print (c)
        --print (h)
        output[t] = h
    end
    return output
end

-- Backpropagate. forward() must have been called previously on the same input.
-- inputs: T x in_dim tensor, where T is the number of time steps.
-- grad_outputs: T x num_layers x mem_dim tensor.
-- reverse: if true, read the input from right to left.
-- Returns the gradients with respect to the inputs (in the same order as the inputs).
function LSTM:backward(inputs, grad_outputs, reverse)
    local size = inputs:size(1)
    if self.depth == 0 then
        error("No cells to backpropagate through")
    end

    local input_grads = torch.Tensor(inputs:size())
    for t = size, 1, -1 do
        local input = reverse and inputs[size - t + 1] or inputs[t]
        local grad_output = reverse and grad_outputs[size - t + 1] or grad_outputs[t]
        local cell = self.cells[self.depth]
        local grads = {self.gradInput[2], self.gradInput[3]}
        grads[2]:add(grad_output)
        --print (grads[2])
        local prev_output = (self.depth > 1) and self.cells[self.depth - 1].output
                or self.initial_values
        self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
        if reverse then
            input_grads[size - t + 1] = self.gradInput[1]
        else
            input_grads[t] = self.gradInput[1]
        end

        self.depth = self.depth - 1
    end
    self:forget() -- important to clear out state
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
    self.depth = 0
    for i = 1, #self.gradInput do
        self.gradInput[i]:zero()
    end
end

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
