--
-- User: zhuangli
-- Date: 25/07/2016
-- Time: 4:03 PM
-- Gradient test, test whether the LSTM implementation is correct
--
require('..')
local ok, cunn = pcall(require, 'clnn')
local ok2, cutorch = pcall(require, 'cltorch')
if not ok then print('package clnn not found!') end
if not ok2 then print('package cltorch not found!') end
if ok and ok2 then
    print('using OpenCL on GPU 1 ...')
    cltorch.setDevice(1) -- note +1 to make it 0 indexed! sigh lua
    torch.manualSeed(1)
end
local emb_dim = 10
local step = 5
local seq = torch.rand(step,emb_dim)
seq = seq:cl()


function forward(model, criterion, regression)
    outputs = model:forward(seq)
    output = outputs:double()
    top_out = regression:forward(output)
    target = torch.Tensor(1,1)
    target[1] = 0.5
    loss = criterion:forward(top_out,target)
    printf('loss = %f \n', loss)
    return loss
end

function test_module(model, model_name,word_idx)
    print("test " .. model_name)
    local criterion = nn.MSECriterion()
    criterion.sizeAverage = true
    local regression= nn.Linear(emb_dim, 1)

    forward(model, criterion,regression)
    -- backpropagation
    local criterion_grad = criterion:backward(top_out, target)
    grad_reg = regression:backward(output, criterion_grad)
    local grad = torch.zeros(step,emb_dim)
    grad[step] = grad_reg
    input_errors = model:backward(seq, grad:cl())
    input_errors = input_errors
    elem_idx = 3
    origin_value = seq[word_idx][elem_idx]
    delta = 0.001
    seq[word_idx][elem_idx] = origin_value + delta
    loss_plus = forward(model,criterion,regression)
    model:forget()

    seq[word_idx][elem_idx] = origin_value - delta
    loss_minus = forward(model,criterion,regression)

    diff = (loss_plus - loss_minus) / (2 * delta)
    printf("%d : %s finite difference is %f and the partial derivative computed by backProp is %f \n", word_idx, model_name, diff, input_errors[word_idx][elem_idx])
    error_range = 0.00001
    assert((input_errors[word_idx][elem_idx] < diff + error_range) and (input_errors[word_idx][elem_idx] > diff - error_range), "test of ".. model_name .." failed")
    print("Pass the gradient test of " .. model_name .. "!")
end

local lstm = sentenceSim.LSTM{
    in_dim = 10,
    mem_dim = 10,
    batch_size = 1,
    seq_length = step,
    gpuidx = 1

}
test_module(lstm,"LSTM",5)