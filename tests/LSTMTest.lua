--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 25/07/2016
-- Time: 4:03 PM
-- To change this template use File | Settings | File Templates.
--
require('..')
--require ('nn')
local emb_dim = 10
local step = 5
local seq = torch.rand(step,emb_dim)



function forward(model, criterion, regression)
    outputs = model:forward(seq,true)
    --print (outputs)
    output = outputs[step]
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
    grad[1] = grad_reg
    input_errors = model:backward(seq, grad,true)
    --print (input_errors)

    elem_idx = 3
    --print(seq[word_idx][elem_idx])
    origin_value = seq[word_idx][elem_idx]
    delta = 0.001
    seq[word_idx][elem_idx] = origin_value + delta
    --print(seq[word_idx][elem_idx])
    loss_plus = forward(model,criterion,regression)
    model:forget()
    --printf ('output forward = %f \n', loss_plus)

    seq[word_idx][elem_idx] = origin_value - delta
    --print(seq[word_idx][elem_idx])
    loss_minus = forward(model,criterion,regression)
    --printf ('output backward = %f \n', loss_minus)
    diff = (loss_plus - loss_minus) / (2 * delta)
    printf("%d : %s finite difference is %f and the partial derivative computed by backProp is %f \n", word_idx, model_name, diff, input_errors[word_idx][elem_idx])
    error_range = 0.00001
    assert((input_errors[word_idx][elem_idx] < diff + error_range) and (input_errors[word_idx][elem_idx] > diff - error_range), "test of ".. model_name .." failed")
    print("Pass the gradient test of " .. model_name .. "!")
end

local lstm = sentenceSim.LSTM{
    in_dim = 10,
    mem_dim = 10

}
test_module(lstm,"LSTM",5)