--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 25/07/2016
-- Time: 4:04 PM
-- To change this template use File | Settings | File Templates.
--
sentenceSim = {}
require('torch')
--require('cutorch')
require('nn')
--require('cunn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('cjson')
include('models/LSTM.lua')
printf = utils.printf
--include('tests/LSTMTest.lua')

function share_params(cell, src)
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