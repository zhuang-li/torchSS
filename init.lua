--
-- Created by IntelliJ IDEA.
-- User: zhuangli
-- Date: 25/07/2016
-- Time: 4:04 PM
-- init file, init global variables
--
sentenceSim = {}
require('torch')
require('nn')
require('rnn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')
require('cjson')
include('sentenceSim/LSTMSim.lua')
include('models/LSTM.lua')
include('utils/data.lua')
include('utils/vocab.lua')
printf = utils.printf

function header(s)
    print(string.rep('-', 80))
    print(s)
    print(string.rep('-', 80))
end