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
