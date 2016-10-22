Normalization with End-to-End Attention Model for Learning Tweet Similarity
===========================================================================
## About this code

This code was developed by me for our paper *Normalization with End-to-End Attention Model for Learning Tweet Similarity*. However， it is only an implementation for the original idea. We've already modified it and migrated our code to another platform. The final version of code along with the explanation of ideas and the full experiment results for the paper would be released after paper published.

## For the Code Reviewers of MILA

Although it is not an official experiment code, I think it is good as the sample code. It is not long ,easy to read and clean.<br>
**Note** The code, which is written by [Element-Research](https://github.com/Element-Research/), in `examples` is only for learning by myself. So if you want to check my code, `sentenceSim` includes the main experiment code, `utils` includes the pre-processing code, `tests` includes gradient check code and `models` includes a LSTM model.

## Requirements

- [Torch7](https://github.com/torch/torch7)
- [nn](https://github.com/torch/nn)
- [nngraph](https://github.com/torch/nngraph)
- [optim](https://github.com/torch/optim)
- [dpnn](https://github.com/Element-Research/dpnn)
- [rnn](https://github.com/Element-Research/rnn)

The Torch/Lua dependencies can be installed using [luarocks](http://luarocks.org). For example:

```
luarocks install nngraph
```


## Usage

First download the following word embedding file into `data/embedding`:

- [Glove Wikipedia 2014 + Gigaword 5](http://nlp.stanford.edu/data/glove.6B.zip)

Then run the following script:

```
./text2th.sh
```
This `upzip` the file `glove.6B.zip` and get text embedding file `glove.6B.200d.txt`. Then convert the `glove.6B.200d.txt` to torch format embedding file `glove.6B.200d.th` and text vocabulary file `glove.6B.vocab`.
For the data set, we use [PIT corpus](http://www.cis.upenn.edu/ xwe/ semeval2015pit) for pre-training and fine-tuning. It is already pushed on the github.
**Note** Pre-training is supposed to use a larger corpus. However, here we just present our code so we simplify the setting and use a small corpus instead. If require a standard experiment, please collect a larger dataset and pre-process it to fit the input requirements of this model.

### About the experiment

This experiment involves two phases,unsupervised pre-training and supervised fine-tuning. 

We use seq2seq model[[1]](https://arxiv.org/pdf/1409.3215v3.pdf) to do pre-training. The encoder encodes a sentence without out-of-vocabulary tokens and decoder decodes itself. The architecture is as the following figure:
![seq2seq](http://i64.tinypic.com/30136te.png)

Then we perserve the model of encoder. During fine-tuning, we use a Siamese LSTM[[2]](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf) to predict semantic similarity between sentences. To learn this model, instead of randomly initializing model parameters , we adopt the parameters of the encoder perserved from pre-training. And we learn the model towards our objective. The architecture is as the following figure:
![siamese](http://i64.tinypic.com/30136te.png)

### Pre-train
To do unsupervised pre-training for representation learning task,
run:

```
th sentenceSim/PreTrain.lua -data_dir <pathToDataDir> -hidden_size <numOfHiddenUnits> -model <lstm|gru|rnn> -learning_rate <learningRate> -reg <alphaOfL2Norm> -max_epochs <maxEpochs> -seq_length <lengthOfTimeStep> -batch_size <batchSize> -gpuidx <GPUIndex> -model_name <NameOfSavingModel>
```

where:

	`-data_dir`, Path to the `data` directory. `data` should contain the data directories, embedding directory and model serialization directory. (default: data)
    `-hidden_size`, Size of LSTM internal state. (default: 200)
    `-model`, The LSTM variant to train. Now only LSTM module is implemented. (default: lstm)
    `-learning_rate`, Learning rate. (default: 0.05)
    `-reg`, Regularization rate. (default: 0.0001)
    `-max_epochs`, Max number of training epochs(default: 15)
    `-seq_length`, Number of timesteps to unroll for (default: 10)
    `-batch_size', Number of sequences to train on in parallel(default: 100)
    `-gpuidx`, Which gpu to use, 0 = use CPU. We haven't implemented the GPU mode(default: 10)
    `-model_name`, Name of the model to be serialized, model will be stored in `data/model_ser`. (default: /model_name)

Example:

```
th sentenceSim/PreTrain.lua -data_dir data -hidden_size 200 -model lstm -learning_rate 0.05 -reg 0.0001 -max_epochs 10 -seq_length 10 -batch_size 50 -model_name pretrain_model
```

### Fine-tine
To do supervised fine-tuning for the semantic relatedness prediction task on the PIT dataset,
run:

```
th sentenceSim/FineTune.lua -data_dir <pathToDataDir> -hidden_size <numOfHiddenUnits> -model <lstm|gru|rnn> -learning_rate <learningRate> -reg <alphaOfL2Norm> -max_epochs <maxEpochs> -seq_length <lengthOfTimeStep> -batch_size <batchSize> -gpuidx <GPUIndex> -load <f|NameOfPretrainedModel>
```

where:

	`-data_dir`, Path to the `data` directory. Should contain the data directories and embedding directory. (default: data)
    `-hidden_size`, Size of LSTM internal state. (default: 200)
    `-model`, The LSTM variant to train. Now only LSTM module is implemented. (default: lstm)
    `-learning_rate`, Learning rate. (default: 0.01)
    `-reg`, Regularization rate. (default: 0.0001)
    `-max_epochs`, Max number of training epochs(default: 15)
    `-seq_length`, Number of timesteps to unroll for (default: 10)
    `-batch_size', Number of sequences to train on in parallel(default: 100)
    `-gpuidx`, Which gpu to use, 0 = use CPU. We haven't implemented the GPU mode(default: 10)
    `-load`, Load pre-trained model or not. If `f`, pre-trained model will not be loaded otherwise the model with `NameOfPretrainedModel` in `data/model_ser` will be loaded. (default: f)

Example without loading pretrained model:

```
th sentenceSim/FineTune.lua -data_dir data -hidden_size 200 -model lstm -learning_rate 0.01 -reg 0.0001 -max_epochs 10 -seq_length 10 -batch_size 50 -load f
```

Example with loading pretrained model:

```
th sentenceSim/FineTune.lua -data_dir data -hidden_size 200 -model lstm -learning_rate 0.01 -reg 0.0001 -max_epochs 10 -seq_length 10 -batch_size 50 -load pretrain_model
```

### Reference

* [ [1] Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [ [2] Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)