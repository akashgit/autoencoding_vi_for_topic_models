# Autoencoding Variational Inference for Topic Models

## Code for the ICLR 2017 paper: Autoencoding Variational Inference for Topic Models

#### > [Arxiv](https://arxiv.org/abs/1703.01488)

#### > [OpenReview](http://openreview.net/forum?id=BybtVK9lg)

#### Quick Start:

This is a tensorflow implementation for both of the Autoencoded Topic Models mentioned in the paper.  
---
To run the `prodLDA` model in the `20Newgroup` dataset:

> `CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 200`

Similarly for `NVLDA`:

> `CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300`

Check `run.py` for other options.

__UPDATE__

[@hyqneuron](https://github.com/hyqneuron) recently implemented a [PyTorch](https://github.com/hyqneuron/pytorch-avitm) version of AVITM. So check out his repo.

Added `topic_prop` method to both the models. Softmax the output of this method to get the topic proportions.
