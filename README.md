# Neural Variational Inference for Topic Models

## Code for the ICLR 2017 paper: Neural Variational Inference for Topic Models

#### > [PDF](http://openreview.net/pdf?id=BybtVK9lg)

#### > [OpenReview](http://openreview.net/forum?id=BybtVK9lg)

#### Quick Start:

This is a tensorflow implementation for both the Neural Topic Models mentioned in the paper. To run the `prodLDA` model in the `20Newgroup` dataset:
---
> `CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 80`

Similarly for `NVLDA`:

> `CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300`

Check `run.py` for other options.

