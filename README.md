# Autoencoding Variational Inference for Topic Models

__UPDATE__

1. As pointed out by [@govg](https://github.com/govg), this code depends on a slightly older version of TF. I will try to update it soon, in the meantime you can look up a quick fix [here](https://github.com/akashgit/autoencoding_vi_for_topic_models/issues/5) for working with newer version of TF or (3) and (2) below if you'd rather prefer Keras or PyTorch.

2. [@nzw0301](https://github.com/nzw0301) has implemented a [Keras](https://github.com/nzw0301/keras-examples/blob/master/prodLDA.ipynb) version of prodLDA.

3. [@hyqneuron](https://github.com/hyqneuron) recently implemented a [PyTorch](https://github.com/hyqneuron/pytorch-avitm) version of AVITM. So check out his repo.

4. Added `topic_prop` method to both the models. Softmax the output of this method to get the topic proportions.

---
#### Code for the ICLR 2017 paper: Autoencoding Variational Inference for Topic Models
---

#### > [Arxiv](https://arxiv.org/abs/1703.01488)

#### > [OpenReview](http://openreview.net/forum?id=BybtVK9lg)

---
###### This is a tensorflow implementation for both of the Autoencoded Topic Models mentioned in the paper.  
---
To run the `prodLDA` model in the `20Newgroup` dataset:

> `CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 200`

Similarly for `NVLDA`:

> `CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300`

Check `run.py` for other options.


