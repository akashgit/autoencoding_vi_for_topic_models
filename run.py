#!/usr/bin/python

import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib.pyplot as plt
import pickle
import sys, getopt
from models import prodlda, nvlda
'''-----------Data--------------'''
def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

dataset_tr = 'data/20news_clean/train.txt.npy'
data_tr = np.load(dataset_tr)
dataset_te = 'data/20news_clean/test.txt.npy'
data_te = np.load(dataset_te)
vocab = 'data/20news_clean/vocab.pkl'
vocab = pickle.load(open(vocab,'r'))
vocab_size=len(vocab)
#--------------convert to one-hot representation------------------
print 'Converting data to one-hot representation'
data_tr = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
data_te = np.array([onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
#--------------print the data dimentions--------------------------
print 'Data Loaded'
print 'Dim Training Data',data_tr.shape
print 'Dim Test Data',data_te.shape
'''-----------------------------'''

'''--------------Global Params---------------'''
n_samples_tr = data_tr.shape[0]
n_samples_te = data_te.shape[0]
docs_tr = data_tr
docs_te = data_te
batch_size=200
learning_rate=0.002
network_architecture = \
    dict(n_hidden_recog_1=100, # 1st layer encoder neurons
         n_hidden_recog_2=100, # 2nd layer encoder neurons
         n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
         n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
         n_z=50)  # dimensionality of latent space

'''-----------------------------'''

'''--------------Netowrk Architecture and settings---------------'''

def make_network(layer1=100,layer2=100,num_topics=50,bs=200,eta=0.002):
    tf.reset_default_graph()
    network_architecture = \
        dict(n_hidden_recog_1=layer1, # 1st layer encoder neurons
             n_hidden_recog_2=layer2, # 2nd layer encoder neurons
             n_hidden_gener_1=data_tr.shape[1], # 1st layer decoder neurons
             n_input=data_tr.shape[1], # MNIST data input (img shape: 28*28)
             n_z=num_topics)  # dimensionality of latent space
    batch_size=bs
    learning_rate=eta
    return network_architecture,batch_size,learning_rate



'''--------------Methods--------------'''
def create_minibatch(data):
    rng = np.random.RandomState(10)

    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        ixs = rng.randint(data.shape[0], size=batch_size)
        yield data[ixs]


def train(network_architecture, minibatches, type='prodlda',learning_rate=0.001,
          batch_size=200, training_epochs=100, display_step=5):
    tf.reset_default_graph()
    vae=''
    if type=='prodlda':
        vae = prodlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size)
    elif type=='nvlda':
        vae = nvlda.VAE(network_architecture,
                                     learning_rate=learning_rate,
                                     batch_size=batch_size)
    emb=0
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples_tr / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = minibatches.next()
            # Fit training using batch data
            cost,emb = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples_tr * batch_size

            if np.isnan(avg_cost):
                print epoch,i,np.sum(batch_xs,1).astype(np.int),batch_xs.shape
                print 'Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.'
                # return vae,emb
                sys.exit()

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost)
    return vae,emb

def print_top_words(beta, feature_names, n_top_words=10):
    print '---------------Printing the Topics------------------'
    for i in range(len(beta)):
        print(" ".join([feature_names[j]
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]))
    print '---------------End of Topics------------------'

def calcPerp(model):
    cost=[]
    for doc in docs_te:
        doc = doc.astype('float32')
        n_d = np.sum(doc)
        c=model.test(doc)
        cost.append(c/n_d)
    print 'The approximated perplexity is: ',(np.exp(np.mean(np.array(cost))))

def main(argv):
    m = ''
    f = ''
    s = ''
    t = ''
    b = ''
    r = ''
    e = ''
    try:
      opts, args = getopt.getopt(argv,"hpnm:f:s:t:b:r:,e:",["default=","model=","layer1=","layer2=","num_topics=","batch_size=","learning_rate=","training_epochs"])
    except getopt.GetoptError:
        print 'CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1] -e <training_epochs>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'CUDA_VISIBLE_DEVICES=0 python run.py -m <model> -f <#units> -s <#units> -t <#topics> -b <batch_size> -r <learning_rate [0,1]> -e <training_epochs>'
            sys.exit()
        elif opt == '-p':
            print 'Running with the Default settings for prodLDA...'
            print 'CUDA_VISIBLE_DEVICES=0 python run.py -m prodlda -f 100 -s 100 -t 50 -b 200 -r 0.002 -e 100'
            m='prodlda'
            f=100
            s=100
            t=50
            b=200
            r=0.002
            e=100
        elif opt == '-n':
            print 'Running with the Default settings for NVLDA...'
            print 'CUDA_VISIBLE_DEVICES=0 python run.py -m nvlda -f 100 -s 100 -t 50 -b 200 -r 0.005 -e 300'
            m='nvlda'
            f=100
            s=100
            t=50
            b=200
            r=0.01
            e=300
        elif opt == "-m":
            m=arg
        elif opt == "-f":
            f=int(arg)
        elif opt == "-s":
            s=int(arg)
        elif opt == "-t":
            t=int(arg)
        elif opt == "-b":
            b=int(arg)
        elif opt == "-r":
            r=float(arg)
        elif opt == "-e":
            e=int(arg)

    minibatches = create_minibatch(docs_tr.astype('float32'))
    network_architecture,batch_size,learning_rate=make_network(f,s,t,b,r)
    print network_architecture
    print opts
    vae,emb = train(network_architecture, minibatches,m, training_epochs=e,batch_size=batch_size,learning_rate=learning_rate)
    print_top_words(emb, zip(*sorted(vocab.items(), key=lambda x: x[1]))[0])
    calcPerp(vae)

if __name__ == "__main__":
   main(sys.argv[1:])
