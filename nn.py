import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow import *


# Load data
data = load_boston()
X_ = data['data']
Y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
n_hidden = 10

# Define parameters

W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Define network

X = Input()
Y = Input()

W1 = Input()
b1 = Input()
W2 = Input()
b2 = Input()

l1 = Linear(X, W1, b1)
s1 = Sigmoid(l1)
l2 = Linear(s1, W2, b2)
cost = MSE(Y, l2)

feed_dict = {
        X: X_,
        Y: Y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
}

# SGD

epochs = 1000
m = X_.shape[0]
batch_size = 11
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        X_batch, Y_batch = resample(X_, Y_, n_samples=batch_size)

        X.value = X_batch
        Y.value = Y_batch

        forward_and_backward(graph)
        sgd_update(trainables)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))
