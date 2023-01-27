#!/usr/bin/env python
# coding: utf-8

# In[4]:


import argparse
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
import numpy as np
from tqdm import tqdm

# argparse for parameter values
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20, help='max epoch')
    parser.add_argument('--lr', type=int, default=0.0001, help='learning rate')
    parser.add_argument('--size', type=int, default=4, help='batch size')
    return parser.parse_args()

# Define model using tinygrad
class TinyModel():
    # Generate random initial c_preds values
    def __init__(self, degree, batch, args):
        self.n = degree + 1
        self.epoch = epoch
        self.batch = batch
        self.c_preds = Tensor.uniform(self.n,1)

    # Calculate y_pred using c_preds and x values
    def forward(self,x):
        y_pred = 0
        for i in range(self.n):
            y_pred += self.c_preds[i]*(x**i)
        return y_pred  

# Train model using SGD optimizer
def train(train_set, optimizer, batch, args):
    n = 0
    for i in tqdm(range(epoch)):
        optimizer.zero_grad()
        epoch_loss = 0
        for j in range(batch):
            y_pred = model.forward(train_set[n][0])
            g_truth = train_set[n][1]
            # Calculate loss using MSE
            mse = (g_truth - y_pred)**2
            epoch_loss += mse
            n += 1
        epoch_loss *= (1/batch)
        epoch_loss.backward()
        optimizer.step()

# Loss function for evaluation using test set
def calc_loss(test_set):
    loss = 0
    n = 0
    for i in range(max(test_set.shape)):
        prediction = model.forward(test_set[n][0])
        mse = (test_set[n][1] - prediction)**2 #Mean Square Error
        loss += mse
        n += 1
    loss *= (1/max(test_set.shape))
    return loss

def main(args):
    # Read data_train and data_test csv files then shuffle order
    train_set = np.loadtxt('data_train.csv', skiprows=1, delimiter=',')
    np.random.shuffle(train_set)
    test_set = np.loadtxt('data_test.csv', skiprows=1, delimiter=',')
    np.random.shuffle(test_set)

    # Number of batces
    batch = round(len(train_set)/size)

    # Convert read data into tinygrad Tensor
    train_set = Tensor(train_set, requires_grad = True)

    # Evaluate each model for different degrees and return cofficients with the least loss
    # Degree < 5 due to Abel-Ruffini Theorem

    degree = 1
    print('Calculating coefficients for degree = {}'.format(degree))
    model = TinyModel(degree, epoch, batch)
    optimizer1 = optim.SGD([model.c_preds], lr = lr)
    train(train_set, optimizer1, epoch, batch)
    c_preds1 = model.c_preds.data
    loss1 = calc_loss(test_set)
    least_loss = loss1.data

    degree = 2
    print('Calculating coefficients for degree = {}'.format(degree))
    model = TinyModel(degree, epoch, batch)
    optimizer2 = optim.SGD([model.c_preds], lr = lr)
    train(train_set, optimizer2, epoch, batch)
    c_preds2 = model.c_preds.data
    loss2 = calc_loss(test_set)
    if (loss2.data < least_loss) : least_loss = loss2.data

    degree = 3
    print('Calculating coefficients for degree = {}'.format(degree))
    model = TinyModel(degree, epoch, batch)
    optimizer3 = optim.SGD([model.c_preds], lr = lr)
    train(train_set, optimizer3, epoch, batch)
    c_preds3 = model.c_preds.data
    loss3 = calc_loss(test_set)
    if (loss3.data < least_loss) : least_loss = loss3.data

    degree = 4
    print('Calculating coefficients for degree = {}'.format(degree))
    model = TinyModel(degree, epoch, batch)
    optimizer4 = optim.SGD([model.c_preds], lr = lr)
    train(train_set, optimizer4, epoch, batch)
    c_preds4 = model.c_preds.data
    loss4 = calc_loss(test_set)
    if (loss4.data < least_loss) : least_loss = loss4.data

    if(loss4.data == least_loss):
        print(f"Degree: 4\nPredicted coeffs: \n{c_preds4}")
    elif(loss3.data == least_loss):
        print(f"Degree: 3\nPredicted coeffs: \n{c_preds3}")
    elif(loss2.data == least_loss):
        print(f"Degree: 2\nPredicted coeffs: \n{c_preds2}")
    else:
        print(f"Degree: 1\nPredicted coeffs: \n{c_preds1}")

if '__name__' == '__main__':
    args = argparser()
    main(args)
    # epoch = 20
    # lr = 0.0001
    # size = 4

