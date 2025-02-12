# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

sets up data sets for use in convolutional network and defines
the network. To change number of layers, you need to modify the
definition of modelCNN near end.


The CNN is intended to reduce input receptive field to a single pixel during 
training. NC0 = input channels, NC1, etc. number of features in subsequent 
layers. Final number of channels should be = 1. Each convolution reduces the 
dimension of the field by K-1 where K = dimension of kernel. Therefore...

N - (K1-1) - (K2-1) - ... -(KL) = 1

for NxN training images, and L hidden layers. 

To create fully connected end layers, use kernel sizes of 1x1.

Note: for max pool layer, must devide by 

VARIABLES NEEDED: X_train, Y_train, X_test, Y_test
NAMES DEFINED: modelCNN, xtrain, ytrain, xtest, ytest, loss_fn   

notes: NC1=6, NC3 = 36, NC4 = 6 has a little overfitting 

"""

# define channels and kernel sizes for layers
#                   description                             size 
NC0 = 1             # layer 0: input                        9x9
K1, NC1 = 4, 6      # layer 1: convolution of layer         6x6
K2, NC2 = 2, 6      # layer 2: max pool of layer 1          3x3
K3, NC3 = 3, 24   # layer 3: convolution of layer 2       1x1
K4, NC4 = 1, 6     # layer 4: fully connected to layer 3   1x1
K5, NC5 = 1, 1      # layer 5: output layer (FC to 4)        1x1

# define the activation function. 
activationFunction = nn.ReLU()

##############################################################################
##############################################################################
# prepare training data.  Include flattening and scaling of X
numberTrainSamples, ydim, xdim = X_train.shape
xtrain = torch.FloatTensor( 
                            X_train[:,np.newaxis,:,:] \
                            / X_train.max()
                           )
ytrain =  torch.FloatTensor(
                            Y_train[:,np.newaxis,np.newaxis,np.newaxis]
                            )
    
# prepare test data.  Include flattening and scaling of X
numberTestSamples, ydim, xdim = X_test.shape
xtest = torch.FloatTensor( 
                            X_test[:,np.newaxis,:,:] \
                            / X_test.max()
                           )
ytest =  torch.FloatTensor(
                            Y_test[:,np.newaxis,np.newaxis,np.newaxis]
                            )

modelCNN = nn.Sequential(
    nn.Conv2d( NC0, NC1, kernel_size=K1, stride=1, padding=0 ), 
    activationFunction,
    nn.MaxPool2d( kernel_size = K2, stride = K2 ),
    nn.Conv2d( NC2, NC3, kernel_size=K3, stride=1, padding=0 ),
    activationFunction,
    nn.Conv2d( NC3, NC4, kernel_size=K4,stride=1, padding=0 ),
    activationFunction,
    nn.Conv2d( NC4, NC5, kernel_size=K5, stride=1, padding=0 ),
    nn.Sigmoid(),
    )
loss_fn = nn.BCELoss()

modelType = 'CNN'
