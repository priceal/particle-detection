# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

sets up data sets for use in convolutional network and defines
the network. To change number of layers, you need to modify the
definition of modelCNN near end.

VARIABLES NEEDED: X_train, Y_train, X_test, Y_test
NAMES DEFINED: modelCNN, xtrain, ytrain, xtest, ytest, loss_fn    

"""

# this is a CNN that is intended to reduce input receptive field to a single
# pixel. NC0 = input channels, NC1, etc. number of features in subsequent 
# layers. K1, etc. are sizes of kernels. Note, kernel sizes must reduce
# input frame size to a single pixel. 
NC0, NC1, NC2, NC3 = 1, 4, 16, 1
K1, K2, K3 = 3, 3, 3

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
    nn.Conv2d(NC0,NC1,kernel_size=K1,stride=1,padding=0), 
    nn.Sigmoid(),
    nn.Conv2d(NC1,NC2,kernel_size=K2,stride=1,padding=0),
    nn.Sigmoid(),
    nn.Conv2d(NC2,NC3,kernel_size=K3,stride=1,padding=0),
    nn.Sigmoid(),
    )
loss_fn = nn.BCELoss()
