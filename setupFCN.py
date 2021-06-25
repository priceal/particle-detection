# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 08:29:38 2021

@author: priceal

sets up data sets for use in fully connected network and defines
the network. To change number of layers, you need to modify the
definition of modelFCN near end.

VARIABLES NEEDED: X_train, Y_train, X_test, Y_test
NAMES DEFINED: modelFCN, xtrain, ytrain, xtest, ytest, loss_fn    

"""

#define fully connected network structure
H1, H2, D_out = 16, 4, 1


##############################################################################

# prepare training data.  Include flattening and scaling of X
numberTrainSamples, ydim, xdim = X_train.shape
xtrain = torch.FloatTensor( 
                            X_train.reshape(numberTrainSamples,xdim*ydim) \
                            / X_train.max()
                           )
ytrain =  torch.FloatTensor(
                            Y_train[:,np.newaxis]
                            )
    
# prepare test data.  Include flattening and scaling of X
numberTestSamples, ydim, xdim = X_test.shape
xtest = torch.FloatTensor( 
                            X_test.reshape(numberTestSamples,xdim*ydim) \
                            / X_test.max()
                           )
ytest =  torch.FloatTensor(
                            Y_test[:,np.newaxis]
                            )

# now define the fully connected network and loss function for training
modelFCN = nn.Sequential(
    nn.Linear(xdim*ydim, H1),
    nn.Sigmoid(),
    nn.Linear(H1, H2),
    nn.Sigmoid(),
    nn.Linear(H2, D_out),
    nn.Sigmoid(),
    )
loss_fn = nn.BCELoss()

