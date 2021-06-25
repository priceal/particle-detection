# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal

sets up data sets for use in logistic regression.
also fits the logistic regression model. so separate trainging step
is not needed.

VARIABLES NEEDED: X_train, Y_train, X_test, Y_test
NAMES DEFINED: modelLR, xtrain, ytrain, xtest, ytest    
    
"""

# use to print out regression coeffs if you want
printCoefficients = True

##############################################################################
##############################################################################

# prepare training data.  Include flattening and scaling of X
numberTrainSamples, ydim, xdim = X_train.shape
xtrain = X_train.reshape(numberTrainSamples,xdim*ydim) / X_train.max()
ytrain =  Y_train
    
# prepare test data.  Include flattening and scaling of X
numberTestSamples, ydim, xdim = X_test.shape
xtest = X_test.reshape(numberTestSamples,xdim*ydim) / X_test.max()
ytest =  Y_test

# define regression model and train!
print('\nFitting {} frames {} x {}'.format(numberTrainSamples,xDim,yDim))
modelLR = linear_model.LogisticRegression()
modelLR.fit(xtrain,ytrain)

# output results
if printCoefficients:
    print('\nRegression coefficents:')
    print(modelLR.coef_)
    print(modelLR.intercept_)


