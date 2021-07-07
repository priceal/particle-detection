# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:26:38 2021

@author: priceal

This will train either the FCN or CNN.

NAMES NEEDED: xtrain, ytrain, xtest, ytest, loss_fn    

"""

# choose the model to refine
model = modelFCN

# define number of reporting macrocycles and epochs per reporting cycle
epochs = 100
reporting = 20

# optimization parameters
learning_rate = 0.01

##############################################################################
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# evaluate and print out initial state
print('\n{:5}     {:10}     {:10}'.format('epoch','training','test'))
y_pred = model(xtrain)
loss = loss_fn( y_pred, ytrain )
y_predTest = model(xtest)
lossTest = loss_fn( y_predTest, ytest )
print('{:5}     {:10.8f}     {:10.8f}'.format(0,loss.item(),lossTest.item()))

for tt in range(1,epochs+1):
    y_pred = model(xtrain)
    loss = loss_fn(y_pred,ytrain)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if tt % reporting == 0:
        y_predTest = model(xtest)
        lossTest = loss_fn( y_predTest, ytest )
        print('{:5}     {:10.8f}     {:10.8f}'.format(tt,loss.item(),lossTest.item()))

# print out final state    
print("\nFINAL STATE")
y_pred = model(xtrain)
loss = loss_fn( y_pred, ytrain )
y_predTest = model(xtest)
lossTest = loss_fn( y_predTest, ytest )
print('{:5}     {:10.8f}     {:10.8f}'.format(tt,loss.item(),lossTest.item()))
