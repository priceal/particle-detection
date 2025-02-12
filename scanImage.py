# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:36:50 2021

@author: priceal
"""



scanImage = image
xBuffer, yBuffer = 3, 3
scanModel = modelLR
stride = 1

###############################################################################
xDim, yDim = 2*xBuffer+1, 2*yBuffer+1
yFrame, xFrame = scanImage.shape
scaledImage = scanImage/scanImage.max()
c = 0
xscan = np.zeros( ((yFrame-2*yBuffer) * (xFrame-2*xBuffer),yDim,xDim) )
for j in range(0,yFrame-yDim+1,stride):
#    if j % 200 == 0:
#        print("row",j)
    for i in range(0,xFrame-xDim+1,stride):
#        if j % 200 == 0 and i % 200 == 0:
#            print("    column",i)
        xscan[c] = scaledImage[j:j+yDim,i:i+xDim]
        c += 1
       
if isinstance(scanModel,linear_model.LogisticRegression):
    x_valid = xscan.reshape(len(xscan),yDim*xDim)
    x_pred_valid = scanModel.predict( x_valid )
#    x_pred_eval_input = np.round( x_pred_eval )
    mapout = x_pred_valid.reshape((yFrame-2*yBuffer,xFrame-2*xBuffer))
    
if isinstance(model,torch.nn.modules.container.Sequential):
    x_pred_eval = model( x_eval )
    y_eval_input = y_eval.detach().numpy()
    x_pred_eval_input = np.round( x_pred_eval.detach().numpy() ) 

'''
mapout = np.zeros((yFrame-2*yBuffer,xFrame-2*xBuffer))
for j in range(0,yFrame-yDim+1,stride):
    if j % 200 == 0:
        print("row",j)
    for i in range(0,xFrame-xDim+1,stride):
        if j % 200 == 0 and i % 200 == 0:
            print("    column",i)
        Xf = scaledImage[j:j+xDim,i:i+yDim].copy().flatten()[np.newaxis,:]
        Xt = torch.FloatTensor(Xf)
        mapout[j,i] = model(Xt).detach().numpy()


'''
