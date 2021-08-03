# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:36:50 2021

@author: priceal
"""

# define the image number of the image to test
testImage = 300

# this should be (n-1)/2 where n is length of side of receptive field of CNN
border = 3

# name of CNN model to test
testModel = modelCNN

stride = 1

###############################################################################
# load image and create scanned view of image array
testImageLoaded = pa.loadim( imageDF['path'][testImage] )
yFrame, xFrame = testImageLoaded.shape
scaledImage = testImageLoaded/testImageLoaded.max()
dim = 2*buffer+1
xscan = np.zeros( ((yFrame-2*buffer) * (xFrame-2*buffer),dim,dim) )
c = 0
for j in range(0,yFrame-dim+1,stride):
    for i in range(0,xFrame-dim+1,stride):
        xscan[c] = scaledImage[j:j+dim,i:i+dim]
        c += 1
       
# now apply model correctly for different cases
if isinstance(testModel,linear_model.LogisticRegression):
    x_valid = xscan.reshape(len(xscan),dim*dim)
    x_pred_valid = testModel.predict( x_valid )
    mapout = x_pred_valid.reshape((yFrame-2*buffer,xFrame-2*buffer))
    
if isinstance(model,torch.nn.modules.container.Sequential):
    x_valid = xscan.reshape(len(xscan),dim*dim)
    testTensor = torch.FloatTensor( testImageLoaded[np.newaxis,np.newaxis,:,:] )
    x_pred_valid = testModel( x_valid ).detach().numpy()
    mapout = x_pred_eval.detach().numpy()

if isinstance(model,torch.nn.modules.container.Sequential):
    x_valid = xscan.reshape(len(xscan),dim*dim)
    testTensor = torch.FloatTensor( testImageLoaded[np.newaxis,np.newaxis,:,:] )
    x_pred_valid = testModel( x_valid ).detach().numpy()
    mapout = x_pred_eval.detach().numpy()

testTensor = torch.FloatTensor( 
                            testImageLoaded[np.newaxis,np.newaxis,:,:] \
                            / testImageLoaded.max()
                           )
    
    
    
    
    
    
    
    
# calcualte feature map and convert to image array and add border
mapout = testModel(testTensor)
imageOut = np.round( np.squeeze( mapout.detach().numpy() ) )
mapFinal = cv.copyMakeBorder(imageOut,border,border,border,border,cv.BORDER_CONSTANT,0)

# create array of particle positions predicted
testY, testX = np.where( mapFinal == 1 )
particleCoords = np.array( list( zip(testX,testY)))

# plot original image, feature map, and overlay of original image w/ predicted
# particle positions overlaid
fig, ax = plt.subplots(1,2,sharex='row', sharey = 'row')
ax[0].imshow( testImageLoaded, cmap = 'gray', interpolation = 'nearest')
ax[1].imshow( mapFinal, cmap = 'gray', interpolation = 'nearest')
pa.showPeaks( testImage, particleCoords, imgDF = imageDF )
