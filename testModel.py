# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:36:50 2021

@author: priceal
"""

# define the image number of the image to test
testImage = 0

# this should be (n-1)/2 where n is length of side of receptive field
buffer = 3
stride = 1

# name of model to test
testModel = modelFCN

# only change the following to override setting from setup script
testModelType = modelType        

###############################################################################
# load image and create scanned view of image array
print("loading and scaling image file {} ...".format(imageDF['path'][testImage] ))
testImageLoaded = pa.loadim( imageDF['path'][testImage] )
yFrame, xFrame = testImageLoaded.shape
print("image dimensions {} x {}".format(xFrame,yFrame))
scaledImage = testImageLoaded/testImageLoaded.max()
dim = 2*buffer+1
xscan = np.zeros( ((yFrame-2*buffer) * (xFrame-2*buffer),dim,dim) )
c = 0
print("creating scanned view of image array ...")
for j in range(0,yFrame-dim+1,stride):
    for i in range(0,xFrame-dim+1,stride):
        xscan[c] = scaledImage[j:j+dim,i:i+dim]
        c += 1
print(" ... {} frames created".format(len(xscan)))        

# now apply model correctly for different cases
print("Now using {} model to create feature map...",format(testModelType))
if testModelType == 'LR':
    x_valid = xscan.reshape(len(xscan),dim*dim)
    x_pred_valid = testModel.predict( x_valid )
    mapout = x_pred_valid.reshape((yFrame-2*buffer,xFrame-2*buffer))

if testModelType == 'FCN':
    x_valid = torch.FloatTensor( xscan.reshape(len(xscan),dim*dim) )
    x_pred_valid = testModel( x_valid ).detach().numpy()
    mapout = x_pred_valid.reshape((yFrame-2*buffer,xFrame-2*buffer))

if testModelType == 'CNN':
    x_valid = torch.FloatTensor( scaledImage[np.newaxis,np.newaxis,:,:] )
    mapout = testModel( x_valid ).detach().numpy()
  
# round, convert to int and pad
imageOut = np.round( np.squeeze( mapout ) ).astype(int)
mapFinal = cv.copyMakeBorder(imageOut,buffer,buffer,buffer,buffer,cv.BORDER_CONSTANT,0)

# create array of particle positions predicted
testY, testX = np.where( mapFinal == 1 )
particleCoords = np.array( list( zip(testX,testY)))
print(" ...{} particles labeled.".format(len(particleCoords)))

# plot original image, feature map, and overlay of original image w/ predicted
# particle positions overlaid
fig, ax = plt.subplots(1,2,sharex='row', sharey = 'row')
ax[0].imshow( testImageLoaded, cmap = 'gray', interpolation = 'nearest')
ax[1].imshow( mapFinal, cmap = 'gray', interpolation = 'nearest')
pa.showPeaks( testImage, particleCoords, imgDF = imageDF )
