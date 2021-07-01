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

###############################################################################
# load, scale and convert image to tensor
testImageLoaded = pa.loadim( imageDF['path'][testImage] )
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
