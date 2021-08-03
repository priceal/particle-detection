# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:36:50 2021

@author: priceal
"""

# define the image number of the image to test
testImage = 0
pickedCoordsFile = '/home/allen/projects/training-data/data/particleCoordinates/reviewedXY_00.npy'

# this should be (n-1)/2 where n is length of side of receptive field
buffer = 3
stride = 1

# name of model to test
testModel = modelCNN

# only change the following to override setting from setup script
testModelType = modelType        

###############################################################################
# load the picked coordinates
pickedCoords = np.load(pickedCoordsFile)

# load image and create scanned view of image array if needed
print("loading and scaling image file {} ...".format(imageDF['path'][testImage] ))
testImageLoaded = pa.loadim( imageDF['path'][testImage] )
yFrame, xFrame = testImageLoaded.shape
print("image dimensions {} x {}".format(xFrame,yFrame))
scaledImage = testImageLoaded/testImageLoaded.max()
dim = 2*buffer+1
xscan = np.zeros( ((yFrame-2*buffer) * (xFrame-2*buffer),dim,dim) )
c = 0
if testModelType != 'CNN':
    print("creating scanned view of image array ...")
    for j in range(0,yFrame-dim+1,stride):
        for i in range(0,xFrame-dim+1,stride):
            xscan[c] = scaledImage[j:j+dim,i:i+dim]
            c += 1
    print(" ... {} frames created".format(len(xscan)))        

# now apply model correctly for different cases
print("Now using {} model to create feature map...".format(testModelType))
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
  
# pad to produce final feature map, then round and convert to int
mapFinal = cv.copyMakeBorder( \
           np.squeeze( mapout ) ,buffer,buffer,buffer,buffer,cv.BORDER_CONSTANT,0)
imageOut = np.round( mapFinal ).astype(int) # binary classification

# create array of particle positions predicted
testY, testX = np.where( imageOut == 1 )
particleCoords = np.array( list( zip(testX,testY)))
print(" ...{} particles labeled.".format(len(particleCoords)))

# create image of hand picked particles
pickedImage = np.zeros((yFrame,xFrame),dtype=int)
for element in pickedCoords:
    pickedImage[ element[1], element[0] ] = 1

# calculate confusion matrix using prediction and actual locations
cf = confusion_matrix( pickedImage.flatten(), imageOut.flatten() )
truePositives = cf[1,1]
falsePositives = cf[0,1]
trueNegatives = cf[0,0]
falseNegatives = cf[1,0]
negatives = trueNegatives + falseNegatives
positives = truePositives + falsePositives
actualHits = truePositives + falseNegatives
actualMisses = trueNegatives + falsePositives
print('')
print('test set contains {:d} TRUE and {:d} FALSE'.format(actualHits,actualMisses))
print('predictions contain {} POSITIVE and {} NEGATIVE'.format(positives,negatives))
print('')
print('recall    {:>2.1f}%'.format(100*truePositives/actualHits) )
print('precision {:>2.1f}%'.format(100*truePositives/(truePositives+falsePositives)) )
frmt = '{:<10} {:>10} {:>10} {:>10}'
print('')
print('           **CONFUSION MATRIX**')
print(frmt.format(' ','FALSE','TRUE','TOTAL'))
print(frmt.format('NEGATIVE',trueNegatives,falseNegatives,negatives))
print(frmt.format('POSITIVE',falsePositives,truePositives,positives))
print(frmt.format('TOTAL',actualMisses,actualHits,negatives+positives))

# create a new image, which shows model predictions in red and actual in white
# using "flag" colormap, black = 12, white = 4, and red = 0
evalImage = 12*np.ones((yFrame,xFrame),dtype=int)
for element in particleCoords:
    evalImage[ element[1], element[0] ] = 0    
for element in pickedCoords:
    evalImage[ element[1], element[0] ] = 4
    
# display new image next to original, and open another window with original
# with predicted peaks circled
fig, ax = plt.subplots(1,2,sharex='row', sharey = 'row')
ax[0].imshow( testImageLoaded, cmap = 'gray', interpolation = 'nearest')
ax[1].imshow( evalImage, cmap = 'flag', interpolation = 'nearest')
pa.showPeaks( testImage, particleCoords, imgDF = imageDF )


# this will be code that applies openCV simpleBlobDetector





