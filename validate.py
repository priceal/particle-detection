# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:36:50 2021

@author: priceal
"""

# define the image number of the image to test
testImage = 8
pickedCoordsFile = '/home/allen/projects/training-data/data/particleCoordinates/reviewedXY_08.npy'

# this should be (n-1)/2 where n is length of side of receptive field
buffer = 3
stride = 1   # normally 1, used to make scanned view of image if needed

# peak finding parameters for final peak filtering of feature map (backend)
minDistance = 10       # min distance between peaks
relThreshold = 0.5    # min relative threshold allowed for peaks

# name of model to test
testModel = modelLR

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
    x_pred_valid = testModel.predict_proba( x_valid )[:,1] #prob for hit
    mapout = x_pred_valid.reshape((yFrame-2*buffer,xFrame-2*buffer))

if testModelType == 'FCN':
    x_valid = torch.FloatTensor( xscan.reshape(len(xscan),dim*dim) )
    x_pred_valid = testModel( x_valid ).detach().numpy()
    mapout = x_pred_valid.reshape((yFrame-2*buffer,xFrame-2*buffer))

if testModelType == 'CNN':
    x_valid = torch.FloatTensor( scaledImage[np.newaxis,np.newaxis,:,:] )
    mapout = testModel( x_valid ).detach().numpy()
  
# pad to produce final feature map (will consist of values 0 to 1)
mapFinal = cv.copyMakeBorder( \
           np.squeeze( mapout ) ,buffer,buffer,buffer,buffer,cv.BORDER_CONSTANT,0)

# create array of particle positions predicted using local max finder backend
particleCoords = peak_local_max( mapFinal, \
                                min_distance = minDistance, \
                                threshold_rel = relThreshold \
                                )
# now create the image w/ a single 1 per particle
particleImage = np.zeros((yFrame,xFrame),dtype=np.uint8)
particleImage[tuple(particleCoords.T)] = 1
print(" ...{} particles labeled.".format(len(particleCoords)))

# create image of hand picked particles (the ground truth)
pickedImage = np.zeros((yFrame,xFrame),dtype=np.uint8)
pickedImage[pickedCoords[:,1],pickedCoords[:,0]] = 1

# calculate confusion matrix using ground truth and particle image
cf = confusion_matrix( pickedImage.flatten(), particleImage.flatten() )
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

# plot original image amd ground truth image, both with predicted particle
# locations circled
pa.showPeaks( testImage, np.fliplr(particleCoords), imgDF = imageDF )
pa.showPeaks( pickedImage, np.fliplr(particleCoords), imgDF = imageDF )





