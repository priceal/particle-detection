# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal

Creates training and test data by either

1) loading in a data set and splitting into train/test sets OR
2) loading in two sets, one training and the other test

"""

trainingSetFile = '/home/allen/projects/training-data/trainingSets/C_NMPH4_span4_0123458_train.pkl'

createTestSet = False   # set to False if you want to load in test set
if createTestSet:
    testFraction = 0.20    
    testState = 1
else:
    testSetFile = '/home/allen/projects/training-data/trainingSets/C_NMPH4_span4_0123458_test.pkl'
    
coordinateDataPresent = False

##############################################################################
##############################################################################
# load in data set and determine parameters from shape
print('\nloading data set:', trainingSetFile)
with open(trainingSetFile, 'rb') as file:
    inputData = pickle.load(file)
if coordinateDataPresent:
    frames, classificationTemp, coords = inputData
else:
    frames, classificationTemp = inputData
numberSamples, yDim, xDim = frames.shape
print('... {} frames, {} x {}'.format(numberSamples,yDim,xDim))
if coordinateDataPresent:
    numberCoords = len(coords)
    print( '...', numberCoords, 'coordinates')
    classification = classificationTemp
else:
    classification = classificationTemp < 1
    
# load in training set and determine parameters from shape
if not createTestSet:
    print('\nloading data set:', testSetFile)
    with open(testSetFile, 'rb') as file:
        inputData = pickle.load(file)
        if coordinateDataPresent:
            framesTest, classificationTestTemp, coordsTest = inputData
        else:
            framesTest, classificationTestTemp = inputData
            # read parameters from test set shape
        numberTestSamples, yTestDim, xTestDim = framesTest.shape
        print('... {} frames, {} x {}'.format(numberTestSamples,yTestDim,xTestDim))
        if coordinateDataPresent:
            numberTestCoords = len(coordsTest)
            print( '...', numberTestCoords, 'coordinates')
            classificationTest = classificationTestTemp 
        else:
            classificationTest = classificationTestTemp < 1
            
# now determine test and training sets
if createTestSet:
    # split into train/test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        frames, classification, test_size=testFraction, random_state = testState)
else:
    X_train, Y_train = frames, classification
    X_test, Y_test = framesTest, classificationTest

print('\nTraining data: {} frames'.format(len(Y_train)))
print('Test data: {} frames'.format(len(Y_test)))

# pre-processing of frames
#scaled_frames = frames/frames.max()




