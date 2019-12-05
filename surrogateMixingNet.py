import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from itertools import product
import keras.backend as K
import os
from keras.models import load_model

#docker run --gpus all -it -v "$PWD":/home --rm tensorflow/tensorflow:latest-gpu-py3 sh /home/surrogateMixingScript.sh

lsize=4096

batch_size=10000
epochs=200
trainingProportion=0.01

def getModel():
    keras.backend.clear_session()
    model = Sequential()
    # Experiment with relu
    model.add(Dense(lsize, activation='relu', input_shape=(864,)))
    model.add(Dropout(0.2))
    model.add(Dense(lsize, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam'  ,metrics=['accuracy'])
    return model

def getData(z):
    numSamples=z.shape[0]
    mark=int(numSamples*trainingProportion)
    mark2=numSamples-mark
    if(mark%2!=0):
        mark=mark-1
        mark2=mark2-1
    trainingIPs=z[0:mark,0:432]
    trainingFits=z[0:mark,432]
    testIPs=z[mark:numSamples,0:432]
    testFits=z[mark:numSamples,432]

    trainingData=[]
    trainingAnswers=[]
    testData=[]
    testAnswers=[]
    for i in range(0,mark,2):
        t1=trainingIPs[i,:]
        t2=trainingIPs[i+1,:]
        dat=np.hstack((t1,t2))
        f1=trainingFits[i]
        f2=trainingFits[i+1]
        trainingData.append(dat)
        if(f1<=f2):
            trainingAnswers.append(0)
        else:
            trainingAnswers.append(1)

    for i in range(0,mark2,2):
        t1=testIPs[i,:]
        t2=testIPs[i+1,:]
        dat=np.hstack((t1,t2))
        f1=testFits[i]
        f2=testFits[i+1]
        testData.append(dat)
        if(f1<=f2):
            testAnswers.append(0)
        else:
            testAnswers.append(1)
    trainingData=np.asarray(trainingData,dtype=np.float32)
    trainingAnswers=np.asarray(trainingAnswers,dtype=np.float32)
    testData=np.asarray(testData,dtype=np.float32)
    testAnswers=np.asarray(testAnswers,dtype=np.float32)
    return trainingData,trainingAnswers,testData,testAnswers

def run(trainingData,trainingAnswers,testData,testAnswers):
    print("Getting Model")
    NN=getModel()
    print("Model Generated")
    history = NN.fit(trainingData, trainingAnswers,batch_size=batch_size,
                     epochs=epochs,verbose=1)
    score = NN.evaluate(testData, testAnswers, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    NN.save('model_test_%s_samples.h5'%trainingData.shape[0])

# trainingData=np.load('SM_TrainingData.npy')
# trainingAnswers=np.load('SM_TrainingAnswers.npy')
# testData=np.load('SM_TestData.npy')
# testAnswers=np.load('SM_TestAnswers.npy')
fullData=np.load('AllData.npy')

trainingData,trainingAnswers,testData,testAnswers=getData(fullData)
print(trainingData.shape,testData.shape)
#print(np.sum(testAnswers),testAnswers.shape[0]/2)
run(trainingData,trainingAnswers,testData,testAnswers)
