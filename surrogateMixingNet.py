import keras
import numpy as np

#docker run --gpus all -it -v "$PWD":/home --rm tensorflow/tensorflow:latest-gpu-py3 sh /home/surrogateMixingScript.sh

lsize=4096

batch_size=100
epochs=200

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

trainingData=np.load('SM_TrainingData.npy')
trainingAnswers=np.load('SM_TrainingAnswers.npy')
testData=np.load('SM_TestData.npy')
testAnswers=np.load('SM_TestAnswers.npy')


def run():
    print("Getting Model")
    NN=getModel()
    print("Model Generated")
    history = NN.fit(trainingData, trainingAnswers,batch_size=batch_size,
                     epochs=epochs,verbose=1)
    score = NN.evaluate(testData, testAnswers, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

#    NN.save('model_test.h5')


run()
