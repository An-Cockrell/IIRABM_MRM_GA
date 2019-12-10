import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from itertools import product
import keras.backend as K
import os
from keras.models import load_model

#docker run --gpus all -it -v "$PWD":/home --rm tensorflow/tensorflow:latest-gpu-py3 sh /home/surrogateMixingScript.sh
#docker run --gpus all -it -v "$PWD":/home --rm tensorflow/tensorflow:latest-gpu-py3 sh /home/surrogateMixingScriptBatch.sh

nonZeroChance=5
tournamentSize=2
numBaseMatrixElements=429
numMatrixElements=432
baseGeneMutation=0.01
baseParamMutation=0.01
baseInjMutation=0.01

numIters=1000

geneLow=-2
geneHigh=2

populationSize=1024

np.random.seed(10287)

def getInitialIP():
    internalParameterization=np.load('baseParameterization2.npy')
    numMatrixElements = internalParameterization.shape[0]
    np.asarray(internalParameterization, dtype=np.float32)

    internalParamArray=np.zeros([populationSize,numMatrixElements],dtype=np.float32)
    nonZeroMatEls=[]
    zeroMatEls=[]

    for i in range(numBaseMatrixElements):
        if(internalParameterization[i]!=0):
            nonZeroMatEls.append(i)
        else:
            zeroMatEls.append(i)
    nonZeroMatEls=np.asarray(nonZeroMatEls)
    zeroMatEls=np.asarray(zeroMatEls)
    for i in range(populationSize):
        for j in range(nonZeroMatEls.shape[0]):
            temp=np.random.uniform(low=geneLow,high=geneHigh)
            internalParamArray[i,nonZeroMatEls[j]]=temp
        for j in range(zeroMatEls.shape[0]):
            chance=np.random.uniform(low=0,high=100)
            if(chance<=nonZeroChance):
                temp=np.random.uniform(low=geneLow,high=geneHigh)
                internalParamArray[i,zeroMatEls[j]]=temp
    internalParamArray[0,:]=np.load('baseParameterization2.npy')
    return internalParamArray

def getRandomIP():
    internalParamArray=np.zeros([populationSize,numMatrixElements],dtype=np.float32)
    for i in range(populationSize):
        for j in range(numBaseMatrixElements):
            temp=np.random.uniform(low=geneLow,high=geneHigh)
            internalParamArray[i,j]=temp
    return internalParamArray

def crossover(p1,p2):
    progeny1=np.zeros(numMatrixElements,dtype=np.float32)
    progeny2=np.zeros(numMatrixElements,dtype=np.float32)
    for i in range(numMatrixElements):
        beta=np.random.uniform(low=0,high=1)
        progeny1[i]=(1-beta)*p1[i]+beta*p2[i]
        progeny2[i]=(1-beta)*p2[i]+beta*p1[i]
    return progeny1,progeny2

def mutate(ip,gmc,imc,pmc):
    rand_g=np.random.uniform(low=0,high=1000)
    rand_i=np.random.uniform(low=0,high=1000)
    rand_p=np.random.uniform(low=0,high=1000)
    if((rand_g/1000)<gmc):
        tempIndex=int(np.random.uniform(low=0,high=numBaseMatrixElements))
        ip[tempIndex]=np.random.uniform(low=geneLow,high=geneHigh)
    if((rand_i/1000)<imc):
        tempIndex=int(np.random.randint(low=0,high=4))
        ip[numBaseMatrixElements-4+tempIndex]=np.random.uniform(low=0,high=2)

    return ip

def getNextParents(surrogateModel,iparray):
    np.random.shuffle(iparray)
    breeders=[]
    candidates=np.zeros([populationSize,numMatrixElements*2],dtype=np.float32)
    k=0
    for i in range(0,populationSize,2):
        candidate1=iparray[i,:]
        candidate2=iparray[i+1,:]
        candidates[k,:]=np.hstack([candidate1,candidate2])
    answer,probability=evaluateCandidates(surrogateModel,candidates)
    numParents=int(populationSize/2)
    parents=np.zeros([numParents,numMatrixElements])
    for i in range(numParents):
#        parents[i,:]=getNextHybridParent(candidates[i],answer[i],probability[i])
#        parents[i,:]=getNextProbabilisticParent(candidates[i],answer[i],probability[i])
        parents[i,:]=getNextDeterministicParent(candidates[i],answer[i])
    return parents

def getNextHybridParent(candidate,answer,probability):
    returnParent=(1.-probability)*candidate[0:numMatrixElements]+probability*candidate[numMatrixElements:2*numMatrixElements]
    return returnParent

def getNextProbabilisticParent(candidate,answer,probability):
    returnParent=(1.-probability)*candidate[0:numMatrixElements]+probability*candidate[numMatrixElements:2*numMatrixElements]
    return returnParent

def getNextDeterministicParent(candidate,answer):
    if(answer==0):
        returnParent=candidate[0:numMatrixElements]
    if(answer==1):
        returnParent=candidate[numMatrixElements:2*numMatrixElements]
    return returnParent

def evaluateCandidates(surrogateModel,candidates):
        #class 0 means the first candidate is fittest, class 1 is the opposite
        #predict probs give the prob of class 1
#    candidates=np.hstack((candidate1,candidate2))
    fits=surrogateModel.predict(candidates)
    fitProbs=surrogateModel.predict_proba(candidates)
    fits=np.rint(fits)
    return fits,fitProbs

def getNextGeneration(breeders,gmc,imc,pmc):
    np.random.shuffle(breeders)
    newIParray=np.zeros([populationSize,numMatrixElements],dtype=np.float32)
    for i in range(0,int(populationSize/2),2):
        p1=breeders[i,:]
        p2=breeders[i+1,:]
        c1,c2=crossover(p1,p2)
        c1=mutate(c1,gmc,imc,pmc)
        c2=mutate(c2,gmc,imc,pmc)
        newIParray[i,:]=c1
        newIParray[i+1,:]=c2
    return newIParray

def gaIter(surrogateModel,iparray,genNumber,gmc,imc,pmc):
    breeders=getNextParents(surrogateModel,iparray)
    newIParray=getNextGeneration(breeders,gmc,imc,pmc)
    return newIParray

trainedModelFile=str('model_test_132608_samples_Acc8568.h5')
surrogateModel = load_model(trainedModelFile)
for i in range(numIters):
    print("Iteration",i)
    iparray=getRandomIP();
    geneMutationChance=baseGeneMutation+0.001*i
    injMutationChance=baseInjMutation+0.001*i
    paramMutationChance=baseParamMutation+0.001*i
    iparray=gaIter(surrogateModel,iparray,i,geneMutationChance,injMutationChance,paramMutationChance)
    filename=str('InternalParameterization_NN_%s'%i)
    np.save(filename,iparray)
