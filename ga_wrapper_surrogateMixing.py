import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from itertools import product
import keras.backend as K
import os
from keras.models import load_model

nonZeroChance=5
tournamentSize=2
numBaseMatrixElements=429
numMatrixElements=432
baseGeneMutation=0.01
baseParamMutation=0.01
baseInjMutation=0.01

numDataPoints=selectedTimePoints.shape[0]

geneLow=-2
geneHigh=2

np.random.seed(10287)

def getInitialIP():
    internalParameterization=np.load('baseParameterization2.npy')
    numMatrixElements = internalParameterization.shape[0]
    np.asarray(internalParameterization, dtype=np.float32)

    internalParamArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    nonZeroMatEls=[]
    zeroMatEls=[]

    for i in range(numBaseMatrixElements):
        if(internalParameterization[i]!=0):
            nonZeroMatEls.append(i)
        else:
            zeroMatEls.append(i)
    nonZeroMatEls=np.asarray(nonZeroMatEls)
    zeroMatEls=np.asarray(zeroMatEls)
    for i in range(size):
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
    internalParamArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    for i in range(size):
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

def getNextParents(iparray):
    breedables=[] #potential breeders
    bfits=[] #potential breeders associated fits
    for i in range(iparray.shape[0]):
        if(fits[i]<critfit):
            breedables.append(iparray[i,:])
            bfits.append(fits[i])
#            print("bfits",fits[i])
    breedables=np.asarray(breedables)
    bfits=np.asarray(bfits)
    sortedFitIndexes=np.argsort(bfits)
    addBreedables=[]
    addfits=[]
    if(bfits.shape[0]<size): #adding fittest candidate duplicates to set of potential breeders
        diff=size-bfits.shape[0]
        k=0;
        for i in range(diff):
            addBreedables.append(iparray[sortedFitIndexes[k],:])
            addfits.append(bfits[k])
            k=k+1;
            if(k>sortedFitIndexes.shape[0]-1):
                k=0
        addfits=np.asarray(addfits)
        bfits=np.hstack([bfits,addfits])
        breedables=np.vstack([breedables,addBreedables])
#    np.random.shuffle(breedables)
    breeders=[]
    breederFits=[]
    for i in range(0,size,2):
        temp1=np.random.randint(low=0,high=bfits.shape[0])
        f1=bfits[temp1]
        ipf1=breedables[temp1,:]
        bfits=np.delete(bfits,temp1)
        breedables=np.delete(breedables,temp1,axis=0)
        temp2=np.random.randint(low=0,high=bfits.shape[0])
        f2=bfits[temp2]
        ipf2=breedables[temp2,:]
        bfits=np.delete(bfits,temp2)
        breedables=np.delete(breedables,temp2,axis=0)
#        print("F",f1,temp1,f2,temp2)
        if(f1<f2):
            winner=ipf1
            winnerFit=f1
        else:
            winner=ipf2
            winnerFit=f2
#        print("winner=",winner)
        breeders.append(winner)
        breederFits.append(winnerFit)

    breeders=np.asarray(breeders)
    breederFits=np.asarray(breederFits)

    return breeders

def getNextGeneration(breeders,gmc,imc,pmc):
    newIParray=np.zeros([size,numMatrixElements],dtype=np.float32)
    parentArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    parentFitArray=np.zeros(size,dtype=np.float32)
    print(breederFits)
    for i in range(0,size,2):
        if(breeders.shape[0]>=2):
            temp1=np.random.randint(low=0,high=breeders.shape[0])
            p1=breeders[temp1,:]
            parentFitArray[i]=breederFits[temp1]
            temp2=np.random.randint(low=0,high=breeders.shape[0])
            p2=breeders[temp2,:]
            parentFitArray[i+1]=breederFits[temp2]
        c1,c2=crossover(p1,p2)
        c1=mutate(c1,gmc,imc,pmc)
        c2=mutate(c2,gmc,imc,pmc)
        newIParray[i,:]=c1
        newIParray[i+1,:]=c2
        parentArray[i,:]=p1
        parentArray[i+1,:]=p2
    return newIParray


def gaIter(surrogateModel,iparray,genNumber,gmc,imc,pmc):
    breeders,breederFits=getNextParents(fits,iparray)
    newIParray,parents=getNextGeneration(breeders,breederFits,gmc,imc,pmc)
    return newIParray

injSize=int(sys.argv[1])
trainedModelFile=str('XXX')
surrogateModel = load_model(filename)
for i in range(numIters):
    iparray=getRandomIP();
    geneMutationChance=baseGeneMutation+0.001*i
    injMutationChance=baseInjMutation+0.001*i
    paramMutationChance=baseParamMutation+0.001*i
    iparray=gaIter(surrogateModel,iparray,i,geneMutationChance,injMutationChance,paramMutationChance)
    filename=str('InternalParameterization_NN_%s'%i)
