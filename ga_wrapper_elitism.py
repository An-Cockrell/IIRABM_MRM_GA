# NOTE THAT THIS CODE DOES NOT WORK WITH CURRENT VERIONS OF NUMPY, USE 1.14 instead
# This is due to some bug with the numpy/cytpes interface and they arent going to fix it soon
# https://github.com/numpy/numpy/pull/11277

import ctypes
import numpy as np
from mpi4py import MPI
from numpy.ctypeslib import ndpointer
from numpy import genfromtxt

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ctypes initialization
_IIRABM = ctypes.CDLL('/home/chase/iirabm_fullga/IIRABM_RuleGA.so')
#_IIRABM = ctypes.CDLL('/users/r/c/rcockrel/iirabm_fullga/IIRABM_RuleGA.so')
#_IIRABM = ctypes.CDLL('/global/cscratch1/sd/cockrell/IIRABM_RuleGA.so')

# (oxyHeal,infectSpread,numRecurInj,numInfectRepeat,inj_number,seed,numMatrixElements,internalParameterization)
_IIRABM.mainSimulation.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int)
_IIRABM.mainSimulation.restype = ndpointer(
    dtype=ctypes.c_float, shape=(20, 5280))

nonZeroChance=5
tournamentSize=2
numMatrixElements=429
mutationChance=0.010
numIters=500
numStochasticReplicates=10
array_type = ctypes.c_float*numMatrixElements
selectedTimePoints=np.array([89,119,149,179,209,239,359,479,719,1199,1919,3599,5279])
tnfMins=np.array([0,0,9.57,1.6,9.57,0,1.6,0,0,0,14.36,19.15,15.96])
tnfMaxs=np.array([47.87,43.09,49.47,47.87,55.85,43.09,60.64,57.45,97.34,121.28,84.57,49.47,76.60])
tMax=np.max(tnfMaxs)
tnfMaxs=tnfMaxs/np.max(tMax);
tnfMins=tnfMins/np.max(tMax);

data=np.array([[0.075,2,2,1,27],[0.075,6,2,1,27],[0.1,4,2,1,32],[0.1,2,2,2,32],[0.1,6,2,1,32]])

geneLow=-1.5
geneHigh=2
critfit=1370

eliteFraction=0.1
numElites=int(eliteFraction*size)

np.random.seed(10287)

def printRuleMat(ip):
    numRules=25
    numParams=17
    k=0
    RM=np.zeros([numRules,numParams])
    for i in range(numRules):
        for j in range(numParams):
            RM[i,j]=ip[k]
            k=k+1

def getFitness(data,numReplicates,internalParam):
    fitnessCompare=np.zeros(13,dtype=np.float32)
    for i in range(data.shape[0]):
        oxyHeal=data[i,0]
        infectSpread=int(data[i,1])
        numRecurInj=int(data[i,2])
        numInfectRepeat=int(data[i,3])
        injurySize=int(data[i,4])
        for seed in range(numStochasticReplicates):
            result=_IIRABM.mainSimulation(oxyHeal, infectSpread, numRecurInj,
                                  numInfectRepeat, injurySize, seed,
                                  numMatrixElements,
                                  array_type(*internalParam),rank)
            tnfResult=np.zeros(13,dtype=np.float32)
            for j in range(13):
                if(result[2,selectedTimePoints[j]-1]<0):
                    result[2,selectedTimePoints[j]-1]=0
                tnfResult[j]=result[2,selectedTimePoints[j]-1]
            fitnessCompare=np.vstack([fitnessCompare,tnfResult])
    np.delete(fitnessCompare,0,0)

    normalizer=0;
    for i in range(fitnessCompare.shape[0]):
        for j in range(13):
            if(fitnessCompare[i,j]>normalizer):
                normalizer=fitnessCompare[i,j]
    if(normalizer>0):
        for i in range(fitnessCompare.shape[0]):
            for j in range(13):
                    fitnessCompare[i,j]=fitnessCompare[i,j]/normalizer

#    print(rank,fitnessCompare)
    fitness=np.zeros(13,dtype=np.float32)
    for i in range(13):
        temp=fitnessCompare[:,i]
        fitMin=np.min(fitnessCompare[:,i])
        fitMax=np.max(fitnessCompare[:,i])
        term1=abs(fitMin-tnfMins[i])
        term2=abs(fitMax-tnfMaxs[i])
        if(fitMax==0):
            term2=100
        fitness[i]=term1+term2
    fitsum=np.sum(fitness)
    print(rank,fitness)
#    print(fitMin,fitMax,fitness)
    return fitsum

def getInitialIP():
    internalParameterization=np.load('baseParameterization.npy')
    numMatrixElements = internalParameterization.shape[0]
    np.asarray(internalParameterization, dtype=np.float32)

    internalParamArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    nonZeroMatEls=[]
    zeroMatEls=[]

    for i in range(numMatrixElements):
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
    internalParamArray[0,:]=np.load('baseParameterization.npy')
    return internalParamArray

def crossover(p1,p2):
    progeny1=np.zeros(numMatrixElements,dtype=np.float32)
    progeny2=np.zeros(numMatrixElements,dtype=np.float32)
    for i in range(numMatrixElements):
        beta=np.random.uniform(low=0,high=1)
        progeny1[i]=(1-beta)*p1[i]+beta*p2[i]
        progeny2[i]=(1-beta)*p2[i]+beta*p1[i]
    return progeny1,progeny2

def mutate(ip):
    tempRand=np.random.uniform(low=0,high=1000)
    if((tempRand/1000)<mutationChance):
        tempIndex=int(np.random.uniform(low=0,high=numMatrixElements))
        ip[tempIndex]=np.random.uniform(low=geneLow,high=geneHigh)
    return ip

def getNextParents(fits,iparray):
    breedables=[] #potential breeders
    bfits=[] #potential breeders associated fits
    for i in range(iparray.shape[0]):
        if(fits[i]<critfit):
            breedables.append(iparray[i,:])
            bfits.append(fits[i])
    breedables=np.asarray(breedables)
#    print("breedables shape=",breedables.shape)
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
        temp2=np.random.randint(low=0,high=bfits.shape[0])
        f1=bfits[temp1]
        f2=bfits[temp2]
        if(f1<f2):
            winner=temp1
        else:
            winner=temp2
        breeders.append(breedables[winner,:])
        breederFits.append(bfits[winner])
        bfits=np.delete(bfits,[temp1,temp2])
        breedables=np.delete(breedables,[temp1,temp2],axis=0)

    breeders=np.asarray(breeders)
    breederFits=np.asarray(breederFits)

    return breeders,breederFits

def getNextGeneration(breeders,breederFits):
    newIParray=np.zeros([size,numMatrixElements],dtype=np.float32)
    parentArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    parentFitArray=np.zeros(size,dtype=np.float32)
    print(breederFits)
    for i in range(0,size,2):
        if(breeders.shape[0]>2):
            temp1=np.random.randint(low=0,high=breeders.shape[0])
            p1=breeders[temp1,:]
#            breeders=np.delete(breeders,temp1,axis=0)
#            print("bs1=",breeders.shape,i)
            parentFitArray[i]=breederFits[temp1]
#            breederFits=np.delete(breederFits,temp1)
            temp2=np.random.randint(low=0,high=breeders.shape[0])
            p2=breeders[temp2,:]
#            breeders=np.delete(breeders,temp2,axis=0)
#            print("bs2=",breeders.shape,breederFits.shape)
            parentFitArray[i+1]=breederFits[temp2]
#            breederFits=np.delete(breederFits,temp2)
#            print("Shape=",breeders.shape)
        # elif(breeders.shape[0]<=2):
        #     print("Entering Else",breeders.shape[0])
        #     p1=breeders[0,:]
        #     p2=breeders[1,:]
        #     parentFitArray[i]=breederFits[0]
        #     parentFitArray[i+1]=breederFits[1]
        #     print("Else Statement Completed",i)
        c1,c2=crossover(p1,p2)
        c1=mutate(c1)
        c2=mutate(c2)
        newIParray[i,:]=c1
        newIParray[i+1,:]=c2
        parentArray[i,:]=p1
        parentArray[i+1,:]=p2
    return newIParray,parentArray,parentFitArray

def compareGenerations(fitArray,iparray,parentArray,parentFitArray):  #parentArrayNotDefined - fix this
    sortFits=np.argsort(fitArray)
    sortParents=np.argsort(parentFitArray)
    for i in range(numElites):
        compIndex=sortFits.shape[0]-(i+1)
        if(parentFitArray[sortParents[i]]<fitArray[compIndex]):
            fitArray[compIndex]=parentFitArray[sortParents[i]]
            iparray[compIndex,:]=parentArray[sortParents[i],:]
    return iparray,fitArray


def gaIter(recvbuf,iparray,parentArray,parentFitArray,genNumber):
    fits=[]
    for i in range(size):
        fits.append(recvbuf[i])
    fits=np.asarray(fits)
#    if(genNumber>0):
    iparray,fits=compareGenerations(fits,iparray,parentArray,parentFitArray)
    breeders,breederFits=getNextParents(fits,iparray)
    newIParray,parents,parentFits=getNextGeneration(breeders,breederFits)
    avgFit=np.mean(breederFits)
    return newIParray,avgFit,parents,parentFits


if(rank==0):
    iparray=getInitialIP();
    averages=[]
else:
    iparray=None

parentArray=np.zeros([size,numMatrixElements])
parentFitArray=np.zeros(size)
parentFitArray=parentFitArray+1000000

for i in range(numIters):
    mutationChance=0.010+0.01*i
    myIP=comm.scatter(iparray,root=0)
    myFitness=getFitness(data,numStochasticReplicates,myIP)
    print("FITNESS_%s="%i,rank,myFitness)
    recvbuf=None
    sendbuf=myFitness
    if rank==0:
        recvbuf=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, recvbuf, root=0)

    if(rank==0):
        iparray,avgFit,parentArray,parentFitArray=gaIter(recvbuf,iparray,parentArray,parentFitArray,i)
        averages.append(avgFit)
        iname=str('InternalParameterizationTNF_Gen%s.csv'%i)
        np.savetxt(iname,iparray,delimiter=',')
        print("Average Fitness=",avgFit)

if(rank==0):
    averages=np.asarray(averages)
    np.savetxt('FinalAverages.csv',averages,delimiter=',')
