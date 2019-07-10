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
numIters=100
numStochasticReplicates=10
array_type = ctypes.c_float*numMatrixElements
selectedTimePoints=np.array([89,119,149,179,209,239,359,479,719,1199,1919,3599,5279])
tnfMins=np.array([0,0,9.57,1.6,9.57,0,1.6,0,0,0,14.36,19.15,15.96])
tnfMaxs=np.array([47.87,43.09,49.47,47.87,55.85,43.09,60.64,57.45,97.34,121.28,84.57,49.47,76.60])

tnfMaxs=tnfMaxs/np.max(tnfMaxs);
tnfMins=tnfMins/np.max(tnfMaxs);

data=np.array([[0.075,2,2,1,27],[0.075,6,2,1,27],[0.1,4,2,1,32],[0.1,2,2,2,32],[0.1,6,2,1,32]])

geneLow=-1.5
geneHigh=2
critfit=130000


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
        fitnessCompare=fitnessCompare/normalizer

#    print("FC=",fitnessCompare)

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
        progeny1[i]=(1-beta)*iparray[p1,i]+beta*iparray[p2,i]
        progeny2[i]=(1-beta)*iparray[p2,i]+beta*iparray[p1,i]
    return progeny1,progeny2

def mutate(ip):
    tempRand=np.random.uniform(low=0,high=1000)
    if((tempRand/1000)<mutationChance):
        tempIndex=int(np.random.uniform(low=0,high=numMatrixElements))
        ip[tempIndex]=np.random.uniform(low=geneLow,high=geneHigh)
    return ip

def getNextParents(fits,indexes):
    breedables=[]
    bfits=[]
    for i in range(indexes.shape[0]):
        if(fits[i]<critfit):
            breedables.append(i)
            bfits.append(fits[i])
    breedables=np.asarray(breedables)
    bfits=np.asarray(bfits)
    sortedFitIndexes=np.argsort(bfits)
    addfits=[]
    # print("BFITS=",bfits.shape,bfits)
    if(bfits.shape[0]<size):
        diff=size-bfits.shape[0]
        k=0;
        for i in range(diff):
            addfits.append(sortedFitIndexes[k])
            k=k+1;
            if(k>sortedFitIndexes.shape[0]-1):
                k=0
        addfits=np.asarray(addfits)
        breedables=np.hstack([breedables,addfits])
    # print("Breedables=",breedables)
    np.random.shuffle(breedables)
    # print("BSIZE=",breedables.shape)
    breeders=[]
    for i in range(0,size,2):
        temp1=fits[breedables[i]]
        temp2=fits[breedables[i]]
        if(temp1<temp2):
            winner=breedables[i]
        else:
            winner=breedables[i+1]
        breeders.append(winner)
    breeders=np.asarray(breeders)
    np.random.shuffle(breeders)

    return breeders,bfits

def getNextGeneration(breeders):
    newIParray=np.zeros([size,numMatrixElements],dtype=np.float32)
    for i in range(0,size,2):
        temp1=np.random.randint(low=0,high=breeders.shape[0])
        temp2=np.random.randint(low=0,high=breeders.shape[0])
        p1=breeders[temp1]
        p2=breeders[temp2]
        c1,c2=crossover(p1,p2)
        c1=mutate(c1)
        c2=mutate(c2)
        newIParray[i,:]=c1
        newIParray[i+1,:]=c2
    return newIParray


def gaIter(recvbuf):
    fits=[]
    indexes=[]
    for i in range(size):
        fits.append(recvbuf[i])
        indexes.append(i)
    indexes=np.asarray(indexes)
    fits=np.asarray(fits)
    breeders,bfits=getNextParents(fits,indexes)
    newIParray=getNextGeneration(breeders)
    avgFit=np.mean(bfits)
    return newIParray,avgFit


if(rank==0):
    iparray=getInitialIP();
    averages=[]
else:
    iparray=None

for i in range(numIters):
    myIP=comm.scatter(iparray,root=0)
    myFitness=getFitness(data,numStochasticReplicates,myIP)
    print("FITNESS_%s="%i,rank,myFitness)

    recvbuf=None
    sendbuf=myFitness
    if rank==0:
        recvbuf=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, recvbuf, root=0)

    if(rank==0):
#        print("RBUF=",recvbuf)
        iparray,avgFit=gaIter(recvbuf)
        averages.append(avgFit)
        iname=str('InternalParameterization_Gen%s.csv'%i)
        np.savetxt(iname,iparray,delimiter=',')
        print("Average Fitness=",avgFit)

if(rank==0):
    averages=np.asarray(averages)
    np.savetxt('FinalAverages.csv',averages,delimiter=',')
