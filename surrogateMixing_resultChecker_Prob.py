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
_IIRABM = ctypes.CDLL('/home/chase/IIRABM_MRM_GA/IIRABM_RuleGA.so')
#_IIRABM = ctypes.CDLL('/users/r/c/rcockrel/iirabm_fullga/IIRABM_RuleGA.so')
#_IIRABM = ctypes.CDLL('/global/cscratch1/sd/cockrell/IIRABM_RuleGA.so')

# (oxyHeal,infectSpread,numRecurInj,numInfectRepeat,inj_number,seed,numMatrixElements,internalParameterization)
_IIRABM.mainSimulation.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int)
_IIRABM.mainSimulation.restype = ndpointer(
    dtype=ctypes.c_float, shape=(20, 5280))

numMatrixElements=432
numStochasticReplicates=40
array_type = ctypes.c_float*numMatrixElements
selectedTimePoints=np.array([29,59,89,119,149,179,209,239,359,479,719,1199,1919,3599,5279])
numDataPoints=selectedTimePoints.shape[0]

tnfMins=np.array([22.34,20.74,0,0,9.57,1.6,9.57,0,1.6,0,0,0,14.36,19.15,15.96])
tnfMaxs=np.array([135.64,79,47.87,43.09,49.47,47.87,55.85,43.09,60.64,57.45,97.34,121.28,84.57,49.47,76.60])
tMax=np.max(tnfMaxs)
tnfMaxs=tnfMaxs/tMax
tnfMins=tnfMins/tMax

il4Mins=np.array([14,12,0,1,0,0,0,0,0,0,0,0,2,13,0])
il4Maxs=np.array([213.53,93.23,58.15,52.13,45.11,36.09,50.13,49.12,49.12,59.15,99.25,151.38,109.27,54.14,69.17])
il4Max=np.max(il4Maxs)
il4Mins=il4Mins/il4Max
il4Maxs=il4Maxs/il4Max

gcsfMins=np.array([47.75,15.92,0,0,0,0,0,19.89,47.75,23.87,3.98,0,0,3.98,0])
gcsfMaxs=np.array([3846,11071,1102,823,831,107,139,159,640,405,508,1090,342,413,441])
gcsfMax=np.max(gcsfMaxs)
gcsfMins=gcsfMins/gcsfMax
gcsfMaxs=gcsfMaxs/gcsfMax

il10Mins=np.array([34.48,11.94,0,2.65,7.96,3.98,23.87,0,11.94,1.33,2.65,1.33,0,1.33,3.98])
il10Maxs=np.array([228,199,454,198,228,243,284,118,842,122,184,3842,49,15,14])
il10Max=np.max(il10Maxs)
il10Mins=il10Mins/il10Max
il10Maxs=il10Maxs/il10Max

ifngMins=np.array([52,0,4.76,0,4.76,0,9.52,0,4.76,9.52,0,4.76,9.52,0,4.76])
ifngMaxs=np.array([11071,2857,974,850,902,759,1136,902,1017,1218,1700,2142,2142,754,587])
ifngMax=np.max(ifngMaxs)
ifngMins=ifngMins/ifngMax
ifngMaxs=ifngMaxs/ifngMax

masterTNFnorm=334.0
masterIL4norm=2162.0
masterIL10norm=28225.0
masterGCSFnorm=8238.0
masterIFNnorm=37726.0

np.random.seed(10287)

def getFitnessResult2(result,index):
    selectResult=np.zeros(numDataPoints,dtype=np.float32)
    for j in range(numDataPoints):
        selectResult[j]=result[index,selectedTimePoints[j]-1]
    return selectResult

def compareFitness(input,mins,maxs):
    fitness=np.zeros(numDataPoints,dtype=np.float32)
    storeMin=[]
    storeMax=[]
    for i in range(numDataPoints):
        temp=input[:,i]
        filteredTemp=[]
        k=0
        for j in range(numStochasticReplicates):
            if(input[j,i]>=0):
                k=k+1
                filteredTemp.append(input[j,i])
        if(k>0):
            filteredTemp=np.asarray(filteredTemp)
            fitMin=np.min(filteredTemp)
            fitMax=np.max(filteredTemp)
            storeMin.append(fitMin)
            storeMax.append(fitMax)
            term1=abs(fitMin-mins[i])
            term2=abs(fitMax-maxs[i])
            if(fitMax==0):
                term2=100
            fitness[i]=term1+term2
        else:
            storeMin.append(-1)
            storeMax.append(-1)
            fitness[i]=200
    fitsum=np.sum(fitness)
    storeMin=np.asarray(storeMin)
    storeMax=np.asarray(storeMax)
    return fitsum,storeMin,storeMax

def compareResult(tnfR,il4R,il10R,gcsfR,ifngR):
    numViable=0
#    print(tnfR.shape)
    for i in range(numStochasticReplicates):
        flag=0
        for j in range(numDataPoints):
            if((tnfR[i,j]>tnfMins[j] and tnfR[i,j]<tnfMaxs[j]) and
                (il4R[i,j]>il4Mins[j] and il4R[i,j]<il4Maxs[j]) and
                (il10R[i,j]>il10Mins[j] and il10R[i,j]<il10Maxs[j]) and
                (gcsfR[i,j]>gcsfMins[j] and gcsfR[i,j]<gcsfMaxs[j]) and
                (ifngR[i,j]>ifngMins[j] and ifngR[i,j]<ifngMaxs[j])):
                    flag=flag+1
        if(flag==numDataPoints):
            numViable=numViable+1
    return numViable

def getFitnessResult(result,index):
    selectResult=np.zeros(numDataPoints,dtype=np.float32)
    for j in range(numDataPoints):
        if(result[index,selectedTimePoints[j]-1]<0):
            result[index,selectedTimePoints[j]-1]=0
        selectResult[j]=result[index,selectedTimePoints[j]-1]
    return selectResult

def normalizeResult(input):
    normalizer=0;
    for i in range(input.shape[0]):
        for j in range(numDataPoints):
            if(input[i,j]>normalizer):
                normalizer=input[i,j]
    if(normalizer>0):
        input=input/normalizer
    return input

def getFitness(numReplicates,internalParam):
    fitnessCompare=np.zeros(numDataPoints,dtype=np.float32)
    tnfResult=np.zeros(numDataPoints,dtype=np.float32)
    il4Result=np.zeros(numDataPoints,dtype=np.float32)
    il10Result=np.zeros(numDataPoints,dtype=np.float32)
    gcsfResult=np.zeros(numDataPoints,dtype=np.float32)
    ifngResult=np.zeros(numDataPoints,dtype=np.float32)

    oxyHeal=0.2
    infectSpread=4
    numRecurInj=2
    numInfectRepeat=2
    injurySize=27
    for seed in range(numStochasticReplicates):
        result=_IIRABM.mainSimulation(oxyHeal, infectSpread, numRecurInj,
                                  numInfectRepeat, injurySize, seed,
                                  numMatrixElements,
                                  array_type(*internalParam),rank)
        tnfResult=np.vstack([tnfResult,getFitnessResult2(result,2)])
        il4Result=np.vstack([il4Result,getFitnessResult2(result,15)])
        il10Result=np.vstack([il10Result,getFitnessResult2(result,4)])
        gcsfResult=np.vstack([gcsfResult,getFitnessResult2(result,5)])
        ifngResult=np.vstack([ifngResult,getFitnessResult2(result,12)])

    tnfResult=np.delete(tnfResult,0,0)
    il4Result=np.delete(il4Result,0,0)
    il10Result=np.delete(il10Result,0,0)
    gcsfResult=np.delete(gcsfResult,0,0)
    ifngResult=np.delete(ifngResult,0,0)

#    print(rank,tnfResult)

    # tnfResult=normalizeResult(tnfResult)
    # il4Result=normalizeResult(il4Result)
    # il10Result=normalizeResult(il10Result)
    # gcsfResult=normalizeResult(gcsfResult)
    # ifngResult=normalizeResult(ifngResult)

    tnfResult=tnfResult/masterTNFnorm
    il4Result=il4Result/masterIL4norm
    il10Result=il10Result/masterIL10norm
    gcsfResult=gcsfResult/masterGCSFnorm
    ifngResult=ifngResult/masterIFNnorm

    numViable=compareResult(tnfResult,il4Result,il10Result,gcsfResult,ifngResult)

    fit1,mn1,mx1=compareFitness(tnfResult,tnfMins,tnfMaxs)
    fit2,mn2,mx2=compareFitness(il4Result,il4Mins,il4Maxs)
    fit3,mn3,mx3=compareFitness(il10Result,il10Mins,il10Maxs)
    fit4,mn4,mx4=compareFitness(gcsfResult,gcsfMins,gcsfMaxs)
    fit5,mn5,mx5=compareFitness(ifngResult,ifngMins,ifngMaxs)
    fitsum=fit1+fit2+fit3+fit4+fit5

    retMn=np.vstack((mn1,mn2,mn3,mn4,mn5))
    retMx=np.vstack((mx1,mx2,mx3,mx4,mx5))
    return fitsum

numGens=1000
genIncrement=10
numTests=10

for i in range(0,numGens,genIncrement):
    filename=str('SM_TestProb/InternalParameterization_NN_%s.npy'%i)
    iparray=np.load(filename)
    myIP=iparray[rank,:]
    myFitness=getFitness(numStochasticReplicates,myIP)
    recvbuf=None
    sendbuf=myFitness
    if rank==0:
        recvbuf=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, recvbuf, root=0)
    if(rank==0):
        fname=str('SM_Fitness_Prob_%s.npy'%i)
        np.savetxt(fname,recvbuf,delimiter=',')
