# NOTE THAT THIS CODE DOES NOT WORK WITH CURRENT VERIONS OF NUMPY, USE 1.14 instead
# This is due to some bug with the numpy/cytpes interface and they arent going to fix it soon
# https://github.com/numpy/numpy/pull/11277

import ctypes
import numpy as np
from mpi4py import MPI
from numpy.ctypeslib import ndpointer
from numpy import genfromtxt
import sys

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank,size)

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
numBaseMatrixElements=429
numMatrixElements=432
baseGeneMutation=0.01
baseParamMutation=0.01
baseInjMutation=0.01
numIters=500
numStochasticReplicates=50
array_type = ctypes.c_float*numMatrixElements
selectedTimePoints=np.array([29,59,89,119,149,179,209,239,359,479,719,1199,1919,3599,5279])
numDataPoints=selectedTimePoints.shape[0]



geneLow=-1.5
geneHigh=2
critfit=7500

eliteFraction=0.1
numElites=int(eliteFraction*size)
#print("NE=",numElites)

np.random.seed(10287)

def getFitness(numReplicates,internalParam,injSize):
    max1=0;
    max2=0;
    max3=0;
    max4=0;
    max5=0;

    oxyHeal=0.2
    infectSpread=4
    numRecurInj=2
    numInfectRepeat=2
    injurySize=injSize
    for seed in range(numStochasticReplicates):
        result=_IIRABM.mainSimulation(oxyHeal, infectSpread, numRecurInj,
                                  numInfectRepeat, injurySize, seed,
                                  numMatrixElements,
                                  array_type(*internalParam),rank)
        tempTNFMax=np.max(result[2,29:5280])
        tempIL4Max=np.max(result[15,29:5280])
        tempIL10Max=np.max(result[4,29:5280])
        tempGCSFMax=np.max(result[5,29:5280])
        tempIFNGMax=np.max(result[12,29:5280])

        if(tempTNFMax>max1):
            max1=tempTNFMax
        if(tempIL4Max>max2):
            max2=tempIL4Max
        if(tempIL10Max>max3):
            max3=tempIL10Max
        if(tempGCSFMax>max4):
            max4=tempGCSFMax
        if(tempIFNGMax>max5):
            max5=tempIFNGMax


    mArray=[max1,max2,max3,max4,max5]

    return mArray

def getRandomIP():
    internalParamArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    for i in range(size):
        for j in range(429):
            temp=np.random.uniform(low=geneLow,high=geneHigh)
            internalParamArray[i,j]=temp
    return internalParamArray


injSize=int(sys.argv[1])
iparray=np.loadtxt('PaperData/InternalParameterization_H_RandCh_IS25_Gen249.csv',delimiter=',')
for i in range(numIters):
    if(rank==0):
        print(i)
    index=rank+i*size;
    myIP=iparray[index,:]
    maxArray=getFitness(numStochasticReplicates,myIP,injSize)

    maxArray=np.asarray(maxArray)

    rb1=None
    sendbuf=maxArray[0]
    if rank==0:
        rb1=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, rb1, root=0)
    rb2=None
    sendbuf=maxArray[1]
    if rank==0:
        rb2=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, rb2, root=0)

    rb3=None
    sendbuf=maxArray[2]
    if rank==0:
        rb3=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, rb3, root=0)

    rb4=None
    sendbuf=maxArray[3]
    if rank==0:
        rb4=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, rb4, root=0)

    rb5=None
    sendbuf=maxArray[4]
    if rank==0:
        rb5=np.empty([size], dtype=np.float32)
    comm.Gather(sendbuf, rb5, root=0)
    if(rank==0):
        np.savetxt('TNFMaxArray_%s.csv'%i,rb1,delimiter=',')
        np.savetxt('IL4MaxArray_%s.csv'%i,rb2,delimiter=',')
        np.savetxt('IL10MaxArray_%s.csv'%i,rb3,delimiter=',')
        np.savetxt('GCSFMaxArray_%s.csv'%i,rb4,delimiter=',')
        np.savetxt('IFNMaxArray_%s.csv'%i,rb5,delimiter=',')
