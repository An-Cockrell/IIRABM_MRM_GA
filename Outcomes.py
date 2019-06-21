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
# (oxyHeal,infectSpread,numRecurInj,numInfectRepeat,inj_number,seed,numMatrixElements,internalParameterization)
_IIRABM.mainSimulation.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.POINTER(ctypes.c_float))
# _IIRABM.mainSimulation.restype = ndpointer(
#     dtype=ctypes.c_float, shape=(20, 100))

internalParameterization=np.load('baseParameterization.npy')
# injurySize = np.array([25,27,30,32,35])
# oxyHeal = np.array([0.05,0.075,0.1,0.0125,0.15])
# infectSpread = np.array([2, 4, 6])
# numRecurInj = 2
# numInfectRepeat = np.array([1, 2])
# seed = 0
injurySize = np.array([27])
oxyHeal = np.array([0.075])
infectSpread = np.array([2])
numRecurInj = 2
numInfectRepeat = np.array([1])
seed = 0
array_type = ctypes.c_float*numMatrixElements


data = [0, 0, 0, 0, 0]
for i in range(injurySize.shape[0]):
    for j in range(oxyHeal.shape[0]):
        for k in range(infectSpread.shape[0]):
            for m in range(numInfectRepeat.shape[0]):
                data = np.vstack(
                    [data, [oxyHeal[j], infectSpread[k], numRecurInj, numInfectRepeat[m], injurySize[i]]])
data = np.delete(data, 0, 0)

numcount = int(data.shape[0]/size)
for count in range(numcount):
    #    print(count)
    answerSum = 0
    for seed in range(100):
        if(rank+count*size<data.shape[0]):
            p1 = data[rank+count*size, 0]
            p2 = int(data[rank+count*size, 1])
            p3 = int(data[rank+count*size, 2])
            p4 = int(data[rank+count*size, 3])
            p5 = int(data[rank+count*size, 4])
            result = _IIRABM.mainSimulation(p1, p2, p3, p4, p5, seed, numMatrixElements, array_type(*internalParameterization))
        answerSum = answerSum+result
    if(answerSum>40 and answerSum<60):
        print(count, rank, p1, p2, p3, p4, p5, answerSum)
