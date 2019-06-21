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


# # Read Rule numMatrix
# RuleMatrix = genfromtxt('RuleMatrix.csv', delimiter=',')
# # print(RuleMatrix)
# injurySupplement=[-0.5,-0.05,-2,-0.25]
# internalParameterization = RuleMatrix.flatten()
# internalParameterization=np.hstack(internalParameterization,injurySupplement)

internalParameterization=np.load('baseParameterization.npy')
numMatrixElements = internalParameterization.shape[0]
#c_float_p = ctypes.POINTER(ctypes.c_float)
np.asarray(internalParameterization, dtype=np.float32)
# print(internalParameterization[16])
injurySize = np.array([25,27,30,32,35])
oxyHeal = np.array([0.05,0.075,0.1,0.0125,0.15])
infectSpread = np.array([2, 4, 6])
numRecurInj = 2
numInfectRepeat = np.array([1, 2])
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

test = _IIRABM.mainSimulation(oxyHeal, infectSpread, numRecurInj,
                              numInfectRepeat, injurySize, seed,
                              numMatrixElements,
                              array_type(*internalParameterization))
print(test.type)
