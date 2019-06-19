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
_IIRABM.mainSimulation.restype = ndpointer(
    dtype=ctypes.c_float, shape=(20, 100))


# Read Rule numMatrix
RuleMatrix = genfromtxt('RuleMatrix.csv', delimiter=',')
#print(RuleMatrix)
internalParameterization = RuleMatrix.flatten()

numMatrixElements = internalParameterization.shape[0]
#c_float_p = ctypes.POINTER(ctypes.c_float)
np.asarray(internalParameterization,dtype=np.float32)
#print(internalParameterization[16])
injurySize = 30
oxyHeal = 0.1
infectSpread = 2
numRecurInj = 0
numInfectRepeat = 2
seed = 0
array_type=ctypes.c_float*numMatrixElements;
print("Starting IIRABM")
test = _IIRABM.mainSimulation(oxyHeal, infectSpread, numRecurInj,
                              numInfectRepeat, injurySize, seed,
                              numMatrixElements,
                              array_type(*internalParameterization))
# print(test.type)
print(test.shape)
print(test)
np.save('Test.npy', test)
np.savetxt('TestRule.csv',test,delimiter=',')
