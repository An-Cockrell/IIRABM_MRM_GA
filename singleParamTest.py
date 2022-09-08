import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from numpy import genfromtxt
import sys

#_IIRABM = ctypes.CDLL('/users/r/c/rcockrel/IIRABM_MRM_GA/IIRABM_RuleGA.so')
_IIRABM = ctypes.CDLL('/Users/rcockrel/IIRABM_MRM_GA/IIRABM_RuleGA.so')

# (oxyHeal,infectSpread,numRecurInj,numInfectRepeat,inj_number,seed,numMatrixElements,internalParameterization)
_IIRABM.mainSimulation.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int)
_IIRABM.mainSimulation.restype = ndpointer(
    dtype=ctypes.c_float, shape=(20, 2000))

internalParam=np.load('/Users/rcockrel/IIRABM_MRM_GA/baseParameterization2.npy')

oxyHeal=0.1
infectSpread=4
numRecurInj=2
numInfectRepeat=2
injurySize=33

seed=5
numMatrixElements=432
array_type = ctypes.c_float*numMatrixElements

result=_IIRABM.mainSimulation(oxyHeal, infectSpread, numRecurInj,
        numInfectRepeat, injurySize, seed,
        numMatrixElements, array_type(*internalParam),0)

#0 oxy
#2 tnf
#4 IL10
#5 GCSF
#12 IFN
#14 IL1
#15 IL4
#15 IL8

element=12
print(np.min(result[element,:]),np.max(result[element,:]))
