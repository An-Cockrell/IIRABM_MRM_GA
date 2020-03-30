import numpy as np

size=1
numMatrixElements=432
numBaseMatrixElements=429

def getInitialIP():
    internalParameterization=np.load('baseParameterization2.npy')
    numMatrixElements = internalParameterization.shape[0]
    np.asarray(internalParameterization, dtype=np.float32)
    print(internalParameterization.shape)
    internalParamArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    print(internalParamArray.shape)
    nonZeroMatEls=[]
    zeroMatEls=[]

    for i in range(numBaseMatrixElements):
        if(internalParameterization[i]!=0):
            nonZeroMatEls.append(i)
        else:
            zeroMatEls.append(i)
    nonZeroMatEls=np.asarray(nonZeroMatEls)
    zeroMatEls=np.asarray(zeroMatEls)
#    for i in range(size):
#        for j in range(nonZeroMatEls.shape[0]):
#            temp=np.random.uniform(low=geneLow,high=geneHigh)
#            internalParamArray[i,nonZeroMatEls[j]]=temp
#        for j in range(zeroMatEls.shape[0]):
#            chance=np.random.uniform(low=0,high=100)
#            if(chance<=nonZeroChance):
#                temp=np.random.uniform(low=geneLow,high=geneHigh)
#                internalParamArray[i,zeroMatEls[j]]=temp
    # for i in range(size):
    #     internalParamArray[i,429]=np.random.uniform(low=0,high=1)
    #     internalParamArray[i,430]=np.random.uniform(low=0,high=10)
    #     internalParamArray[i,431]=np.random.uniform(low=0,high=8)
    internalParamArray[0,:]=np.load('baseParameterization2.npy')
    return internalParamArray

x=getInitialIP()
np.asarray(x)
print(x.shape)
