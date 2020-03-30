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

# Ctypes initialization
#_IIRABM = ctypes.CDLL('/home/chase/iirabm_fullga/IIRABM_RuleGA.so')
_IIRABM = ctypes.CDLL('/users/r/c/rcockrel/IIRABM_MRM_GA/IIRABM_RuleGA.so')
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
numOuterIters=2
numIters=5
numStochasticReplicates=2
array_type = ctypes.c_float*numMatrixElements
selectedTimePoints=np.array([29,59,89,119,149,179,209,239,359,479,719,1199,1919,3599,5279])
numDataPoints=selectedTimePoints.shape[0]

tnfMins=np.array([22.34,20.74,0,0,9.57,1.6,9.57,0,1.6,0,0,0,14.36,19.15,15.96])
tnfMaxs=np.array([135.64,79,47.87,43.09,49.47,47.87,55.85,43.09,60.64,57.45,97.34,121.28,84.57,49.47,76.60])
tMax=np.max(tnfMaxs)
tnfMaxs=tnfMaxs/np.max(tMax)
tnfMins=tnfMins/np.max(tMax)

il4Mins=np.array([14,12,0,1,0,0,0,0,0,0,0,0,2,13,0])
il4Maxs=np.array([213.53,93.23,58.15,52.13,45.11,36.09,50.13,49.12,49.12,59.15,99.25,151.38,109.27,54.14,69.17])
il4Max=np.max(il4Maxs)
il4Mins=il4Mins/il4Max
il4Maxs=il4Maxs/il4Max

gcsfMins=np.array([47.75,15.92,0,0,0,0,0,19.89,47.75,23.87,3.98,0,0,3.98,0])
gcsfMaxs=np.array([3846,11071,1102,823,831,107,139,159,640,405,508,1090,342,413,441.0])
gcsfMax=np.max(gcsfMaxs)
gcsfMins=gcsfMins/gcsfMax
gcsfMaxs=gcsfMaxs/gcsfMax

il10Mins=np.array([34.48,11.94,0,2.65,7.96,3.98,23.87,0,11.94,1.33,2.65,1.33,0,1.33,3.98])
il10Maxs=np.array([228,199,454,198,228,243,284,118,842,122,184,3842,49,15,14.0])
il10Max=np.max(il10Maxs)
il10Mins=il10Mins/il10Max
il10Maxs=il10Maxs/il10Max

ifngMins=np.array([52,0,4.76,0,4.76,0,9.52,0,4.76,9.52,0,4.76,9.52,0,4.76])
ifngMaxs=np.array([11071,2857,974,850,902,759,1136,902,1017,1218,1700,2142,2142,754,587.0])
ifngMax=np.max(ifngMaxs)
ifngMins=ifngMins/ifngMax
ifngMaxs=ifngMaxs/ifngMax

masterTNFnorm=334.0
masterIL4norm=2162.0
masterIL10norm=28225.0
masterGCSFnorm=8238.0
masterIFNnorm=37726.0

geneLow=-1.5
geneHigh=2
critfit=15000

eliteFraction=0.1
numElites=int(eliteFraction*size)
#print("NE=",numElites)

np.random.seed(10287)

def getRandomIP():
    internalParamArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    for i in range(size):
        for j in range(numBaseMatrixElements):
            temp=np.random.uniform(low=geneLow,high=geneHigh)
            internalParamArray[i,j]=temp
    internalParamArray[0,:]=np.load('baseParameterization2.npy')
    return internalParamArray

def getFitnessResult(result,index):
    selectResult=np.zeros(numDataPoints,dtype=np.float32)
    for j in range(numDataPoints):
        if(result[index,selectedTimePoints[j]-1]<0):
            result[index,selectedTimePoints[j]-1]=0
        selectResult[j]=result[index,selectedTimePoints[j]-1]
    return selectResult

def getFitnessResult2(result,index):
    selectResult=np.zeros(numDataPoints,dtype=np.float32)
    for j in range(numDataPoints):
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

def getFitness(numReplicates,internalParam,injSize):
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
    injurySize=injSize
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
    return fitsum,retMn,retMx,numViable

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
    # for i in range(size):
    #     internalParamArray[i,429]=np.random.uniform(low=0,high=1)
    #     internalParamArray[i,430]=np.random.uniform(low=0,high=10)
    #     internalParamArray[i,431]=np.random.uniform(low=0,high=8)
    internalParamArray[0,:]=np.load('baseParameterization2.npy')
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
    # if((rand_p/1000)<pmc):
    #     tempIndex=int(np.random.uniform(low=0,high=3))
    #     if(tempIndex==0):
    #         tempElement=ip[numBaseMatrixElements+tempIndex]
    #         tempRand=np.random.randint(low=0,high=2)
    #         if(tempRand==0):
    #             tempElement=tempElement-0.05
    #             if(tempElement<0):
    #                 tempElement=0.01
    #         else:
    #             tempElement=tempElement+0.05
    #         ip[numBaseMatrixElements+tempIndex]=tempElement
    #     else:
    #         tempElement=ip[numBaseMatrixElements+tempIndex]
    #         tempRand=np.random.randint(low=0,high=2)
    #         if(tempRand==0):
    #             tempElement=tempElement-1
    #             if(tempElement<0):
    #                 tempElement=1
    #         else:
    #             tempElement=tempElement+1
    #             if(tempElement>8):
    #                 tempElement=8
    #         ip[numBaseMatrixElements+tempIndex]=tempElement
    return ip

def getNextParents(fits,iparray):
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

    return breeders,breederFits

def getNextGeneration(breeders,breederFits,gmc,imc,pmc):
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

def gaIter(recvbuf,iparray,parentArray,parentFitArray,genNumber,gmc,imc,pmc):
    fits=[]
    for i in range(size):
        fits.append(recvbuf[i])
    fits=np.asarray(fits)
#    if(genNumber>0):
    iparray,fits=compareGenerations(fits,iparray,parentArray,parentFitArray)
    breeders,breederFits=getNextParents(fits,iparray)
    newIParray,parents,parentFits=getNextGeneration(breeders,breederFits,gmc,imc,pmc)
    avgFit=np.mean(breederFits)
    return newIParray,avgFit,parents,parentFits

if(rank==0):
    iparray=getRandomIP();
    averages=[]
else:
    iparray=None

parentArray=np.zeros([size,numMatrixElements])
parentFitArray=np.zeros(size)
parentFitArray=parentFitArray+1000000

injSize=int(sys.argv[1])

for k in range(numOuterIters):
    for i in range(numIters):
        geneMutationChance=baseGeneMutation+0.001*i
        injMutationChance=baseInjMutation+0.001*i
        paramMutationChance=baseParamMutation+0.001*i
        myIP=comm.scatter(iparray,root=0)
        myFitness,MNS,MXS,numViable=getFitness(numStochasticReplicates,myIP,injSize)

        mnFile=str('MinMax/Mins_%s_%s_%s_%s.csv'%(runID,k,i,rank))
        mxFile=str('MinMax/Maxs_%s_%s_%s_%s.csv'%(runID,k,i,rank))
        np.savetxt(mnFile,MNS,delimiter=',')
        np.savetxt(mxFile,MXS,delimiter=',')

        recvbuf=None
        sendbuf=np.float32(myFitness)
        if rank==0:
            recvbuf=np.empty([size], dtype=np.float32)
        comm.Gather(sendbuf, recvbuf, root=0)

        recvbuf2=None
        sendbuf2=np.int16(numViable)
        if rank==0:
            recvbuf2=np.empty([size], dtype=np.int16)
        comm.Gather(sendbuf2, recvbuf2, root=0)
        if(rank==0):
            fnamev=str('NumViable_IS%s_Gen%s_%s.csv'%(injSize,k,i))
            np.savetxt(fnamev,iparray,delimiter=',')

        if(rank==0):
            iname=str('InternalParameterization_IS%s_Gen%s_%s.csv'%(injSize,k,i))
            fname=str('Fitness_IS%s_Gen%s_%s.csv'%(injSize,k,i))
            np.savetxt(iname,iparray,delimiter=',')
            iparray,avgFit,parentArray,parentFitArray=gaIter(recvbuf,iparray,parentArray,parentFitArray,i,geneMutationChance,injMutationChance,paramMutationChance)
            np.savetxt(fname,recvbuf,delimiter=',')
            averages.append(avgFit)
            print("Average Fitness=",avgFit)
