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

# (oxyHeal,infectSpread,numRecurInj,numInfectRepeat,inj_number,seed,numMatrixElements,internalParameterization)
_IIRABM.mainSimulation.argtypes = (ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int)
_IIRABM.mainSimulation.restype = ndpointer(
    dtype=ctypes.c_float, shape=(20, 2000))

eliteFraction=0.1
numElites=int(eliteFraction*size)
np.random.seed(10287)

geneLow=-1.5
geneHigh=2
critfit=8000
nonZeroChance=5
tournamentSize=2
numBaseMatrixElements=429
numMatrixElements=432
baseGeneMutation=0.01
baseParamMutation=0.01
baseInjMutation=0.01
numOuterIters=20
numIters=50
numStochasticReplicates=20
array_type = ctypes.c_float*numMatrixElements
selectedTimePoints=np.array([240,480,720,960,1200,1440,1680,1920])
numDataPoints=selectedTimePoints.shape[0]

masterTNFnorm=0.1
masterIL4norm=0.075
masterIL10norm=0.2
masterGCSFnorm=1
masterIFNnorm=2

data=np.load('ARDS_Pos_D6_10.npy')
normalize=0;

mmfile1='MinMax/Mins_Pos'
mmfile2='MinMax/Maxs_Pos'
ipfile='InternalParameterization_Pos'
nvfile='NumViable_Pos'
fitfile='Fitness_Pos'

tnfMins=data[0,:]
tnfMaxs=data[1,:]
il4Mins=data[2,:]
il4Maxs=data[3,:]
gcsfMins=data[4,:]
gcsfMaxs=data[5,:]
il10Mins=data[6,:]
il10Maxs=data[7,:]
ifngMins=data[8,:]
ifngMaxs=data[9,:]

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
    if(normalize==1):
        tnfResult=normalizeResult(tnfResult)
        il4Result=normalizeResult(il4Result)
        il10Result=normalizeResult(il10Result)
        gcsfResult=normalizeResult(gcsfResult)
        ifngResult=normalizeResult(ifngResult)
    else:
        tnfResult=tnfResult*masterTNFnorm
        il4Result=il4Result*masterIL4norm
        il10Result=il10Result*masterIL10norm
        gcsfResult=gcsfResult*masterGCSFnorm
        ifngResult=ifngResult*masterIFNnorm

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
    # if((rand_i/1000)<imc):
    #     tempIndex=int(np.random.randint(low=0,high=4))
    #     ip[numBaseMatrixElements-4+tempIndex]=np.random.uniform(low=0,high=2)
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
    #This block places any candidates that do not die before the first comparator
    #time point into the breeding pool
    for i in range(iparray.shape[0]):
        if(fits[i]<critfit):
            breedables.append(iparray[i,:])
            bfits.append(fits[i])
            print("bfits",fits[i])
    breedables=np.asarray(breedables)
    bfits=np.asarray(bfits)
    sortedFitIndexes=np.argsort(bfits)
    ############################################################################
    #This block determines the number of parameterizations that have been removed
    # and replaces them with the most fit parameterizations
    addBreedables=[]
    addfits=[]
    if(bfits.shape[0]<size):
        diff=size-bfits.shape[0]  #number of candidates that were removed above
        print("Diff=",diff)
        k=0;
        for i in range(diff):
            tempBreeder=iparray[sortedFitIndexes[k],:]
            tempBreeder=mutate(tempBreeder,1,0,0)
            addBreedables.append(tempBreeder)
            addfits.append(bfits[k]+1)
            k=k+1;
            if(k>sortedFitIndexes.shape[0]-1):
                k=0
        addfits=np.asarray(addfits)
        addBreedables=np.asarray(addBreedables)
        print("AddFitsShape=",addfits.shape)
        bfits=np.hstack([bfits,addfits])
        breedables=np.vstack([breedables,addBreedables])
    ############################################################################
    #Randomly selecting 2 candidates and then the fitter one get put into the breeding pool
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
        print("winner=",winnerFit)
        breeders.append(winner)
        breederFits.append(winnerFit)

    breeders=np.asarray(breeders)
    breederFits=np.asarray(breederFits)

    return breeders,breederFits

def getNextGeneration(breeders,breederFits,gmc,imc,pmc):
    newIParray=np.zeros([size,numMatrixElements],dtype=np.float32)
    parentArray=np.zeros([size,numMatrixElements],dtype=np.float32)
    parentFitArray=np.zeros(size,dtype=np.float32)  ###MUST BE SOME ERROR HERE, maybe propagates to compareGenerations
    #taking 2 candidates, breeding them, then removing from pool
    print("Starting Next Gen, BreedersShape",breeders.shape,breederFits.shape)
    for i in range(0,size,2):
        if(breeders.shape[0]>=2):
            temp1=np.random.randint(low=0,high=breeders.shape[0])
            p1=breeders[temp1,:]
            parentFitArray[i]=breederFits[temp1]
#            breeders=np.delete(breeders,temp1,axis=0)
#            breederFits=np.delete(breederFits,temp1)
            temp2=np.random.randint(low=0,high=breeders.shape[0])
            p2=breeders[temp2,:]
            parentFitArray[i+1]=breederFits[temp2]
#            breeders=np.delete(breeders,temp2,axis=0)
#            breederFits=np.delete(breederFits,temp2)
            c1,c2=crossover(p1,p2)
            c1=mutate(c1,gmc,imc,pmc)
            c2=mutate(c2,gmc,imc,pmc)
            newIParray[i,:]=c1
            newIParray[i+1,:]=c2
            parentArray[i,:]=p1
            parentArray[i+1,:]=p2
    return newIParray,parentArray,parentFitArray

def compareGenerations(fitArray,iparray,parentArray,parentFitArray):
#Implementation of elitism, in this function, the $numElites least fit
#progeny are compared to the most fit parents to ensure that some of the most
#fit parameterizations remain in the algorithm
    sortFits=np.argsort(fitArray)
    sortParents=np.argsort(parentFitArray) #sorts low to high
    for i in range(numElites):
        compIndex=sortFits.shape[0]-(i+1) #comparing the most fit parent to least fit candidate
        if(parentFitArray[sortParents[i]]<fitArray[sortFits[compIndex]]):
            fitArray[sortFits[compIndex]]=parentFitArray[sortParents[i]]
            iparray[sortFits[compIndex],:]=parentArray[sortParents[i],:]
    return iparray,fitArray

def gaIter(recvbuf,iparray,parentArray,parentFitArray,genNumber,gmc,imc,pmc):
    fits=[]
    for i in range(size):
        fits.append(recvbuf[i])
    fits=np.asarray(fits)
#    if(genNumber>0):
    iparray,fits=compareGenerations(fits,iparray,parentArray,parentFitArray) #elitism: most fit parameterizations remain in breeding pool
    breeders,breederFits=getNextParents(fits,iparray) #doing the tournament, selecting breeders
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
        geneMutationChance=baseGeneMutation+0.005*i
        injMutationChance=baseInjMutation+0.005*i
        paramMutationChance=baseParamMutation+0.005*i
        myIP=comm.scatter(iparray,root=0)
        myFitness,MNS,MXS,numViable=getFitness(numStochasticReplicates,myIP,injSize) #comment this

        recvbufMin=None;
        recvbufMax=None;
        sendbufMin=np.float32(MNS.flatten())
        sendbufMax=np.float32(MXS.flatten())
        if(rank==0):
            recvbufMin=np.empty([size*40], dtype=np.float32)
            recvbufMax=np.empty([size*40], dtype=np.float32)
        comm.Gather(sendbufMin,recvbufMin,root=0)
        comm.Gather(sendbufMax,recvbufMax,root=0)
        if(rank==0):
            mnFile=str(mmfile1 + '_%s_%s.csv'%(k,i))
            mxFile=str(mmfile2 + '_%s_%s.csv'%(k,i))
            np.savetxt(mnFile,recvbufMin,delimiter=',')
            np.savetxt(mxFile,recvbufMax,delimiter=',')

        recvbufFit=None
        sendbufFit=np.float32(myFitness)
        if rank==0:
            recvbufFit=np.empty([size], dtype=np.float32)
        comm.Gather(sendbufFit, recvbufFit, root=0)

        recvbufV=None
        sendbufV=np.int16(numViable)
        if rank==0:
            recvbufV=np.empty([size], dtype=np.int16)
        comm.Gather(sendbufV, recvbufV, root=0)
        if(rank==0):
            fnamev=str(nvfile + '_IS%s_Gen%s_%s.csv'%(injSize,k,i))
            np.savetxt(fnamev,recvbufV,delimiter=',')

        if(rank==0):
            iname=str(ipfile + '_IS%s_Gen%s_%s.csv'%(injSize,k,i))
            fname=str(fitfile + '_IS%s_Gen%s_%s.csv'%(injSize,k,i))
            np.savetxt(iname,iparray,delimiter=',')
            iparray,avgFit,parentArray,parentFitArray=gaIter(recvbufFit,iparray,parentArray,parentFitArray,i,geneMutationChance,injMutationChance,paramMutationChance)
            np.savetxt(fname,recvbufFit,delimiter=',')
            averages.append(avgFit)
            print("Average Fitness=",avgFit)
