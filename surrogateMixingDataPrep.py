import numpy as np
#import keras
import os

mindist=1

def getMasterList():
    ips=[]
    fits=[]
    for i in range(1000):
        for j in range(1000):
            filename1=str('SurrogateMixingData/InternalParameterization_IS27_Gen%s_%s.csv'%(i,j))
            filename2=str('SurrogateMixingData/Fitness_IS27_Gen%s_%s.csv'%(i,j))
            if(os.path.exists(filename1)):
                x=np.loadtxt(filename1,delimiter=',')
                y=np.loadtxt(filename2,delimiter=',')
                for k in range(x.shape[0]):
                    ips.append(x[k,:])
                    fits.append(y[k])

    for i in range(1000):
        for j in range(1000):
            filename1=str('SurrogateMixingData/InternalParameterizationBase_IS27_Gen%s_%s.csv'%(i,j))
            filename2=str('SurrogateMixingData/FitnessBase_IS27_Gen%s_%s.csv'%(i,j))
            if(os.path.exists(filename1)):
                x=np.loadtxt(filename1,delimiter=',')
                y=np.loadtxt(filename2,delimiter=',')
                for k in range(x.shape[0]):
                    ips.append(x[k,:])
                    fits.append(y[k])


    ips=np.asarray(ips,dtype=np.float32)
    fits=np.asarray(fits,dtype=np.float32)
    return ips,fits

def trainingSetGenerator(listOfIPs,listOfFits):
    filteredIPs=np.zeros([1,432],dtype=np.float32)
    filteredFits=[]
    numIPs=listOfIPs.shape[0]
    print("Num IPS=",numIPs)
    filteredIPs[0,:]=listOfIPs[0,:]
    filteredFits.append(listOfFits[0])
    k1=0
#    for i in range(3456,3457):
    for i in range(numIPs):
        flag=0
        t2=listOfIPs[i,:]
        for j in range(k1+1):
            t1=filteredIPs[j,:]
#            print(t1,t2)
#            tempAns=np.sum(np.abs(np.subtract(t1,t2)))
            tempAns=np.linalg.norm(a-b)
            if(tempAns<mindist):
                flag=1
                break
        if(flag==0):
            filteredIPs=np.vstack((filteredIPs,listOfIPs[i,:]))
            filteredFits.append(listOfFits[i])
            k1=k1+1
    filteredFits=np.asarray(filteredFits,dtype=np.float32)
    return filteredIPs,filteredFits

IPs,Fits=getMasterList()

np.save('TempIPs.npy',IPs)
np.save('TempFits.npy',Fits)

# IPs=np.load('TempIPs.npy')
# Fits=np.load('TempFits.npy')

# For now, try without filtering
fIPs=IPs
fFits=Fits
# fIPs,fFits=trainingSetGenerator(IPs,Fits)
#
# np.save('FilteredIPs.npy',fIPs)
# np.save('FilteredFits.npy',fFits)
# #dist = numpy.linalg.norm(a-b)
#
# x=np.load('FilteredIPs.npy')
# y=np.load('FilteredFits.npy')
y=y.reshape(x.shape[0],1)
print(x.shape)
print(y.shape)
z=np.hstack((x,y))
np.random.shuffle(z)


numSamples=z.shape[0]
mark=int(numSamples*0.9)

trainingIPs=z[0:mark,0:432]
trainingFits=z[0:mark,432]
testIPs=z[mark+1:numSamples,0:432]
testFits=z[mark+1:numSamples,432]

mark2=numSamples-mark

trainingData=[]
trainingAnswers=[]
#0 means the first IP is the most fit, 1 means the second is
for i in range(0,mark,2):
    t1=trainingIPs[i,:]
    t2=trainingIPs[i+1,:]
    dat=np.hstack(t1,t2)
    f1=trainingFits[i]
    f2=trainingFits[i+1]
    trainingData.append(dat)
    if(f1<=f2):
        trainingAnswers.append(0)
    else:
        trainingAnswers.append(1)

for i in range(0,mark2,2):
    t1=testIPs[i,:]
    t2=testIPs[i+1,:]
    dat=np.hstack(t1,t2)
    f1=testFits[i]
    f2=testFits[i+1]
    testData.append(dat)
    if(f1<=f2):
        testAnswers.append(0)
    else:
        testAnswers.append(1)

np.save('SM_TrainingData.npy',trainingData)
np.save('SM_TrainingAnswers.npy',trainingAnswers)
np.save('SM_TestData.npy',testData)
np.save('SM_TestAnswers.npy',testAnswers)
#
#
# # trainingData=np.load('TrainingData.npy')
# # trainingAnswers=np.load('TrainingAnswers.npy')
# # testData=np.load()
# # testAnswers=np.load()
