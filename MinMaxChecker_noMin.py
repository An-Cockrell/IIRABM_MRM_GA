import numpy as np
import os

tnfMins=np.array([0.947885074,1.598247366,2.058404265,2.256387083,2.199038987,1.963562366,1.623701284,1.224384024,0.789916636])
tnfMaxs=np.array([6.180651324,5.965206746,5.93996756,6.176902456,6.669168266,7.339562601,8.114341397,8.948576371,9.817961472])
tMax=np.max(tnfMaxs)
tnfMaxs=tnfMaxs/np.max(tMax)
tnfMins=tnfMins/np.max(tMax)

il4Mins=np.array([1.309698675,0,0,0,0,0,0,0,0])
il4Maxs=np.array([3.481727176,0,0,0,0,0,0,0,0])
il4Max=np.max(il4Maxs)
il4Mins=il4Mins/il4Max
il4Maxs=il4Maxs/il4Max

gcsfMins=np.array([103.7548431,94.14580326,74.56279206,52.86703522,33.97868338,18.87252035,7.265101859,0,0])
gcsfMaxs=np.array([194.9797422,159.4010515,140.6316002,129.7762341,121.0372697,112.6951296,104.4011245,96.17333074,88.11472345])
gcsfMax=np.max(gcsfMaxs)
gcsfMins=gcsfMins/gcsfMax
gcsfMaxs=gcsfMaxs/gcsfMax

il10Mins=np.array([11.08606329,5.403217376,1.094709419,0,0,0,0,0,0])
il10Maxs=np.array([21.7339152,14.93531557,11.61138007,8.674288263,7.604002664,8.204982326,8.88548607,8.680415013,11.26069095])
il10Max=np.max(il10Maxs)
il10Mins=il10Mins/il10Max
il10Maxs=il10Maxs/il10Max

ifngMins=np.array([6.22804419,6.572429796,6.541646886,6.097664539,5.36148558,4.490286326,3.581525757,2.682793205,1.816213722])
ifngMaxs=np.array([13.23136666,12.17794586,11.52552842,11.31120394,11.41306254,11.67305393,11.99287699,12.32413092,12.64390878])
ifngMax=np.max(ifngMaxs)
ifngMins=ifngMins/ifngMax
ifngMaxs=ifngMaxs/ifngMax

tolerance=0.1
numPts=8
countTol=numPts-1
totalCount=0

def checker(filename,filename2,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount,minarray,maxarray):
    x=np.loadtxt(filename,delimiter=',')
    y=np.loadtxt(filename2,delimiter=',')
    counter=0
    for k in range(numPts):
        if(x[0,k]<(tnfMaxs[k]+tolerance) and
            x[1,k]<(il4Maxs[k]+tolerance) and
            x[2,k]<(il10Maxs[k]+tolerance) and
            x[3,k]<(gcsfMaxs[k]+tolerance) and
            x[4,k]<(ifngMaxs[k]+tolerance)):
            counter=counter+1
    if(counter>=countTol):
        for k in range(numPts):
            if(x[0,k]>maxArray[0,k]):
                maxArray[0,k]=x[0,k]
            if(x[1,k]>maxArray[1,k]):
                maxArray[1,k]=x[1,k]
            if(x[2,k]>maxArray[2,k]):
                maxArray[2,k]=x[2,k]
            if(x[3,k]>maxArray[3,k]):
                maxArray[3,k]=x[3,k]
            if(x[4,k]>maxArray[4,k]):
                maxArray[4,k]=x[4,k]
            if(y[0,k]<minArray[0,k]):
                minArray[0,k]=y[0,k]
            if(y[1,k]<minArray[1,k]):
                minArray[1,k]=y[1,k]
            if(y[2,k]<minArray[2,k]):
                minArray[2,k]=y[2,k]
            if(y[3,k]<minArray[3,k]):
                minArray[3,k]=y[3,k]
            if(y[4,k]<minArray[4,k]):
                minArray[4,k]=y[4,k]
        totalCount=totalCount+1
        print("Success",totalCount)
  #          print(x)
  #          print(x.shape)
        tnfArray=np.vstack((tnfArray,x[0,:]))
        il4Array=np.vstack((il4Array,x[1,:]))
        il10Array=np.vstack((il10Array,x[2,:]))
        gcsfArray=np.vstack((gcsfArray,x[3,:]))
        ifngArray=np.vstack((ifngArray,x[4,:]))

    return tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount,minArray,maxArray

tnfArray=np.zeros([1,numPts],dtype=np.float32)
il4Array=np.zeros([1,numPts],dtype=np.float32)
il10Array=np.zeros([1,numPts],dtype=np.float32)
gcsfArray=np.zeros([1,numPts],dtype=np.float32)
ifngArray=np.zeros([1,numPts],dtype=np.float32)

minArray=np.zeros([5,numPts],dtype=np.float32)
maxArray=np.zeros([5,numPts],dtype=np.float32)

for i in range(5):
    minArray[i]=minArray[i]+100

for i in range(100):
    for j in range(100):
        for k in range(300):
            filename1=str('../Temp/MMTemp/Maxs_Neg_1_%s_%s_%s.csv'%(i,j,k))
            filename2=str('../Temp/MMTemp/Mins_Neg_1_%s_%s_%s.csv'%(i,j,k))

            if(os.path.exists(filename1) and os.path.exists(filename2)):
                tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount,minArray,maxArray=checker(filename1,filename2,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount,minArray,maxArray)

print("Total Count=",totalCount)
np.save('TnfArrayNoMin.npy',tnfArray)
np.save('IL4ArrayNoMin.npy',il4Array)
np.save('IL10ArrayNoMin.npy',il10Array)
np.save('GCSFArrayNoMin.npy',gcsfArray)
np.save('IFNGArrayNoMin.npy',ifngArray)

np.savetxt('MinArrayNoMin.csv',minArray,delimiter=',')
np.savetxt('MaxArrayNoMin.csv',maxArray,delimiter=',')
