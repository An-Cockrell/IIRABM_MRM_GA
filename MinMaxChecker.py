import numpy as np
import os

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

tolerance=0.1
countTol=14
totalCount=0


def checker(filename,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount):
    x=np.loadtxt(filename,delimiter=',')
    counter=0
    for k in range(15):
        if(x[0,k]<(tnfMaxs[k]+tolerance) and
            x[1,k]<(il4Maxs[k]+tolerance) and
            x[2,k]<(il10Maxs[k]+tolerance) and
            x[3,k]<(gcsfMaxs[k]+tolerance) and
            x[4,k]<(ifngMaxs[k]+tolerance)):
            counter=counter+1
        if(counter>=countTol):
            totalCount=totalCount+1
            print("Success")
  #          print(x)
  #          print(x.shape)
            tnfArray=np.vstack((tnfArray,x[0,:]))
            il4Array=np.vstack((il4Array,x[1,:]))
            il10Array=np.vstack((il10Array,x[2,:]))
            gcsfArray=np.vstack((gcsfArray,x[3,:]))
            ifngArray=np.vstack((ifngArray,x[4,:]))

    return tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount

tnfArray=np.zeros([1,15],dtype=np.float32)
il4Array=np.zeros([1,15],dtype=np.float32)
il10Array=np.zeros([1,15],dtype=np.float32)
gcsfArray=np.zeros([1,15],dtype=np.float32)
ifngArray=np.zeros([1,15],dtype=np.float32)

for i in range(250):
    for j in range(1050):
        filename1=str('MinMaxScan/MaxsH1_%s_%s.csv'%(i,j))
        filename2=str('MinMaxScan/MaxsK1_%s_%s.csv'%(i,j))
        filename3=str('MinMaxScan/MaxsK2_%s_%s.csv'%(i,j))
        filename4=str('MinMaxScan/MaxsK3_%s_%s.csv'%(i,j))
        filename5=str('MinMaxScan/MaxsK4_%s_%s.csv'%(i,j))
        filename6=str('MinMaxScan/MaxsK5_%s_%s.csv'%(i,j))
        if(os.path.exists(filename1)):
            tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount=checker(filename1,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount)
        if(os.path.exists(filename2)):
            tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount=checker(filename1,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount)
        if(os.path.exists(filename3)):
            tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount=checker(filename1,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount)
        if(os.path.exists(filename4)):
            tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount=checker(filename1,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount)
        if(os.path.exists(filename5)):
            tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount=checker(filename1,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount)
        if(os.path.exists(filename6)):
            tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount=checker(filename1,tnfArray,il4Array,il10Array,gcsfArray,ifngArray,totalCount)

print("Total COunt=",totalCount)
np.save('TnfArrayNoMin.npy',tnfArray)
np.save('IL4ArrayNoMin.npy',il4Array)
np.save('IL10ArrayNoMin.npy',il10Array)
np.save('GCSFArrayNoMin.npy',gcsfArray)
np.save('IFNGArrayNoMin.npy',ifngArray)
