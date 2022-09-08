import numpy as np

tnfMins=np.array([1.62,2.92,4.00,4.76,5.09,4.97,4.50,3.80,2.95])
tnfMaxs=np.array([14.96,14.22,13.69,13.48,13.70,14.37,15.39,16.64,18.04])
il4Mins=np.array([0.11,0.10,0.09,0.05,0.00,0.00,0.00,0.00,0.00])
il4Maxs=np.array([0.58,0.50,0.43,0.37,0.34,0.32,0.31,0.31,0.31])
gcsfMins=np.array([478.28,84.94,0.00,0.00,0.00,0.00,0.00,0.00,0.00])
gcsfMaxs=np.array([1469.02,1025.06,841.67,635.77,452.61,309.70,206.08,134.34,86.20])
il10Mins=np.array([58.91,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00])
il10Maxs=np.array([244.64,120.99,32.55,16.22,25.48,41.30,61.66,79.03,83.73])
ifngMins=np.array([13.01,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00])
ifngMaxs=np.array([91.78,65.97,47.75,28.88,15.96,8.36,4.22,2.08,1.01])

data=np.vstack((tnfMins,tnfMaxs,il4Mins,il4Maxs,gcsfMins,gcsfMaxs,il10Mins,il10Maxs,ifngMins,ifngMaxs))
np.save('ARDS_Pos_D6_10.npy',data)

tMax=np.max(tnfMaxs)
tnfMaxs=tnfMaxs/np.max(tMax)
tnfMins=tnfMins/np.max(tMax)
il4Max=np.max(il4Maxs)
il4Mins=il4Mins/il4Max
il4Maxs=il4Maxs/il4Max
gcsfMax=np.max(gcsfMaxs)
gcsfMins=gcsfMins/gcsfMax
gcsfMaxs=gcsfMaxs/gcsfMax
il10Max=np.max(il10Maxs)
il10Mins=il10Mins/il10Max
il10Maxs=il10Maxs/il10Max
ifngMax=np.max(ifngMaxs)
ifngMins=ifngMins/ifngMax
ifngMaxs=ifngMaxs/ifngMax

data=np.vstack((tnfMins,tnfMaxs,il4Mins,il4Maxs,gcsfMins,gcsfMaxs,il10Mins,il10Maxs,ifngMins,ifngMaxs))
np.save('ARDS_Pos_D6_10_Norm.npy',data)

tnfMins=np.array([2.07,2.53,2.89,3.10,3.16,3.12,3.01,2.88,2.72])
tnfMaxs=np.array([5.06,5.03,5.11,5.34,5.71,6.19,6.72,7.29,7.88])

il4Mins=np.array([0.24,0.25,0.21,0.10,0.00,0.00,0.00,0.00,0.00])
il4Maxs=np.array([1.38,1.20,1.06,0.98,0.95,0.96,1.00,1.05,1.10])

gcsfMins=np.array([129.70,161.90,96.95,0.00,0.00,0.00,0.00,0.00,0.00])
gcsfMaxs=np.array([876.57,696.08,634.58,625.03,621.89,612.27,594.49,569.83,540.11])

il10Mins=np.array([11.09,5.40,1.09,0.00,0.00,0.00,0.00,0.00,0.00])
il10Maxs=np.array([21.73,14.94,11.61,8.67,7.60,8.20,8.89,8.68,11.26])

ifngMins=np.array([6.32,6.57,6.22,5.35,4.25,3.13,2.05,1.07,0.18])
ifngMaxs=np.array([13.70,12.23,11.43,11.22,11.30,11.48,11.66,11.80,11.91])

data=np.vstack((tnfMins,tnfMaxs,il4Mins,il4Maxs,gcsfMins,gcsfMaxs,il10Mins,il10Maxs,ifngMins,ifngMaxs))
np.save('ARDS_Neg_D6_10.npy',data)

tMax=np.max(tnfMaxs)
tnfMaxs=tnfMaxs/np.max(tMax)
tnfMins=tnfMins/np.max(tMax)
il4Max=np.max(il4Maxs)
il4Mins=il4Mins/il4Max
il4Maxs=il4Maxs/il4Max
gcsfMax=np.max(gcsfMaxs)
gcsfMins=gcsfMins/gcsfMax
gcsfMaxs=gcsfMaxs/gcsfMax
il10Max=np.max(il10Maxs)
il10Mins=il10Mins/il10Max
il10Maxs=il10Maxs/il10Max
ifngMax=np.max(ifngMaxs)
ifngMins=ifngMins/ifngMax
ifngMaxs=ifngMaxs/ifngMax

data=np.vstack((tnfMins,tnfMaxs,il4Mins,il4Maxs,gcsfMins,gcsfMaxs,il10Mins,il10Maxs,ifngMins,ifngMaxs))
np.save('ARDS_Neg_D6_10_Norm.npy',data)
