#include <stdlib.h>
#include <vector>
#include <iostream>
#include "agents.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include "Parameters.h"
#include <random>
using namespace std;

extern mt19937 generator;
extern uniform_int_distribution<int> distribution100,distribution3;
extern int cellGrid[101][101],procID;
extern float system_oxy,oxyDeficit,totalInfection,total_TNF,total_sTNFr,total_IL10,
      	total_IL6,total_GCSF,total_proTH1,total_proTH2,oxyheal,total_IFNg,total_PAF,
      	total_IL1,total_IL4,total_IL8,total_IL12,total_sIL1r,total_IL1ra;
extern vector<EC> ecArray;
extern vector<int> ecIndexes;
extern vector<pmn> pmnArray;
extern vector<mono> monoArray;
extern vector<TH0> TH0array;
extern vector<TH1> TH1array;
extern vector<TH2> TH2array;
extern vector<pmn_marrow> pmn_marrowArray;
extern vector<mono_marrow> mono_marrowArray;
extern vector<TH0_germ> TH0_germArray;
extern vector<TH1_germ> TH1_germArray;
extern vector<TH2_germ> TH2_germArray;

extern float RM[numRules][numRuleParams];
extern float injurySupplement[4];

void wiggle(int* orientation, int* x, int* y);
void getAhead(int orient, int x, int y, int *xl, int *xm, int *xr, int *yl, int *ym, int *yr);
void adjustOrientation(int* orientation, int leftOrRight);
void move(int orient);
void updateSystemOxy(int step);
void evaporate();
void initialize();
void injure_infectionFRD(int inj_number);
void readIntervention();
void updateTrajectoryOutput(float allSignals[][numTimeSteps], int q);
float cytokineComboRule(int ruleRow, int cellIndex, float tnfr, float il1r);

void diffuse(){
	float nFactor;
	int i,j,totalCells;

	totalCells=xDim*yDim;
	nFactor=1.0/8.0;

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].endotoxin*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].endotoxin-=ecArray[i].endotoxin;
		ecArray[i].endotoxin+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].PAF*0.6*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].PAF-=(0.6*ecArray[i].PAF);
		ecArray[i].PAF+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].cytotox*0.4*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].cytotox-=(0.4*ecArray[i].cytotox);
		ecArray[i].cytotox+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].TNF*0.6*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].TNF-=(0.6*ecArray[i].TNF);
		ecArray[i].TNF+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].sTNFr*0.8*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].sTNFr-=(0.8*ecArray[i].sTNFr);
		ecArray[i].sTNFr+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL1*0.6*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].IL1-=(0.6*ecArray[i].IL1);
		ecArray[i].IL1+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IFNg*0.8*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].IFNg-=(0.8*ecArray[i].IFNg);
		ecArray[i].IFNg+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL8*0.6*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].IL8-=(0.6*ecArray[i].IL8);
		ecArray[i].IL8+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL10*0.8*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].IL10-=(0.8*ecArray[i].IL10);
		ecArray[i].IL10+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL1ra*0.8*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].IL1ra-=(0.8*ecArray[i].IL1ra);
		ecArray[i].IL1ra+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].sIL1r*0.8*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].sIL1r-=(0.8*ecArray[i].sIL1r);
		ecArray[i].sIL1r+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL12*0.8*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].IL12-=(0.8*ecArray[i].IL12);
		ecArray[i].IL12+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].IL4*0.8*nFactor);
		}
	}
	for(i=0;i<totalCells;i++){
		ecArray[i].IL4-=(0.8*ecArray[i].IL4);
		ecArray[i].IL4+=ecArray[i].tempSignal;
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].tempSignal=0;
//		cout<<"GCSF="<<ecArray[i].GCSF<<"\n";
	}
	for(i=0;i<totalCells;i++){
		for(j=0;j<8;j++){
			ecArray[ecArray[i].neighbors[j]].tempSignal+=(ecArray[i].GCSF*nFactor);
		}
	}

	for(i=0;i<totalCells;i++){
		ecArray[i].GCSF-=ecArray[i].GCSF;
		ecArray[i].GCSF+=ecArray[i].tempSignal;
	}
}

void evaporate(){
	int size,i;
	size=ecArray.size();
	for(i=0;i<size;i++){
		ecArray[i].endotoxin*=0.7;
		if(ecArray[i].endotoxin<0.01){ecArray[i].endotoxin=0;}
		ecArray[i].PAF*=0.7;
		if(ecArray[i].PAF<0.01){ecArray[i].PAF=0;}
		ecArray[i].cytotox*=0.7;
		if(ecArray[i].cytotox<0.01){ecArray[i].cytotox=0;}
		ecArray[i].TNF*=0.8;
		if(ecArray[i].TNF<0.01){ecArray[i].TNF=0;}
		ecArray[i].IL1*=0.8;
		if(ecArray[i].IL1<0.01){ecArray[i].IL1=0;}
		ecArray[i].sTNFr*=0.9;
		if(ecArray[i].sTNFr<0.01){ecArray[i].sTNFr=0;}
		ecArray[i].IL1ra*=0.9;
		if(ecArray[i].IL1ra<0.01){ecArray[i].IL1ra=0;}
		ecArray[i].sIL1r*=0.9;
		if(ecArray[i].sIL1r<0.01){ecArray[i].sIL1r=0;}
		ecArray[i].IFNg*=0.8;
		if(ecArray[i].IFNg<0.01){ecArray[i].IFNg=0;}
		ecArray[i].IL8*=0.7;
		if(ecArray[i].IL8<0.01){ecArray[i].IL8=0;}
		ecArray[i].IL10*=0.95;
		if(ecArray[i].IL10<0.01){ecArray[i].IL10=0;}
		ecArray[i].IL12*=0.8;
		if(ecArray[i].IL12<0.01){ecArray[i].IL12=0;}
		ecArray[i].IL4*=0.95;
		if(ecArray[i].IL4<0.01){ecArray[i].IL4=0;}
		ecArray[i].GCSF*=0.95;
		if(ecArray[i].GCSF<0.01){ecArray[i].GCSF=0;}
	}
}

void updateSystemOxy(int step){
	int size,i;

	system_oxy=0;
	oxyDeficit=0;
	totalInfection=0;
	total_TNF=0;
	total_sTNFr=0;
	total_IL10=0;
	total_GCSF=0;
	total_proTH1=0;
	total_proTH2=0;
	total_IFNg=0;
	total_PAF=0;
	total_IL1=0;
	total_IL4=0;
	total_IL8=0;
	total_IL12=0;
	total_sIL1r=0;
	total_IL1ra=0;
	size=ecArray.size();
	for(i=0;i<size;i++){
		system_oxy+=(ecArray[i].oxy/100);
		totalInfection+=(ecArray[i].infection/100);
		total_TNF+=(ecArray[i].TNF/100);
//		if(ecArray[i].TNF>mTNF){mTNF=ecArray[i].TNF;}
		total_sTNFr+=(ecArray[i].sTNFr/100);
//		if(ecArray[i].sTNFr>mSTNFR){mSTNFR=ecArray[i].sTNFr;}
		total_IL10+=(ecArray[i].IL10/100);
//		if(ecArray[i].IL10>mIL10){mIL10=ecArray[i].IL10;}
		total_GCSF+=(ecArray[i].GCSF/100);
//		if(ecArray[i].GCSF>mGCSF){mGCSF=ecArray[i].GCSF;}
		total_IFNg+=(ecArray[i].IFNg/100);
//		if(ecArray[i].IFNg>mIFNg){mIFNg=ecArray[i].IFNg;}
		total_PAF+=(ecArray[i].PAF/100);
//		if(ecArray[i].PAF>mPAF){mPAF=ecArray[i].PAF;}
		total_IL1+=(ecArray[i].IL1/100);
//		if(ecArray[i].IL1>mIL1){mIL1=ecArray[i].IL1;}
		total_IL4+=(ecArray[i].IL4/100);
//		if(ecArray[i].IL4>mIL4){mIL4=ecArray[i].IL4;}
		total_IL8+=(ecArray[i].IL8/100);
//		if(ecArray[i].IL8>mIL8){mIL8=ecArray[i].IL8;}
		total_IL12+=(ecArray[i].IL12/100);
//		if(ecArray[i].IL12>mIL12){mIL12=ecArray[i].IL12;}
		total_sIL1r+=(ecArray[i].sIL1r/100);
//		if(ecArray[i].sIL1r>mSIL1R){mSIL1R=ecArray[i].sIL1r;}
		total_IL1ra+=(ecArray[i].IL1ra/100);
//		if(ecArray[i].IL1ra>mIL1RA){mIL1RA=ecArray[i].IL1ra;}

	}
	size=TH0array.size();
	for(i=0;i<size;i++){
		total_proTH1+=(TH0array[i].proTH1/100);
		total_proTH2+=(TH0array[i].proTH2/100);
	}
	oxyDeficit=(xDim*yDim)-system_oxy;
//	if(step==1745){cout<<"ProcID="<<procID<<" oxy="<<oxyDeficit<<" Infect="<<totalInfection<<"\n";cout.flush();}

}

void adjustOrientation(int* orientation, int leftOrRight){
//if leftOrRight=-1, adjust orientation left, if =1, adjust orientation right
	int tempOrient;
	tempOrient=*orientation;
	tempOrient+=leftOrRight;
	if(tempOrient>7){tempOrient=0;}
	if(tempOrient<0){tempOrient=7;}
	*orientation=tempOrient;
}

void move(int orient, int* x, int* y){
	int oldx,newx,oldy,newy;

	oldx=*x;
	oldy=*y;
	newx=oldx;
	newy=oldy;

	if(orient==0){ //Move North
		newy=oldy+1;
		if(newy>=yDim){newy=0;}
	}

	if(orient==1){ //Move Northeast
		newy=oldy+1;
		newx=oldx+1;
		if(newy>=yDim){newy=0;}
		if(newx>=yDim){newx=0;}
	}

	if(orient==2){ //Move East
		newx=oldx+1;
		if(newx>=xDim){newx=0;}
	}

	if(orient==3){ //Move Southeast
		newy=oldy-1;
		newx=oldx+1;
		if(newx>=xDim){newx=0;}
		if(newy<0){newy=yDim-1;}
	}

	if(orient==4){ //Move South
		newy=oldy-1;
		if(newy<0){newy=yDim-1;}
	}

	if(orient==5){ //Move Southwest
		newy=oldy-1;
		newx=oldx-1;
		if(newy<0){newy=yDim-1;}
		if(newx<0){newx=xDim-1;}
	}

	if(orient==6){ //Move West
		newx=oldx-1;
		if(newx<0){newx=xDim-1;}
	}

	if(orient==7){ //Move Northwest
		newx=oldx-1;
		newy=oldy+1;
		if(newx<0){newx=xDim-1;}
		if(newy>=yDim){newy=0;}
	}

	if(cellGrid[newx][newy]<cellCapacity){
		*x=newx;
		*y=newy;
		cellGrid[oldx][oldy]--;
		cellGrid[newx][newy]++;
	}
}

void getAhead(int orient, int x, int y, int *xl, int *xm, int *xr, int *yl, int *ym, int *yr){
	int txl,txm,txr,tyl,tym,tyr;

	if(orient==0){
		txl=x-1;
		txm=x;
		txr=x+1;
		tyl=y+1;
		tym=y+1;
		tyr=y+1;
	}

	if(orient==1){
		txl=x;
		txm=x+1;
		txr=x+1;
		tyl=y+1;
		tym=y+1;
		tyr=y;
	}

	if(orient==2){
		txl=x+1;
		txm=x+1;
		txr=x+1;
		tyl=y+1;
		tym=y;
		tyr=y-1;
	}

	if(orient==3){
		txl=x+1;
		txm=x+1;
		txr=x;
		tyl=y;
		tym=y-1;
		tyr=y-1;
	}

	if(orient==4){
		txl=x+1;
		txm=x;
		txr=x-1;
		tyl=y-1;
		tym=y-1;
		tyr=y-1;
	}

	if(orient==5){
		txl=x;
		txm=x-1;
		txr=x-1;
		tyl=y-1;
		tym=y-1;
		tyr=y;
	}

	if(orient==6){
		txl=x-1;
		txm=x-1;
		txr=x-1;
		tyl=y-1;
		tym=y;
		tyr=y+1;
	}

	if(orient==7){
		txl=x-1;
		txm=x-1;
		txr=x;
		tyl=y;
		tym=y+1;
		tyr=y+1;
	}

	if(txl<0){txl=xDim-1;}
	if(txl>=xDim){txl=0;}
	if(txm<0){txm=xDim-1;}
	if(txm>=xDim){txm=0;}
	if(txr<0){txr=xDim-1;}
	if(txr>=xDim){txr=0;}
	if(tyl<0){tyl=yDim-1;}
	if(tyl>=yDim){tyl=0;}
	if(tym<0){tym=yDim-1;}
	if(tym>=yDim){tym=0;}
	if(tyr<0){tyr=yDim-1;}
	if(tyr>=yDim){tyr=0;}

	*xl=txl;
	*xm=txm;
	*xr=txr;
	*yl=tyl;
	*ym=tym;
	*yr=tyr;
}

void wiggle(int* orientation){ //Should always be followed by move() to match NetLogo
	int dir,tempOrient;
	tempOrient=*orientation;
	dir=distribution3(generator);
	if(dir==0){tempOrient--;}
	if(dir==2){tempOrient++;}
	if(tempOrient>7){tempOrient=0;}
	if(tempOrient<0){tempOrient=7;}
	*orientation=tempOrient;
}

void injure_infectionFRD(int inj_number){ //Fixed Radius Disk
	int i,x,y,rad,size,id;
	size=ecArray.size();
	rad=inj_number;
	for(i=0;i<size;i++){
		x=ecArray[i].xLoc-xDim/2;
		y=ecArray[i].yLoc-yDim/2;
		if((pow(x,2)+pow(y,2))<=(pow(rad,2))){
			id=ecArray[i].yLoc*xDim+ecArray[i].xLoc;
			ecArray[id].infection=100;
		}
	}
}

void recur_injury(){
	int x,y,id;
	x=distribution100(generator);
	y=distribution100(generator);
	id=y*xDim+x;
	ecArray[id].infection=100;
}

void EC::getNeighbors(){
	int nw,n,ne,e,se,s,sw,w;

	nw=id+xDim-1;
	n=id+xDim;
	ne=id+xDim+1;
	e=id+1;
	se=id-xDim+1;
	s=id-xDim;
	sw=id-xDim-1;
	w=id-1;

	if(id%xDim==0){ //object lies on west grid border
		nw=id+2*xDim-1;
		w=id+xDim-1;
		sw=id-1;
	}

	if(id%xDim==xDim-1){ //object lies on east grid border
		ne=id+1;
		e=id-(xDim-1);
		se=id-2*xDim+1;
	}

	if(id+xDim>=xDim*yDim){ //object lies on northern border
		n=(id+xDim)%(xDim*yDim);
		ne=(id+xDim)%(xDim*yDim)+1;
		nw=(id+xDim)%(xDim*yDim)-1;
		if(id%xDim==0){
			nw=(id+xDim)%(xDim*yDim)+xDim-1;
		}
		if(id%xDim==xDim-1){
			ne=(id+xDim)%(xDim*yDim)-(xDim-1);
		}
	}

	if(id<xDim){ //object lies on southern border
		s=(xDim*yDim)-xDim+id;
		se=(xDim*yDim)-xDim+id+1;
		sw=(xDim*yDim)-xDim+id-1;
		if(id%xDim==0){
			sw=(xDim*yDim)-1;
		}
		if(id%xDim==xDim-1){
			se=(xDim*yDim)-xDim;
		}
	}

	neighbors[0]=nw;
	neighbors[1]=n;
	neighbors[2]=ne;
	neighbors[3]=e;
	neighbors[4]=se;
	neighbors[5]=s;
	neighbors[6]=sw;
	neighbors[7]=w;
}

void initialize(){
	int i,j,k,xTemp,yTemp;
//	cout<<"ENTERING INITIALIZE\n";
	ecArray.clear();
	pmnArray.clear();
	monoArray.clear();
	TH0array.clear();
	TH1array.clear();
	TH2array.clear();
	pmn_marrowArray.clear();
	mono_marrowArray.clear();
	TH0_germArray.clear();
	TH1_germArray.clear();
	TH2_germArray.clear();

	system_oxy=10201;
	oxyDeficit=0;
	totalInfection=0;
	total_TNF=0;
	total_sTNFr=0;
	total_IL10=0;
	total_GCSF=0;
	total_proTH1=0;
	total_proTH2=0;

  for(i=0;i<xDim*yDim;i++){
    ecIndexes.push_back(i);
  }

	for(i=0;i<xDim;i++){
		for(j=0;j<yDim;j++){
			cellGrid[i][j]=0;}}

	k=0; //initialization
	for(j=0;j<yDim;j++){       //Initialize EC grid
		for(i=0;i<xDim;i++){
			ecArray.push_back(EC(i,j,k));
			ecArray[k].getNeighbors();
			k++;
		}
	}

	for(i=0;i<500;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
		pmnArray.push_back(pmn(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<50;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		monoArray.push_back(mono(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<50;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		TH1array.push_back(TH1(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<50;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		TH2array.push_back(TH2(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<100;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		pmn_marrowArray.push_back(pmn_marrow(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<100;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		mono_marrowArray.push_back(mono_marrow(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<100;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		TH0_germArray.push_back(TH0_germ(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<100;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		TH1_germArray.push_back(TH1_germ(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}

	for(i=0;i<100;i++){
		xTemp=distribution100(generator);
		yTemp=distribution100(generator);
//		cout<<xTemp<<" "<<yTemp<<"\n";
		TH2_germArray.push_back(TH2_germ(xTemp,yTemp));
		cellGrid[xTemp][yTemp]++;
	}
}

float cytokineProductionRule(int ruleRow, int index, float tnfr, float il1r){
  int i;
  float tempSum=0;
  float currentCyto[16];
  currentCyto[0]=ecArray[index].PAF;
  currentCyto[1]=ecArray[index].IL8;
  currentCyto[2]=ecArray[index].IL1ra;
  currentCyto[3]=ecArray[index].TNF;
  currentCyto[4]=ecArray[index].IL1;
  currentCyto[5]=tnfr;
  currentCyto[6]=il1r;
  currentCyto[7]=ecArray[index].IL1ra;
  currentCyto[8]=ecArray[index].sTNFr;
  currentCyto[9]=ecArray[index].sIL1r;
  currentCyto[10]=ecArray[index].GCSF;
  currentCyto[11]=ecArray[index].IL12;
  currentCyto[12]=ecArray[index].IL10;
  currentCyto[13]=ecArray[index].IL4;
  currentCyto[14]=ecArray[index].IFNg;
  currentCyto[15]=ecArray[index].endotoxin;
  for(i=0;i<numRuleParams-1;i++){
    tempSum+=currentCyto[i]*RM[ruleRow][i];
  }
//  cout<<"Const="<<RM[ruleRow][16]<<"\n";
  tempSum+=RM[ruleRow][16];
//  if(tempSum<0){tempSum=0;}
  return tempSum;
}

float cytokineComboRule(int ruleRow, int index, float tnfr, float il1r){
  int i;
  float tempSum=0;
  float currentCyto[16];
  currentCyto[0]=ecArray[index].PAF;
  currentCyto[1]=ecArray[index].IL8;
  currentCyto[2]=ecArray[index].IL1ra;
  currentCyto[3]=ecArray[index].TNF;
  currentCyto[4]=ecArray[index].IL1;
  currentCyto[5]=tnfr;
  currentCyto[6]=il1r;
  currentCyto[7]=ecArray[index].IL1ra;
  currentCyto[8]=ecArray[index].sTNFr;
  currentCyto[9]=ecArray[index].sIL1r;
  currentCyto[10]=ecArray[index].GCSF;
  currentCyto[11]=ecArray[index].IL12;
  currentCyto[12]=ecArray[index].IL10;
  currentCyto[13]=ecArray[index].IL4;
  currentCyto[14]=ecArray[index].IFNg;
  currentCyto[15]=ecArray[index].endotoxin;
  for(i=0;i<numRuleParams-1;i++){
    tempSum+=currentCyto[i]*RM[ruleRow][i];
  }
  tempSum+=RM[ruleRow][16];
  if(tempSum<0){tempSum=0;}
  return tempSum;
}

void updateTrajectoryOutput(float allSignals[][numTimeSteps], int q){
	allSignals[0][q]=oxyDeficit;
	allSignals[1][q]=totalInfection;
	allSignals[2][q]=total_TNF;
	allSignals[3][q]=total_sTNFr;
	allSignals[4][q]=total_IL10;
	allSignals[5][q]=total_GCSF;
	allSignals[6][q]=total_proTH1;
	allSignals[7][q]=total_proTH2;
	allSignals[8][q]=pmnArray.size();
	allSignals[9][q]=monoArray.size();
	allSignals[10][q]=TH1array.size();
	allSignals[11][q]=TH2array.size();
	allSignals[12][q]=total_IFNg;
	allSignals[13][q]=total_PAF;
	allSignals[14][q]=total_IL1;
	allSignals[15][q]=total_IL4;
	allSignals[16][q]=total_IL8;
	allSignals[17][q]=total_IL12;
	allSignals[18][q]=total_sIL1r;
	allSignals[19][q]=total_IL1ra;
}

void getRuleMatrix(float internalParam[], int numMatEls){
  int i,j,k=0;
  for(i=0;i<numRules;i++){
    for(j=0;j<numRuleParams;j++){
//      cout<<"INPUT="<<internalParam[k]<<"\n";
      RM[i][j]=internalParam[k];
      k++;
    }
  }
  for(i=0;i<4;i++){
    injurySupplement[i]=internalParam[k];
    k++;
  }
// cout<<"RM[0][16]="<<RM[0][16]<<"\n";
}
