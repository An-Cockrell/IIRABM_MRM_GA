#include <vector>
#include <random>
#include <stdlib.h>
#include <algorithm>
#include "agents.h"
#include "Parameters.h"

using namespace std;

void initialize();
void simulationStep(int step,int infectSpread,int numInfectRepeat,float oxyHeal,int numRecurInj, int numABX);
void cellStep(int infectSpread,int numInfectRepeat,float oxyHeal,int numRecurInj);
void recurrentInjury(int step,int numRecurInj);
void giveABX(int step, int *numABX);

extern void injure_infectionFRD(int inj_number);
extern void updateSystemOxy(int step);
extern void evaporate();
extern void diffuse();
extern void recur_injury();
extern void applyAntibiotics();
extern void updateTrajectoryOutput(float allSignals[][numTimeSteps], int q);
extern void getRuleMatrix(float internalParam[], int numMatEls);

vector<EC> ecArray;
vector<int> ecIndexes;
vector<pmn> pmnArray;
vector<mono> monoArray;
vector<TH0> TH0array;
vector<TH1> TH1array;
vector<TH2> TH2array;
vector<pmn_marrow> pmn_marrowArray;
vector<mono_marrow> mono_marrowArray;
vector<TH0_germ> TH0_germArray;
vector<TH1_germ> TH1_germArray;
vector<TH2_germ> TH2_germArray;
mt19937 generator;
uniform_int_distribution<int> distribution10k(0,9999);
uniform_int_distribution<int> distribution1000(0,999);
uniform_int_distribution<int> distribution100(0,99);
uniform_int_distribution<int> distribution50(0,49);
uniform_int_distribution<int> distribution12(0,11);
uniform_int_distribution<int> distribution10(0,9);
uniform_int_distribution<int> distribution9(0,8);
uniform_int_distribution<int> distribution8(0,7);
uniform_int_distribution<int> distribution5(0,4);
uniform_int_distribution<int> distribution3(0,2);
uniform_int_distribution<int> distribution2(0,1);

float RM[numRules][numRuleParams];

float system_oxy,oxyDeficit,totalInfection,total_TNF,total_sTNFr,total_IL10,
total_IL6,total_GCSF,total_proTH1,total_proTH2,total_IFNg,total_PAF,
total_IL1,total_IL4,total_IL8,total_IL12,total_sIL1r,total_IL1ra;

int d_pmn[numTimeSteps],d_mono[numTimeSteps],d_TH1[numTimeSteps],d_TH2[numTimeSteps];
int cellGrid[101][101];
float allSignals[20][numTimeSteps];
float allSignalsReturn[20*numTimeSteps];

extern const int cellCapacity,xDim,yDim,injuryStep,parameterInput,numTimeSteps;
extern const float antibioticMultiplier;

extern "C" float* mainSimulation(float oxyHeal, int infectSpread,
	int numRecurInj, int numInfectRepeat, int inj_number, int seed, int numMatrixElements, float* internalParameterization){

		int i,step,iend,jend,antibiotic1,antibiotic2,istep,k,j;
		int numABX;
		generator.seed(seed);
		cout<<"oxyHeal="<<oxyHeal<<"\n";
//		return allSignalsReturn;
//		cout<<"Test10\n";
		getRuleMatrix(internalParameterization,numMatrixElements);
		initialize();
		step=0;
		istep=0;
		ecIndexes.clear();
		for(i=0;i<xDim*yDim;i++){
			ecIndexes.push_back(i);
		}
		antibiotic1=0;
		antibiotic2=0;
		numABX=0;
		injure_infectionFRD(inj_number);
		for(i=0;i<numTimeSteps;i++){

			step++;
			istep++;
			if(step==injuryStep){step=1;}
			antibiotic1++;
			antibiotic2++;

			simulationStep(i,infectSpread,numInfectRepeat,oxyHeal,numRecurInj,numABX);
			updateSystemOxy(istep);
			updateTrajectoryOutput(allSignals,i);
			cout<<"i="<<i<<" "<<oxyDeficit<<"\n";
			if(oxyDeficit>8161||(oxyDeficit<5&&i>0)){
				for(iend=i+1;iend<numTimeSteps;iend++){
					for(jend=0;jend<20;jend++){
							allSignals[jend][iend]=-1.0;
					}
				}
				break;
			}
			if(oxyDeficit>=8161){
					break;
			}

			if(i==62){
// 				for(j=0;j<monoArray.size();j++){
// //					cout<<j<<" "<<monoArray[j].xLoc<<" "<<monoArray[j].yLoc<<"\n";
// 					cout<<j<<" "<<monoArray[j].IL_1r<<"\n";
// 					cout<<j<<" "<<monoArray[j].TNFr<<"\n";
// 				}
				cout<<"Total IL10="<<total_IL10<<"\n";
				cout<<"Total IL1="<<total_IL1<<"\n";
				cout<<"Total TNF="<<total_TNF<<"\n";
				// for(j=0;j<ecArray.size();j++){
				// 	cout<<j<<" "<<ecArray[j].endotoxin<<"\n";
				// }
			}

		}

		k=0;
		for(j=0;j<20;j++){
			for(i=0;i<numTimeSteps;i++){
				allSignalsReturn[k]=allSignals[j][i];
				k++;
			}
		}
		return allSignalsReturn;
}

void simulationStep(int step,int infectSpread,int numInfectRepeat,float oxyHeal,int numRecurInj, int numABX){
  cellStep(infectSpread,numInfectRepeat,oxyHeal,numRecurInj);
  evaporate();
  recurrentInjury(step,numRecurInj);
  diffuse();
  if(antibioticMultiplier>0){
    giveABX(step, &numABX);
  }
}

void cellStep(int infectSpread,int numInfectRepeat,float oxyHeal,int numRecurInj){
  int length,j;
  length=TH0array.size();
  if(length>0){
    shuffle(TH0array.begin(),TH0array.end(),generator);}
  j=0;
  while(j<length){
    TH0array[j].TH0function(j);
    j++;
    length=TH0array.size();}
  length=ecArray.size();
  shuffle(ecIndexes.begin(),ecIndexes.end(),generator);
  j=0;
  while(j<length){
    ecArray[ecIndexes[j]].inj_function(infectSpread,numInfectRepeat);
    ecArray[ecIndexes[j]].ECfunction(oxyHeal);
    j++;
    length=ecArray.size();}
  length=pmnArray.size();
  if(length>0){
    shuffle(pmnArray.begin(),pmnArray.end(),generator);}
  j=0;
  while(j<length){
    pmnArray[j].pmn_function(j);
    j++;
    length=pmnArray.size();}
  length=monoArray.size();
    if(length>0){
      shuffle(monoArray.begin(),monoArray.end(),generator);}
  j=0;
  while(j<length){
    monoArray[j].mono_function(j);
    j++;
    length=monoArray.size();}
  length=TH1array.size();
    if(length>0){
      shuffle(TH1array.begin(),TH1array.end(),generator);}
  j=0;
  while(j<length){
    TH1array[j].TH1function(j);
    j++;
    length=TH1array.size();}
  length=TH2array.size();
    if(length>0){
      shuffle(TH2array.begin(),TH2array.end(),generator);}
  j=0;
  while(j<length){
    TH2array[j].TH2function(j);
    j++;
    length=TH2array.size();}
  length=pmn_marrowArray.size();
    if(length>0){
      shuffle(pmn_marrowArray.begin(),pmn_marrowArray.end(),generator);}
  j=0;
  while(j<length){
    pmn_marrowArray[j].pmn_marrow_function();
    j++;
    length=pmn_marrowArray.size();}

  length=mono_marrowArray.size();
    if(length>0){
      shuffle(mono_marrowArray.begin(),mono_marrowArray.end(),generator);}
  j=0;
  while(j<length){
    mono_marrowArray[j].mono_marrow_function();
    j++;
    length=mono_marrowArray.size();}

  length=TH1_germArray.size();
    if(length>0){
      shuffle(TH1_germArray.begin(),TH1_germArray.end(),generator);}
  j=0;
  while(j<length){
    TH1_germArray[j].TH1_germ_function();
    j++;
    length=TH1_germArray.size();}

  length=TH2_germArray.size();
    if(length>0){
      shuffle(TH2_germArray.begin(),TH2_germArray.end(),generator);}
  j=0;
  while(j<length){
    TH2_germArray[j].TH2_germ_function();
    j++;
    length=TH2_germArray.size();}

  length=TH0_germArray.size();
    if(length>0){
      shuffle(TH0_germArray.begin(),TH0_germArray.end(),generator);}
  j=0;
  while(j<length){
    TH0_germArray[j].TH0_germ_function();
    j++;
    length=TH0_germArray.size();
  }
}

void recurrentInjury(int step, int numRecurInj){
  int i;
  if(step==injuryStep-1){
    for(i=1;i<=numRecurInj;i++){
      recur_injury();
    }
  }
}

void giveABX(int step, int *numABX){
  if((step%injuryStep==102)&&(*numABX<1100)){
    applyAntibiotics();
    *numABX++;
  }
}
