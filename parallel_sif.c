// Included Fracture Criteria in Neighbor List
// Max-Min stress calculation called only after Tequilibrate
// Zeroed the EPSPART after a damage has occurred

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define THREADS 80
#define NX      91
#define NY      169
#define DY      0.001
#define N       15324 //NX*NY + 576
#define NB      91 //tip mass
#define NINIT   91
#define RHO0    2.70e3
#define MASSP   0.002311783926 //Comes from consistency 0.002320081245
#define MASSB   1.00
#define VBALL   0.0e-3
#define SIGMA   0.80
#define EPS     100.0
#define RADIUS  6.0
#define MAX     10000.0
#define DT      0.000000001
#define H       0.0016
#define M       2.0
#define EMOD    7.14e10
#define G       2.86e10
#define B       4.76e10
#define YIELD   276.0e6
#define EPSMAX  0.00483
#define ULTI    0.310
#define NPRINT  10000
#define alpha   0.5
#define beta    0.5
#define NCOUNT	20 //The number of neighbor to be included
#define ART_VISCOSITY 1
#define MONA_CORR     1
#define MONA_CORR2    0
#define JAUMANN       1
#define TENSILE       1
#define GRADCORR      1 //Gradient Correction
#define GRADCORREN    0 //Symmetric Form of Gradient Correction
#define PSEUDOSPRING  1 //Pseudospring approach: 1 is strain governed, 2 is stress governed, 0 is no pseudospring
#define SIGAPP		  1.0e6 // Mean Stress
#define SIGAMP        0.0e6 // Amplitude of Stress
#define TPERIOD       0.0001
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAXI(a,b) (((a)>(b))?(a):(b))
#define PI            3.1415926535
#define PARISC		  2.838573e-12
#define SQRTPIPHI	  0.02802495608
#define PARISM		  4.60

double X[N], Y[N], VX[N], VY[N], RHO[N];
double E[N], SXX[N], SXY[N], SYY[N];
double STRAINXX[N], STRAINXY[N], STRAINYY[N];
double XO[N], YO[N], VXO[N], VYO[N], RHOO[N]; //For Storing Old variables
double EO[N], SXXO[N], SXYO[N], SYYO[N];
double XN[N], YN[N], VXN[N], VYN[N], RHON[N]; //For Storing New variables
double EN[N], SXXN[N], SXYN[N], SYYN[N];
double STRAINXXO[N], STRAINXYO[N], STRAINYYO[N];
double P[N], KE[N], PE[N], MASS[N];
double STRAINXXN[N], STRAINXYN[N], STRAINYYN[N];
double STRAINXXDOT[N], STRAINXYDOT[N], STRAINYYDOT[N];
double XDOT[N], YDOT[N], VXDOT[N], VYDOT[N], RHODOT[N], EDOT[N], SXXDOT[N], SXYDOT[N], SYYDOT[N];
double DVXDX[N], DVXDY[N], DVYDX[N], DVYDY[N], FX[N], FY[N], VXBAR[N], VYBAR[N];
double EPLXX[N],EPLYY[N],EPLXY[N];
double SIGXXMAX[N],SIGXYMAX[N],SIGYYMAX[N];
double SIGXXMIN[N],SIGXYMIN[N],SIGYYMIN[N];
double DELSIGXX[N],DELSIGXY[N],DELSIGYY[N];
double OLDDELSIGXX[N],OLDDELSIGXY[N],OLDDELSIGYY[N];
double AVGSIGXX[N],AVGSIGXY[N],AVGSIGYY[N];
double AVGSIGXX2[N],AVGSIGXY2[N],AVGSIGYY2[N];
double EINT, WP;
double DAMAGE[N][NCOUNT], wfgrad[N][5], EPSPART[N];
int PARTSUM[N], NEIGH[N][NCOUNT], NDAMAGE, BCOUNT[N],BC[N];
int NEARESTNEIGH[N][NCOUNT];
double EXTSIG = 0.0, EXTSIGCOUNTER=0;
double t, NLIFE;

FILE *fp;
FILE *fp5;

void initialize();
void derivatives(int tno, int mt);
void kernel(double wf[], double dist );
void Update(double dt,int tno, int mt);
void callprint(int loopcounter);
void plasticity();
void viscosity(double visc[], int i, int j, double xij, double yij, double vxij, double vyij, double dist, double wf[], double wf_grad, double wfdx, double wfdy);
void artificial_pressure(double Pi, double Pj, double rho2i, double rho2j, double wf[], double xij, double yij, double dist, int i, int j, double vxij, double vyij, double wf_grad, double wfdx, double wfdy);
void monacorr(int i, int j, double wf[], double vxij, double vyij, double wf_grad, double wfdx, double wfdy);
void basicsph(int i,int j,double wf[], double xij, double yij, double vxij, double vyij, double rho2i, double rho2j, double dist, double sigxxi, double sigyyi, double sigxyi, double sigxxj, double sigyyj, double sigxyj, double drhobar, double wf_grad, double wfdx, double wfdy);
double calcdamagestress(double xij, double yij, double dist, int ind1, int ind2);
double calcdamagestrain(double xij, double yij, double dist, int ind1, int ind2);
void generateneighbor();
void stressbc(int i, int j, double wfdx, double wfdy);
void computemaxminstress(double t);
void breakneighbor();
void breakneighbornew();

void gradcalc(int tno, int mt);
void integrate1(int tno, int mt);
void update1(int tno, int mt);
void update2(int tno, int mt);

int main()  {
	
	omp_set_num_threads(THREADS);
	
    t=0.00;
    int i, loopcounter=0,loadcounter=0;
    double tfactor;
    fp = fopen("Trajectory3.lammpstrj","w");
    fp5 = fopen("Energy.output","w");
	clock_t temp_time;
    temp_time = clock();
    
    initialize();
    generateneighbor();

while(loadcounter<1)	{
	tfactor = 1.0;
	if(loadcounter==0)	tfactor = 20.0;
	//EXTSIG = 0.0;
    while(t<tfactor*TPERIOD)    {
		EXTSIG += 0.1*SIGAPP;
		
		if(EXTSIG >= 0.1*SIGAPP) {
			EXTSIG = 0.1*SIGAPP;
			EXTSIGCOUNTER++;
    		for(i=0;i<N;i++)  AVGSIGXX2[i] += (SXX[i]+P[i])*DT;
    		for(i=0;i<N;i++)  AVGSIGXY2[i] += (SXY[i])*DT;
    		for(i=0;i<N;i++)  AVGSIGYY2[i] += (SYY[i]+P[i])*DT;
        }
		
        #pragma omp parallel
		{
		integrate1(omp_get_thread_num(),omp_get_num_threads());
		gradcalc(omp_get_thread_num(),omp_get_num_threads());
		}
		#pragma omp parallel
		{
        derivatives(omp_get_thread_num(),omp_get_num_threads());
		}
		#pragma omp parallel
		{
        update1(omp_get_thread_num(),omp_get_num_threads());
		}
		
		#pragma omp parallel
		{
		gradcalc(omp_get_thread_num(),omp_get_num_threads());
		}
		
        #pragma omp parallel
        {
        derivatives(omp_get_thread_num(),omp_get_num_threads());
		}
		//Rate Computation is with respect to XN
        #pragma omp parallel
		{
        update2(omp_get_thread_num(),omp_get_num_threads());
		}//Update is with respect to XO;

        
		
        //plasticity();

        if(loopcounter%NPRINT==0) {
            double KE1 = 0.0, IE1 = 0.0, KE2 = 0.0, IE2 = 0.0;
            printf("Time(ms): %lf ", (double) DT*loopcounter);
            for(i=0; i<N; i++) {
                if(i< N-NB) {
                    KE1 += 0.5 * MASS[i] * ( VX[i]*VX[i] + VY[i]*VY[i] );
                    IE1 += MASS[i]* E[i];
                }
                if(i>= N-NB)    {
                    KE2 += 0.5 * MASS[i] * ( VX[i]*VX[i] + VY[i]*VY[i] );
                    IE2 += MASS[i]* E[i];
                }
            }
            fprintf(fp5,"%lf %lf %lf %lf %lf %lf \n",(double) DT*loopcounter,KE1,IE1,KE2,IE2,KE1+IE1+KE2+IE2+EINT);
            callprint(loopcounter);
        }
        t+=DT;
        loopcounter++;
    }
    //breakneighbor();
    //computemaxminstress(t);            
    //breakneighbornew();    
    t = 0.0;
	loadcounter++;
	loopcounter=0;
}
    fclose(fp5);
    
    temp_time = clock() - temp_time;
    double time_taken = ((double)temp_time)/CLOCKS_PER_SEC; // in seconds
    printf("Total Time: %lf \n", time_taken);
    
    return(0);
}

void integrate1(int tno, int mt){
	if(tno>=mt)return;
	double xij,yij,dist=0,hsq = H*H,wf[2];
	int trange = N/mt;
	int tmin = (tno)*trange;
	int tmax = (tno==(mt-1)) ? N : (tno+1)*trange;
    int i;
	
	for(i=tmin;i<tmax;i++)   {
        XO[i]   = X[i];     YO[i]   = Y[i];
        VXO[i]  = VX[i];   VYO[i]  = VY[i];
        RHOO[i] = RHO[i];   EO[i]   = E[i];
        SXXO[i] = SXX[i];   SXYO[i] = SXY[i];  SYYO[i] = SYY[i];
        STRAINXXO[i] = STRAINXX[i];   STRAINXYO[i] = STRAINXY[i];  STRAINYYO[i] = STRAINYY[i];
    }
}

void gradcalc(int tno, int mt){
	if(tno>=mt)return;
	double xij,yij,dist=0,hsq = H*H,wf[2];
	int trange = N/mt;
	int tmin = (tno)*trange;
	int tmax = (tno==(mt-1)) ? N : (tno+1)*trange;
    int i,j,k;
	
	for(i=tmin;i<tmax;i++)	{
    	for(j=0;j<5;j++)	{
    		wfgrad[i][j] = 0.0;
    	}
		P[i]        = -B*(RHO[i]/RHO0-1.0);
    }
    if(GRADCORR == 0)	{
    	for(i=tmin;i<tmax;i++)	{
    		wfgrad[i][0] = 1.0;
    		wfgrad[i][1] = 1.0;
    		wfgrad[i][2] = 0.0;
    		wfgrad[i][3] = 0.0;
    		wfgrad[i][4] = 1.0;    		    		    		    		
    	}
    }
    if(GRADCORR == 1)	{
    	for(i=tmin;i<tmax;i++)	{
    		//for(j=0;j<N;j++)	{
			for(k=0;k<NCOUNT;k++)	{
    			j = NEIGH[i][k];
    		//for(j=0;j<N;j++)	{
				if((i!=j) && (j>-10000))    {
					xij  = X[i] - X[j];
                	yij  = Y[i] - Y[j];
					dist = xij*xij + yij*yij;
                	if(dist<=hsq)   {
                		dist = sqrt(dist);
                    	kernel(wf, dist);
                    	wfgrad[i][0] += wf[0]*MASS[j]/RHO[j];
                    	wfgrad[i][1] += -1.0*xij*wf[1]*xij/dist*MASS[j]/RHO[j];
                    	wfgrad[i][2] += -1.0*yij*wf[1]*xij/dist*MASS[j]/RHO[j];
                    	wfgrad[i][3] += -1.0*xij*wf[1]*yij/dist*MASS[j]/RHO[j];
                    	wfgrad[i][4] += -1.0*yij*wf[1]*yij/dist*MASS[j]/RHO[j];	
                	}
				}
    		}
    	}
    }
	
}

void derivatives(int tno, int mt)  {
	
	if(tno>=mt)return;
	int trange = N/mt;
	int tmin = (tno)*trange;
	int tmax = (tno==(mt-1)) ? N : (tno+1)*trange;
	
	
    double xij, yij, vxij, vyij, dist, hsq = H*H;
    double rhobar,drhobar;
    double smxx, smxy, smyx, smyy, rho0rho;
    double wf[2], lambda=B - G, eta=G, trace;
    double sigxxi, sigxyi, sigyyi, sigxxj, sigxyj, sigyyj;
    double rho2i, rho2j;
    double term1, term12, term2;
    double visc[1], rxydot[N];
    int i,j,k;
    double tensterm = 0.0, tensgamma = 0.3, distinit, xinitij, yinitij, wfinit[2];
    double mata, matb, matc, matd, wfdx, wfdy, wf_grad, fij, det1, det2;

    for(i=tmin;i<tmax;i++)    {
        RHODOT[i]   = 0.0;
        VXDOT[i]    = VYDOT[i] = 0.0;
        DVXDX[i]    = DVXDY[i] = DVYDX[i] = DVYDY[i] = 0.0;
        VXBAR[i]    = 0.0;
        VYBAR[i]    = 0.0;
        FX[i]       = FY[i] = 0.0;
        EDOT[i]     = 0.0;
        rho0rho     = RHO0/RHO[i];
        //P[i]        = -6.0*M*RHO0*( pow(2.0-rho0rho,2*M-1) - pow(2.0-rho0rho,M-1) );
       // P[i]        = -B*(RHO[i]/RHO0-1.0);
        rxydot[i]   = 0.0;
        EINT        = 0.0;
    }
    
    //Kernel Computation for Gradient Correction
    /* for(i=0;i<N;i++)	{
    	for(j=0;j<5;j++)	{
    		wfgrad[i][j] = 0.0;
    	}
    }
    if(GRADCORR == 0)	{
    	for(i=0;i<N;i++)	{
    		wfgrad[i][0] = 1.0;
    		wfgrad[i][1] = 1.0;
    		wfgrad[i][2] = 0.0;
    		wfgrad[i][3] = 0.0;
    		wfgrad[i][4] = 1.0;    		    		    		    		
    	}
    } */
//Actual Computation
    for(i=tmin;i<tmax;i++)    {
        sigxxi = SXX[i] + P[i];
        sigyyi = SYY[i] + P[i];
        sigxyi = SXY[i];
        
        AVGSIGXX[i] = sigxxi*MASS[i]/RHO[i];
	    AVGSIGXY[i] = sigxyi*MASS[i]/RHO[i];
    	AVGSIGYY[i] = sigyyi*MASS[i]/RHO[i];      
        
        rho2i  = 1.0/(RHO[i]*RHO[i]);
    		for(k=0;k<NCOUNT;k++)	{
    			j = NEIGH[i][k];
    		//for(j=0;j<N;j++)	{
            if((i!=j) && (j>-10000))    {
                //For Plate-Plate Interaction
                    xij  = X[i] - X[j];
                    yij  = Y[i] - Y[j];
                    vxij = VX[i] - VX[j];
                    vyij = VY[i] - VY[j];

                    dist = xij*xij + yij*yij;
                    if(dist<=hsq)   {
                        dist = sqrt(dist);
                        mata = wfgrad[i][1];
                        matb = wfgrad[i][2];
                        matc = wfgrad[i][3];
                        matd = wfgrad[i][4];

                        kernel(wf, dist);
                        wf_grad = wf[0] * MASS[j] / RHO[j] / wfgrad[i][0];
                        wfdx = 1.0;
                        wfdy = 1.0;
                        if ((GRADCORR == 1) && (GRADCORREN == 0) && (fabs(mata * matd - matb * matc)>1.0e-6)) {
                            wfdx = 1.0 / (mata * matd - matb * matc) * (wfgrad[i][4] * wf[1] * xij / dist - wfgrad[i][2] * wf[1] * yij / dist);
                            wfdy = 1.0 / (mata * matd - matb * matc) * (wfgrad[i][1] * wf[1] * yij / dist - wfgrad[i][3] * wf[1] * xij / dist);
                        }
                        if ((GRADCORR == 1) && (GRADCORREN == 1) && (fabs(mata * matd - matb * matc)>1.0e-6)) {
                            wfdx = 0.50 / (mata * matd - matb * matc) * (wfgrad[i][4] * wf[1] * xij / dist - wfgrad[i][2] * wf[1] * yij / dist);
                            wfdy = 0.50 / (mata * matd - matb * matc) * (wfgrad[i][1] * wf[1] * yij / dist - wfgrad[i][3] * wf[1] * xij / dist);

                            mata = wfgrad[j][1];
                            matb = wfgrad[j][2];
                            matc = wfgrad[j][3];
                            matd = wfgrad[j][4];
                            if(fabs(mata * matd - matb * matc)>1.0e-6)	{
                            	wfdx -= 0.50 / (mata * matd - matb * matc) * (-1.0 * wfgrad[j][4] * wf[1] * xij / dist + wfgrad[j][2] * wf[1] * yij / dist);
                            	wfdy -= 0.50 / (mata * matd - matb * matc) * (-1.0 * wfgrad[j][1] * wf[1] * yij / dist + wfgrad[j][3] * wf[1] * xij / dist);
                            }
                            if(fabs(mata * matd - matb * matc)<=1.0e-6)	{
                            	wfdx = 1.0;
                            	wfdy = 1.0;
                            }
                        }

                        fij = 1.0;
                        //if (PSEUDOSPRING == 2) fij = calcdamagestress(xij, yij, dist, i, j);
                        //if (PSEUDOSPRING == 1) fij = calcdamagestrain(xij, yij, dist, i, j);
                        wfdx = fij * wfdx;
                        wfdy = fij * wfdy;

                        rhobar      = RHO[j];
                        drhobar     = dist*rhobar;
                    	//RHODOT[i]   += wf[1]*MASS[j]*(vxij*xij + vyij*yij)/dist;
                    	RHODOT[i]   += MASS[j]*(vxij*wfdx + vyij*wfdy);
                    
                        sigxxj      = SXX[j] + P[j];
                        sigyyj      = SYY[j] + P[j];
                        sigxyj      = SXY[j];
                        rho2j       = 1.0/(RHO[j]*RHO[j]);

                        if(MONA_CORR == 1)  monacorr(i,j,wf,vxij,vyij,wf_grad,wfdx,wfdy);
                        if(JAUMANN == 1) rxydot[i] += -0.5 * MASS[j]/RHO[j]*(vxij*wfdy - vyij*wfdx);
                        
                        basicsph(i,j,wf,xij,yij,vxij,vyij,rho2i,rho2j,dist,sigxxi,sigyyi,sigxyi,sigxxj,sigyyj,sigxyj,drhobar,wf_grad,wfdx,wfdy);
                        
                        if(ART_VISCOSITY == 1)  viscosity(visc,i,j,xij,yij,vxij,vyij,dist,wf,wf_grad,wfdx,wfdy);
                        if(TENSILE == 1) artificial_pressure(P[i],P[j],rho2i,rho2j,wf, xij, yij, dist, i, j, vxij, vyij,wf_grad,wfdx,wfdy);
                        if(SIGAPP > 0.0)	{
                        	//EXTSIG = SIGAPP + SIGAMP*sin(2.0*PI/TPERIOD*t);
                            stressbc(i,j, wfdx, wfdy);                        	
                        }
                        
                        AVGSIGXX[i] += sigxxj*MASS[j]/RHO[j]*wf[0];
	        			AVGSIGXY[i] += sigxyj*MASS[j]/RHO[j]*wf[0];
    	    			AVGSIGYY[i] += sigyyj*MASS[j]/RHO[j]*wf[0];

                    }
            }
        }
        trace       = -1.0/3.0*(DVXDX[i] + DVYDY[i]);
        SXXDOT[i]   = 2.0*G*(DVXDX[i] + trace) + 2.0*SXY[i]*rxydot[i];
        SYYDOT[i]   = 2.0*G*(DVYDY[i] + trace) - 2.0*SXY[i]*rxydot[i];
        SXYDOT[i]   = 1.0*G*(DVXDY[i] + DVYDX[i]) - rxydot[i]*(SXX[i] - SYY[i]);
        XDOT[i] = VX[i] + VXBAR[i];
        YDOT[i] = VY[i] + VYBAR[i];
        EDOT[i] = -0.5*EDOT[i];
        STRAINXXDOT[i] = DVXDX[i];
        STRAINYYDOT[i] = DVYDY[i];
        STRAINXYDOT[i] = 0.5 * (DVXDY[i] + DVYDX[i]);
    }
}

void basicsph(int i,int j,double wf[], double xij, double yij, double vxij, double vyij, double rho2i, double rho2j, double dist, double sigxxi, double sigyyi, double sigxyi, double sigxxj, double sigyyj, double sigxyj, double drhobar, double wf_grad, double wfdx, double wfdy)   {
    double smxx, smyx, smxy, smyy;
	double rhobar = drhobar/dist;
	
    smxx        = -wfdx*vxij/rhobar;
    smxy		= -wfdy*vxij/rhobar;
    smyx        = -wfdx*vyij/rhobar;
    smyy        = -wfdy*vyij/rhobar;
    
    DVXDX[i]    += MASS[j]*smxx;
    DVXDY[i]    += MASS[j]*smxy;
    DVYDX[i]    += MASS[j]*smyx;
    DVYDY[i]    += MASS[j]*smyy;

    VXDOT[i]    += MASS[j]*(sigxxi*rho2i + sigxxj*rho2j)*wfdx + MASS[j]*(sigxyi*rho2i + sigxyj*rho2j)*wfdy;
    VYDOT[i]    += MASS[j]*(sigyyi*rho2i + sigyyj*rho2j)*wfdy + MASS[j]*(sigxyi*rho2i + sigxyj*rho2j)*wfdx;
    
    EDOT[i]     += MASS[j]*(vxij*(sigxxi*rho2i + sigxxj*rho2j)*wfdx + vxij*(sigxyi*rho2i + sigxyj*rho2j)*wfdy);
    EDOT[i]     += MASS[j]*(vyij*(sigyyi*rho2i + sigyyj*rho2j)*wfdy + vyij*(sigxyi*rho2i + sigxyj*rho2j)*wfdx);
}

void monacorr(int i, int j, double wf[], double vxij, double vyij, double wf_grad, double wfdx, double wfdy)  {
    VXBAR[i]    -= 0.50*MASS[j]*wf[0]*vxij/(RHO[i] + RHO[j]);
    VYBAR[i]    -= 0.50*MASS[j]*wf[0]*vyij/(RHO[i] + RHO[j]);
}

void artificial_pressure(double Pi, double Pj, double rho2i, double rho2j, double wf[], double xij, double yij, double dist, int i, int j, double vxij, double vyij, double wf_grad, double wfdx, double wfdy) {
    double tensterm1=0.0, tensterm2=0.0, tensterm, wfinit[2];
    double tensgamma = 0.50;
    double temp;
    kernel(wfinit, DY);

    temp = wf[0]/wfinit[0];
    temp = temp*temp;
    temp = temp*temp;
    //temp = temp*temp;
    
    if(Pi > 0.0) tensterm1 = Pi*rho2i;
    if(Pj > 0.0) tensterm2 = Pj*rho2j;
    tensterm = tensgamma*(tensterm1 + tensterm2)*temp;
    VXDOT[i]    += -1.0*MASS[j]*tensterm*wfdx;
    VYDOT[i]    += -1.0*MASS[j]*tensterm*wfdy;
    EDOT[i]     += -1.0*MASS[j]*vxij*tensterm*wfdx;
    EDOT[i]     += -1.0*MASS[j]*vyij*tensterm*wfdy;
}

void kernel(double wf[], double dist )   {
    double h = H;
    double q = dist / h;
    double par = 5.0 / (3.14159265359 * h * h);

    wf[0] = par * (1.0 + 3.0 * q) * (1.0 - q) * (1.0 - q) * (1.0 - q);
    wf[1] = -par * 12 / h * q * (1 - q) * (1 - q);
}

void update1(int tno, int mt){
	if(tno>=mt)return;
	double xij,yij,dist=0,hsq = H*H,wf[2];
	int trange = N/mt;
	int tmin = (tno)*trange;
	int tmax = (tno==(mt-1)) ? N : (tno+1)*trange;
	
    int i;
	
	for ( i = tmin;i<tmax;i++){
		if(MONA_CORR2 == 1)     {
            VX[i]   = VX[i] + VXBAR[i];
            VY[i]   = VY[i] + VYBAR[i];
        }
	}
    Update(DT,tno,mt);
	
	for(i=tmin;i<tmax;i++)    {
        X[i]   = XN[i];     Y[i]   = YN[i];
        VX[i]  = VXN[i];    VY[i]  = VYN[i];
        RHO[i] = RHON[i];   E[i]   = EN[i];
        SXX[i] = SXXN[i];   SXY[i] = SXYN[i];  SYY[i] = SYYN[i];
        STRAINXX[i] = STRAINXXN[i];   STRAINXY[i] = STRAINXYN[i];  STRAINYY[i] = STRAINYYN[i];
    }
}

void update2(int tno, int mt){
	if(tno>=mt)return;
	double xij,yij,dist=0,hsq = H*H,wf[2];
	int trange = N/mt;
	int tmin = (tno)*trange;
	int tmax = (tno==(mt-1)) ? N : (tno+1)*trange;
	
    int i;
	for ( i = tmin;i<tmax;i++){
		
	    if(MONA_CORR2 == 1)     {
            VX[i]   = VX[i] + VXBAR[i];
            VY[i]   = VY[i] + VYBAR[i];
        }
	}
	
	
	
	for(i=tmin;i<tmax;i++)     {
        X[i]   = XO[i];     Y[i]   = YO[i];
        VX[i]  = VXO[i];    VY[i]  = VYO[i];
        RHO[i] = RHOO[i];   E[i]   = EO[i];
        SXX[i] = SXXO[i];   SXY[i] = SXYO[i];  SYY[i] = SYYO[i];
        STRAINXX[i] = STRAINXXO[i];   STRAINXY[i] = STRAINXYO[i];  STRAINYY[i] = STRAINYYO[i];
    }
	
	Update(DT,tno,mt);
	
	for(i=tmin;i<tmax;i++)   {
        X[i]        = 2.0 * XN[i] - XO[i];
        Y[i]        = 2.0 * YN[i] - YO[i];
        VX[i]       = 2.0 * VXN[i] - VXO[i];
        VY[i]       = 2.0 * VYN[i] - VYO[i];
        SXX[i]      = 2.0 * SXXN[i] - SXXO[i];
        SXY[i]      = 2.0 * SXYN[i] - SXYO[i];
        SYY[i]      = 2.0 * SYYN[i] - SYYO[i];
        RHO[i]      = 2.0 * RHON[i] - RHOO[i];
        E[i]        = 2.0 * EN[i] - EO[i];
        STRAINXX[i] = 2.0 * STRAINXXN[i] - STRAINXXO[i];
        STRAINXY[i] = 2.0 * STRAINXYN[i] - STRAINXYO[i];
        STRAINYY[i] = 2.0 * STRAINYYN[i] - STRAINYYO[i];
    }
}

void Update(double dt,int tno, int mt)  {
	
	if(tno>=mt)return;
	double xij,yij,dist=0,hsq = H*H,wf[2];
	int trange = N/mt;
	int tmin = (tno)*trange;
	int tmax = (tno==(mt-1)) ? N : (tno+1)*trange;
	
    int i;
/*
    for(i=0;i<NINIT;i++)   {
        RHON[i] = RHO[i] + 0.5*dt*RHODOT[i];
        EN[i]   = E[i] + 0.5*dt*EDOT[i];
        XN[i]   = X[i] - 0.5*VBALL*dt;
        YN[i]   = Y[i];
        VXN[i]  = -1.0*VBALL;
        VYN[i]  = VY[i];
        SXXN[i] = SXX[i] + 0.5*dt*SXXDOT[i];
        SXYN[i] = SXY[i] + 0.5*dt*SXYDOT[i];
        SYYN[i] = SYY[i] + 0.5*dt*SYYDOT[i];
        STRAINXXN[i] = STRAINXX[i] + 0.5 * dt * STRAINXXDOT[i];
        STRAINXYN[i] = STRAINXY[i] + 0.5 * dt * STRAINXYDOT[i];
        STRAINYYN[i] = STRAINYY[i] + 0.5 * dt * STRAINYYDOT[i];
    }

    for(i=NINIT;i<N-NB;i++)   {
        RHON[i] = RHO[i] + 0.5*dt*RHODOT[i];
        EN[i]   = E[i] + 0.5*dt*EDOT[i];
        XN[i]   = X[i] + 0.5*dt*XDOT[i];
        YN[i]   = Y[i] + 0.5*dt*YDOT[i];
        VXN[i]  = VX[i] + 0.5*dt*VXDOT[i];
        VYN[i]  = VY[i] + 0.5*dt*VYDOT[i];
        SXXN[i] = SXX[i] + 0.5*dt*SXXDOT[i];
        SXYN[i] = SXY[i] + 0.5*dt*SXYDOT[i];
        SYYN[i] = SYY[i] + 0.5*dt*SYYDOT[i];
        STRAINXXN[i] = STRAINXX[i] + 0.5 * dt * STRAINXXDOT[i];
        STRAINXYN[i] = STRAINXY[i] + 0.5 * dt * STRAINXYDOT[i];
        STRAINYYN[i] = STRAINYY[i] + 0.5 * dt * STRAINYYDOT[i];
    }
    
    for(i=N-NB;i<N;i++)   {
        RHON[i] = RHO[i] + 0.5*dt*RHODOT[i];
        EN[i]   = E[i] + 0.5*dt*EDOT[i];
        XN[i]   = X[i] + 0.5*VBALL*dt;
        YN[i]   = Y[i];
        VXN[i]  = VBALL;
        VYN[i]  = VY[i];
        SXXN[i] = SXX[i] + 0.5*dt*SXXDOT[i];
        SXYN[i] = SXY[i] + 0.5*dt*SXYDOT[i];
        SYYN[i] = SYY[i] + 0.5*dt*SYYDOT[i];
        STRAINXXN[i] = STRAINXX[i] + 0.5 * dt * STRAINXXDOT[i];
        STRAINXYN[i] = STRAINXY[i] + 0.5 * dt * STRAINXYDOT[i];
        STRAINYYN[i] = STRAINYY[i] + 0.5 * dt * STRAINYYDOT[i];
    }    
    */
    for(i=tmin;i<tmax;i++)   {
        RHON[i] = RHO[i] + 0.5*dt*RHODOT[i];
        EN[i]   = E[i] + 0.5*dt*EDOT[i];
        XN[i]   = X[i] + 0.5*dt*XDOT[i];
        YN[i]   = Y[i] + 0.5*dt*YDOT[i];
        VXN[i]  = VX[i] + 0.5*dt*VXDOT[i];
        VYN[i]  = VY[i] + 0.5*dt*VYDOT[i];
        SXXN[i] = SXX[i] + 0.5*dt*SXXDOT[i];
        SXYN[i] = SXY[i] + 0.5*dt*SXYDOT[i];
        SYYN[i] = SYY[i] + 0.5*dt*SYYDOT[i];
        STRAINXXN[i] = STRAINXX[i] + 0.5 * dt * STRAINXXDOT[i];
        STRAINXYN[i] = STRAINXY[i] + 0.5 * dt * STRAINXYDOT[i];
        STRAINYYN[i] = STRAINYY[i] + 0.5 * dt * STRAINYYDOT[i];
    }
    /*
    for(i=N-NB;i<N;i++)   {
        RHON[i] = RHO[i] + 0.5*dt*RHODOT[i];
        EN[i]   = E[i] + 0.5*dt*EDOT[i];
        XN[i]   = X[i];
        YN[i]   = Y[i]+ 0.5*dt*YDOT[i];
        VXN[i]  = 0.0;
        VYN[i]  = VY[i] + 0.5*dt*VYDOT[i];
        SXXN[i] = SXX[i] + 0.5*dt*SXXDOT[i];
        SXYN[i] = SXY[i] + 0.5*dt*SXYDOT[i];
        SYYN[i] = SYY[i] + 0.5*dt*SYYDOT[i];
        STRAINXXN[i] = STRAINXX[i] + 0.5 * dt * STRAINXXDOT[i];
        STRAINXYN[i] = STRAINXY[i] + 0.5 * dt * STRAINXYDOT[i];
        STRAINYYN[i] = STRAINYY[i] + 0.5 * dt * STRAINYYDOT[i];
    }
    */
}

void plasticity()   {
    int i;
    double factor, sxx, sxy, syy, J2, yield = YIELD/sqrt(3.0), temp;
	
    for(i=0;i<N;i++)  {
        sxx = SXX[i];
        sxy = SXY[i];
        syy = SYY[i];
        J2 = 0.5*(sxx*sxx + 2.0*sxy*sxy + syy*syy);
        J2 = sqrt(J2);
        
        if(J2 > yield)	{
            factor = MIN(yield/J2,1.0);
            sxx = factor*sxx;
            sxy = factor*sxy;
            syy = factor*syy;
            temp = (1.0-factor)/(2.0*G);
            EPLXX[i] += temp*sxx;
            EPLXY[i] += temp*sxy;
            EPLYY[i] += temp*sxy;
            WP += temp*(sxx*sxx + 2.0*sxy*sxy + syy*syy)*MASS[i]/RHO[i];          
        }
        
        SXX[i] = sxx;
        SYY[i] = syy;
        SXY[i] = sxy;
    }
}

void callprint(int loop) {

    int i;
    double KE1 = 0.0,IE1 = 0.0, KE2 = 0.0, IE2 = 0.0, SumStressx = 0.0, SumStressy=0.0, SumStrainx = 0.0, SumStrainy = 0.0;
    //For trajectory visualization in LAMMPS
    fprintf(fp,"ITEM: TIMESTEP\n");
    fprintf(fp,"%d\n",loop);
    fprintf(fp,"ITEM: NUMBER OF ATOMS\n");
    fprintf(fp,"%d\n",N);
    fprintf(fp,"ITEM: BOX BOUNDS ss ss ss\n");
    fprintf(fp,"0.00000 1.0\n");
    fprintf(fp,"0.00000 1.0\n");
    fprintf(fp,"0.00000 0.00000\n");
    fprintf(fp,"ITEM: ATOMS id type xs ys zs sxx sxy syy Avgsxx Avgsxy Avgsyy BCOUNT\n");
    for(i=0; i<N; i++) {
        //fprintf(fp,"%d %d %lf %lf 0.000 %lf %lf %lf %lf %lf %lf \n", i, 0, X[i], Y[i],(SXX[i] +P[i])*MASS[i]/RHO[i],(SXY[i] +P[i])*MASS[i]/RHO[i],(SYY[i] +P[i])*MASS[i]/RHO[i], STRAINXX[i], STRAINXY[i], STRAINYY[i] );
        fprintf(fp,"%d %d %lf %lf 0.000 %lf %lf %lf %lf %lf %lf %d \n", i, 0, X[i], Y[i],(SXX[i]+P[i]),SXY[i],(SYY[i]+P[i]), (double) AVGSIGXX2[i]/(EXTSIGCOUNTER*DT), (double) AVGSIGXY2[i]/(EXTSIGCOUNTER*DT), (double) AVGSIGYY2[i]/(EXTSIGCOUNTER*DT), BCOUNT[i]);
        //fprintf(fp,"%d %d %lf %lf 0.000 %lf %lf %lf %lf %lf %lf \n", i, 0, X[i], Y[i],SXX[i],SXY[i],SYY[i], STRAINXX[i], STRAINXY[i], STRAINYY[i] );
        //Total Energy calculation
        if(i< N-NB) {
            KE1 += 0.5 * MASS[i] * ( VX[i]*VX[i] + VY[i]*VY[i] );
            IE1 += MASS[i]* E[i];
        }
        if(i>= N-NB)    {
            KE2 += 0.5 * MASS[i] * ( VX[i]*VX[i] + VY[i]*VY[i] );
            IE2 += MASS[i]* E[i];
        }
        if(PARTSUM[i] == 1)	{
        	SumStressx += (SXX[i] +P[i])*MASS[i]/RHO[i];
        	SumStressy += (SYY[i] +P[i])*MASS[i]/RHO[i];
            SumStrainx += STRAINXX[i];
            SumStrainy += STRAINYY[i];
        }
    }
    SumStressx = SumStressx/297.0;
    SumStressy = SumStressy/297.0;
    SumStrainx = SumStrainx/297.0;
    SumStrainy = SumStrainy/297.0;
    printf("KE1:%lf IE1:%lf KE2:%lf IE2:%lf WP:%lf Total Energy:%lf Interaction Energy: %lf Total Energy: %lf SumStressx: %lf SumStessy: %lf SumStrainx: %lf SumStrainy: %lf \n",KE1,IE1,KE2,IE2,WP,KE1+IE1+KE2+IE2, EINT, KE1+IE1+KE2+IE2+EINT, SumStressx, SumStressy,SumStrainx, SumStrainy);
}

void viscosity(double visc[], int i, int j, double xij, double yij, double vxij, double vyij, double dist, double wf[], double wf_grad, double wfdx, double wfdy)    {

    double delVdelR, muij, ci = sqrt(EMOD/RHO[i]), cj = sqrt(EMOD/RHO[j]);

    delVdelR    = xij*vxij + yij*vyij;
    muij        = (H*delVdelR)/(dist*dist + 0.01*H*H);
    visc[0]     = 0.0; //Force part
    if(delVdelR < 0.0) visc[0] = ( -0.5*(ci+cj)*muij*alpha + beta*muij*muij )/ ( 0.5*(RHO[i] + RHO[j]) );
    VXDOT[i]    += -1.0*MASS[j]*visc[0]*wfdx;
    VYDOT[i]    += -1.0*MASS[j]*visc[0]*wfdy;
    EDOT[i]     += -1.0*MASS[j]*vxij*visc[0]*wfdx;
    EDOT[i]     += -1.0*MASS[j]*vyij*visc[0]*wfdy;
}

void initialize()   {
    
    int i,j,count=0;
    double dy = DY, dx = sqrt(3.0)/2.0*dy, displacement = 0.5*DY;
    double xlim = (double) NY*dx, ylim = (double) NX*dy*0.30;
    double tot_momentum = (double) MASSB*VBALL*NB, tot_mass = 0.0;
    double temp;
    int tempint;

    FILE *fp2, *fp3;
    fp2 = fopen("Graphene_si_nano13.dat","r");
    fp3 = fopen("Initial.dat","w");
    
    for(i=0;i<N;i++)	{
    	fscanf(fp2,"%d %d %d %d %lf %lf %d",&tempint,&tempint,&tempint,&tempint,&X[i],&Y[i],&BC[i]);
    	fprintf(fp3,"%lf %lf \n",X[i],Y[i]);
    	VX[i]=VY[i]=0.0;
    	RHO[i] = RHO0;
    	E[i]=SXX[i]=SXY[i]=SYY[i]=0.0;
    	MASS[i] = MASSP;
    	tot_mass += MASS[i];
    	
    	PARTSUM[i] = 0;
    	if((X[i]>-0.90)&&(X[i]<0.90)) PARTSUM[i] = 1;    	
    }    
    fclose(fp2);
    fclose(fp3);
    
    WP = 0.0; //Initialization of work done during plastic process
    for(i=0;i<N;i++)	{
    	EPLXX[i] = 0.0;
    	EPLXY[i] = 0.0;
    	EPLYY[i] = 0.0;
    	SIGXXMAX[i] = SIGXYMAX[i] = SIGYYMAX[i] = 0.0;
    	SIGXXMIN[i] = SIGXYMIN[i] = SIGYYMIN[i] = 0.0;
    	AVGSIGXX[i] = AVGSIGXY[i] = AVGSIGYY[i] = 0.0;
    	AVGSIGXX2[i] = AVGSIGXY2[i] = AVGSIGYY2[i] = 0.0;
    	EPSPART[i] = 0.0;
    }
	EXTSIG = 0.0;
	EXTSIGCOUNTER = 0;
}

double calcdamagestress(double xij, double yij, double dist, int ind1, int ind2) {
    double temp1, temp2, temp3, temp4, maxtemp12, maxtemp34;
    double theta1, theta2, sigxx1, sigyy1, sigxx2, sigyy2, sigxy1, sigxy2;
    double cos2theta, sin2theta;
    double norm1, norm2, avgnorm;

    if (DAMAGE[ind1][ind2] < 1.0) {
        theta1 = acos(xij / dist);
        theta2 = acos(-1.0 * xij / dist);

        sigxx1 = SXX[ind1] + P[ind1];
        sigyy1 = SYY[ind1] + P[ind1];
        sigxy1 = SXY[ind1];
        sigxx2 = SXX[ind2] + P[ind2];
        sigyy2 = SYY[ind2] + P[ind2];
        sigxy2 = SXY[ind2];

        cos2theta = cos(2.0 * theta1);
        sin2theta = sin(2.0 * theta1);
        temp1 = 0.5 * (sigxx1 + sigyy1) + 0.5 * (sigxx1 - sigyy1) * cos2theta + sigxy1 * sin2theta;
        //temp2 = 0.5 * (sigxx1 + sigyy1) - 0.5 * (sigxx1 - sigyy1) * cos2theta1 - sigxy1 * sin2theta1;
        //norm1 = temp2;
        //if (fabs(temp1) > fabs(temp2)) norm1 = temp1;

        cos2theta = cos(2.0 * theta2);
        sin2theta = sin(2.0 * theta2);
        temp3 = 0.5 * (sigxx2 + sigyy2) + 0.5 * (sigxx2 - sigyy2) * cos2theta + sigxy2 * sin2theta;
        //temp4 = 0.5 * (sigxx2 + sigyy2) - 0.5 * (sigxx2 - sigyy2) * cos2theta2 - sigxy2 * sin2theta2;
        //norm2 = temp4;
        //if (fabs(temp3) > fabs(temp4)) norm2 = temp3;

        avgnorm = 0.5 * (temp1 + temp3);
        if (avgnorm >= YIELD) {
            DAMAGE[ind1][ind2] = 1.0;
            return(1.0 - DAMAGE[ind1][ind2]);
        }
        if (avgnorm < YIELD) {
            DAMAGE[ind1][ind2] = 0.0;
            return(1.0);
        }
    }
    if (DAMAGE[ind1][ind2] == 1.0) {
        return(0.0);
    }

}

double calcdamagestrain(double xij, double yij, double dist, int ind1, int ind2) {
    double temp1, temp2, temp3, temp4, maxtemp12, maxtemp34;
    double theta1, theta2, strxx1, stryy1, strxx2, stryy2, strxy1, strxy2;
    double cos2theta, sin2theta;
    double norm1, norm2, avgnorm;
    int i;
/*    
    if (DAMAGE[ind1][ind2] == 1.0) {
        return(0.0);
    }
    
    if (DAMAGE[ind1][ind2] < 1.0) {
        theta1 = acos(xij / dist);
        theta2 = acos(-1.0 * xij / dist);
        
        strxx1 = STRAINXX[ind1];
        stryy1 = STRAINYY[ind1];
        strxy1 = STRAINXY[ind1];
        strxx2 = STRAINXX[ind2];
        stryy2 = STRAINYY[ind2];
        strxy2 = STRAINXY[ind2];

        cos2theta = cos(2.0 * theta1);
        sin2theta = sin(2.0 * theta1);
        temp1 = 0.5 * (strxx1 + stryy1) + 0.5 * (strxx1 - stryy1) * cos2theta + strxy1 * sin2theta;

        cos2theta = cos(2.0 * theta2);
        sin2theta = sin(2.0 * theta2);
        temp3 = 0.5 * (strxx2 + stryy2) + 0.5 * (strxx2 - stryy2) * cos2theta + strxy2 * sin2theta;

        avgnorm = 0.5 * (temp1 + temp3);
        if (avgnorm + 0.5*(EPSPART[ind1]+EPSPART[ind2]) >= EPSMAX) {
            DAMAGE[ind1][ind2] = 1.0;
            NDAMAGE = 1;
            return(1.0 - DAMAGE[ind1][ind2]);
        }
        if (avgnorm + 0.5*(EPSPART[ind1]+EPSPART[ind2]) < EPSMAX) {
            DAMAGE[ind1][ind2] = 0.0;
            return(1.0);
        }
    }

*/
	if (DAMAGE[ind1][ind2] == 1.0) {
        return(0.0);
    }
    
	temp1 = (dist-DY + 0.5*(EPSPART[ind1]+EPSPART[ind2]) )/DY;
	if (temp1 >= EPSMAX) {
        DAMAGE[ind1][ind2] = 1.0;
        NDAMAGE = 1;
        return(1.0 - DAMAGE[ind1][ind2]);
    }
    if (temp1 < EPSMAX) {
        DAMAGE[ind1][ind2] = 0.0;
        return(1.0);
    }

}

void generateneighbor()	{
	int i, j, count;
	double hsq = H*H, xij, yij, dist;
	
	for(i=0;i<N;i++)	{
		for(j=0;j<NCOUNT;j++)	{
			NEIGH[i][j] = -10000;
		}
	}
	for(i=0;i<N;i++)	{
		count = 0;
		BCOUNT[i] = 0;		
		for(j=0;j<N;j++)	{
			if(i!=j)    {
				xij  = X[i] - X[j];
               	yij  = Y[i] - Y[j];
				dist = xij*xij + yij*yij;
                if(dist<=2.25*hsq)   {
                	NEIGH[i][count] = j;
                	NEARESTNEIGH[i][count] = -10000;
                	if(dist<=1.0*hsq) NEARESTNEIGH[i][count] = 1;
                	if(NEIGH[i][count] > -10000) BCOUNT[i]+=1;
                	count = count + 1;
                	/*
                	if((X[i]==0.008335495)&&(X[j]==0.008552001)&&(Y[i]<=-0.005) &&(Y[j]<=-0.005))	{
                		count = count - 1;
                		NEIGH[i][count] = -10000;
                	}
                	if((X[i]==0.008552001)&&(X[j]==0.008335495)&&(Y[i]<=-0.005) &&(Y[j]<=-0.005))	{
                		count = count - 1;
                		NEIGH[i][count] = -10000;
                	}
                	*/
                }
            }
		}
	}
}

void stressbc(int i, int j, double wfdx, double wfdy)	{
	
	/*
	if(BC[i] == 1)	{
		VXDOT[i] -= MASS[j]/(RHO[i]*RHO[j])*1.0*EXTSIG*wfdy;
		VYDOT[i] -= MASS[j]/(RHO[i]*RHO[j])*1.0*EXTSIG*wfdx;
	}
	*/
	
	
	if(i<NINIT)	VXDOT[i] -= MASS[j]/(RHO[i]*RHO[j])*1.0*EXTSIG*wfdx;
//	if(i<=NINIT)	VYDOT[i] -= MASS[j]/(RHO[i]*RHO[j])*1.0*EXTSIG*wfdx;	
	if(i>=N-NB)	VXDOT[i] -= MASS[j]/(RHO[i]*RHO[j])*1.0*EXTSIG*wfdx;
//	if(i>=N-NB)	VYDOT[i] -= MASS[j]/(RHO[i]*RHO[j])*1.0*EXTSIG*wfdx;
	
	//if((i>=NINIT)||(i<N-NB)) VXDOT[i] -= 0.0;
}

void computemaxminstress(double t)	{
	int i;
	double sigxx, sigxy, sigyy, princistressmax, princistressmin;
	
	for(i=0;i<N;i++)	{
		sigxx = SXX[i]+P[i];
		sigxy = SXY[i]+P[i];
		sigyy = SXY[i];
		princistressmax = 0.5*(sigxx+sigyy) + sqrt( 0.25*(sigxx-sigyy)*(sigxx-sigyy) + sigxy*sigxy );
		princistressmin = 0.5*(sigxx+sigyy) - sqrt( 0.25*(sigxx-sigyy)*(sigxx-sigyy) + sigxy*sigxy );
		
		SIGXXMAX[i] = MAXI(SIGXXMAX[i], sigxx);
		SIGXXMIN[i] = MIN(SIGXXMIN[i], sigxx);
		//SIGXYMAX[i] = MAXI(SIGXYMAX[i], SXY[i]);
		//SIGXYMIN[i] = MIN(SIGXYMIN[i], SXY[i]);	
		//SIGYYMAX[i] = MAXI(SIGYYMAX[i], SYY[i] + P[i]);
		//SIGYYMIN[i] = MIN(SIGYYMIN[i], SYY[i] + P[i]);
		DELSIGXX[i] = (SIGXXMAX[i]);
		//DELSIGXY[i] = (SIGXYMAX[i] - SIGXYMIN[i])*SQRTPIPHI;
		//DELSIGYY[i] = (SIGYYMAX[i] - SIGYYMIN[i])*SQRTPIPHI;
	}
}

void breakneighbor()	{
	int i, j, k, maxindex, maxindex2, tempint;
	double maxdelsigxx=0.0, temp, princistress, sigxx1, sigxy1, sigyy1, sigxx2, sigxy2, sigyy2;
	double theta1, theta2, temp1, temp3, avgnorm[NCOUNT],maxavgnorm=0.0;
	double xij,yij,dist,cos2theta,sin2theta;
	int ind1, ind2, nearestneigh;
	
	for(i=0;i<N;i++)	{
		temp = maxdelsigxx;
		maxdelsigxx = MAXI(maxdelsigxx,DELSIGXX[i]);
		if(temp!=maxdelsigxx)	maxindex = i;
	}
	//printf("%lf %d \n", maxdelsigxx, maxindex);
	//printf("%d ",maxindex);
	//for(tempint=0;tempint<NCOUNT;tempint++)	printf("%d ", NEIGH[maxindex][tempint]);
	//printf("\n");

	i = maxindex;
	if(maxdelsigxx >= 2.0*SIGAPP)	{
	for(k=0;k<NCOUNT;k++)	{
		j = NEIGH[i][k];
		nearestneigh = NEARESTNEIGH[i][k];
		avgnorm[k] = -10000.0;
		xij  = X[i] - X[j];
        yij  = Y[i] - Y[j];
		dist = sqrt(xij*xij + yij*yij);
		if((j>-10000)&&(nearestneigh > -10000))	{
			
			theta1 = acos(xij / dist);
        	theta2 = acos(-1.0 * xij / dist);

        	sigxx1 = SXX[i] + P[i];
        	sigyy1 = SYY[i] + P[i];
        	sigxy1 = SXY[i];
        	sigxx2 = SXX[j] + P[j];
        	sigyy2 = SYY[j] + P[j];
        	sigxy2 = SXY[j];

        	cos2theta = cos(2.0 * theta1);
        	sin2theta = sin(2.0 * theta1);
        	temp1 = 0.5 * (sigxx1 + sigyy1) + 0.5 * (sigxx1 - sigyy1) * cos2theta + sigxy1 * sin2theta;

        	cos2theta = cos(2.0 * theta2);
        	sin2theta = sin(2.0 * theta2);
        	temp3 = 0.5 * (sigxx2 + sigyy2) + 0.5 * (sigxx2 - sigyy2) * cos2theta + sigxy2 * sin2theta;
        	
        	/*
        	sigxx1 = SXX[i] + P[i];
        	sigyy1 = SYY[i] + P[i];
        	sigxy1 = SXY[i];
        	sigxx2 = SXX[j] + P[j];
        	sigyy2 = SYY[j] + P[j];
        	sigxy2 = SXY[j];     
        	
        	temp1 = 0.5*(sigxx1+sigyy1) + sqrt( 0.25*(sigxx1-sigyy1)*(sigxx1-sigyy1) + sigxy1*sigxy1 );
        	temp3 = 0.5*(sigxx2+sigyy2) + sqrt( 0.25*(sigxx2-sigyy2)*(sigxx2-sigyy2) + sigxy2*sigxy2 );        	
        	*/
        	avgnorm[k] = 0.5*(temp1+temp3);
        	//printf("%d %d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf \n", i, j, k, avgnorm[k], temp1, temp3, theta1, theta2, SXX[i], P[i], SXX[j],P[j]);
        	temp = maxavgnorm;
        	maxavgnorm = MAXI(maxavgnorm,avgnorm[k]);
        	if(temp!=maxavgnorm)	maxindex = k;
        	
        	/*
        	temp1 = (dist-DY)/DY;
        	avgnorm[k] = temp1;
        	temp = maxavgnorm;
        	maxavgnorm = MAXI(maxavgnorm,avgnorm[k]);
        	if(temp!=maxavgnorm)	maxindex = k;
        	*/
        }
	}
	//printf("%d %d %d \n", i, maxindex, NEIGH[i][maxindex]);
	//for(tempint=0;tempint<NCOUNT;tempint++)	printf("%d ", NEIGH[i][tempint]);
	//printf("\n");

	ind1 = i;
	ind2 = NEIGH[ind1][maxindex];
	NEIGH[ind1][maxindex] = -10000;
	//printf("%d %d %d \n", ind1, maxindex, NEIGH[i][maxindex]);
	for(j=0;j<NCOUNT;j++)	{
		if(NEIGH[ind2][j]==ind1)	{
			NEIGH[ind2][j]=-10000;
			maxindex2 = j;
		}
	}
	for(tempint=0;tempint<NCOUNT;tempint++)	printf("%d ", NEIGH[ind1][tempint]);	
	printf("%d %d %d \n", ind2, maxindex2, NEIGH[ind2][maxindex2]);
	NEIGH[ind2][maxindex2] = -10000;	
	printf("%d %d %d \n", ind2, maxindex2, NEIGH[ind2][maxindex2]);
	BCOUNT[ind1] -= 1;
	BCOUNT[ind2] -= 1;	
	
	
	//initialize();
	for(i=0;i<N;i++)	{
		E[i]=SXX[i]=SXY[i]=SYY[i]=0.0;
		VX[i]=VY[i]=0.0;
    	RHO[i] = RHO0;
	}
	}
	for(i=0;i<N;i++)	{
		SIGXXMAX[i] = 0.0;
		SIGXXMIN[i] = 0.0;
		DELSIGXX[i] = 0.0;
	}
	
}

void breakneighbornew()	{
	int i, j, k, l, m, maxindex, maxindex2, tempint;
	double maxdelsigxx=0.0, maxavgnorm = 0.0, avgnorm[NCOUNT], temp;
	double temp1, temp3;
	double xij,yij,dist,cos2theta,sin2theta;
	int ind1[N], ind2[N], nearestneigh, breakcount=0;
	double theta1, theta2, sigxx1, sigxy1, sigyy1, sigxx2, sigxy2, sigyy2;
	double avgsigxx[N],avgsigxy[N],avgsigyy[N],princistress[N];
	
	for(i=0;i<N;i++)	ind1[i] = -10000;
	for(i=0;i<N;i++)	ind2[i] = -10000;	
	
	//Calculates and Stores the average and maximum principal stress
	for(i=0;i<N;i++)	{
		avgsigxx[i] = (double) AVGSIGXX2[i]/(EXTSIGCOUNTER*DT);
		avgsigxy[i] = (double) AVGSIGXY2[i]/(EXTSIGCOUNTER*DT);
		avgsigyy[i] = (double) AVGSIGYY2[i]/(EXTSIGCOUNTER*DT);
		princistress[i] = 0.5*(avgsigxx[i]+avgsigyy[i]) + sqrt( 0.25*(avgsigxx[i]-avgsigyy[i])*(avgsigxx[i]-avgsigyy[i]) + avgsigxy[i]*avgsigxy[i]);
		temp = princistress[i];
		maxdelsigxx = MAXI(maxdelsigxx,temp);
	}
		
	//Computes the indices for maximum maxdelsigxx now. Taking into account the symmetricity. Stores in the variable ind1
	if(maxdelsigxx >= 0.3*SIGAPP)	{
		for(i=0;i<N;i++)	{
			temp = princistress[i];
			if(fabs(maxdelsigxx - temp)/maxdelsigxx <= 0.01)	{
				ind1[breakcount] = i;
				breakcount++;
			}
		}
	}
	
	if(maxdelsigxx >= 0.3*SIGAPP)	{
		for(k=0;k<breakcount;k++)	{
			printf("BreakCOUNT IS: %d Breakstress is: %lf \n", breakcount,maxdelsigxx);
			i = ind1[k];
			printf("BreakPART IS: %d \n", i);
			maxavgnorm = 0;
			for(l=0;l<NCOUNT;l++)	{
				j = NEIGH[i][l];
				nearestneigh = NEARESTNEIGH[i][l];
				avgnorm[l] = -10000.0;
				xij  = X[i] - X[j];
        		yij  = Y[i] - Y[j];
				dist = sqrt(xij*xij + yij*yij);
				if((j>-10000)&&(nearestneigh > -10000))	{
				
					theta1 = acos(xij / dist);
        			theta2 = acos(-1.0 * xij / dist);

        			sigxx1 = avgsigxx[i];
        			sigyy1 = avgsigyy[i];
        			sigxy1 = avgsigxy[i];
        			sigxx2 = avgsigxx[j];
        			sigyy2 = avgsigyy[j];
        			sigxy2 = avgsigxy[j];

        			cos2theta = cos(2.0 * theta1);
        			sin2theta = sin(2.0 * theta1);
        			temp1 = 0.5 * (sigxx1 + sigyy1) + 0.5 * (sigxx1 - sigyy1) * cos2theta + sigxy1 * sin2theta;

        			cos2theta = cos(2.0 * theta2);
        			sin2theta = sin(2.0 * theta2);
        			temp3 = 0.5 * (sigxx2 + sigyy2) + 0.5 * (sigxx2 - sigyy2) * cos2theta + sigxy2 * sin2theta;
        	
        			temp1 = 0.5*(temp1+temp3);
        			avgnorm[l] = fabs(temp1);
        			temp = maxavgnorm;
        			maxavgnorm = MAXI(temp,avgnorm[l]);
//        			if(temp!=maxavgnorm)	maxindex = l;
//        			if(temp==maxavgnorm)	breakcount2++;
					printf("%d %d %3.8lf %3.8lf %3.8lf \n", i,j,avgnorm[l], maxavgnorm, fabs(avgnorm[l]-maxavgnorm)/maxavgnorm);
        		}
			}
			
			for(l=0;l<NCOUNT;l++)	{
				if( fabs(avgnorm[l] - maxavgnorm)/maxavgnorm <= 0.01 )	{
					maxindex = l;
					ind2[k] = NEIGH[i][maxindex];
					printf("NEIGH[i][maxindex]=NEIGH[%d][%d]=%d \n",i,maxindex, NEIGH[i][maxindex]);
					NEIGH[i][maxindex] = -10000;
					BCOUNT[i] -= 1;			
					j = ind2[k];
					for(m=0;m<NCOUNT;m++)	{
						if(NEIGH[j][m]==i)	NEIGH[j][m]=-10000;
					}
					BCOUNT[j] -= 1;
				}			
			}		
		}
		/*
		for(i=0;i<N;i++)	{
			E[i]=SXX[i]=SXY[i]=SYY[i]=0.0;
			VX[i]=VY[i]=0.0;
    		RHO[i] = RHO0;
		}
		*/
		initialize();	

	}
	
}





