#include <iomanip>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

#ifndef F_PI
#define F_PI (float)M_PI
#endif

// print debugging messages?
#ifndef DEBUG
#define DEBUG false
#endif

// setting the number of threads to use:
// (this a default value -- it can also be set from the outside by your script)
#ifndef NUMT
#define NUMT 2
#endif

// setting the number of trials in the monte carlo simulation:
// (this a default value -- it can also be set from the outside by your script)
#ifndef NUMTRIALS
#define NUMTRIALS 50000
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES 30
#endif
using std::cout;
using std::endl;

int NowYear = 2025;
int NowMonth = 0;
float NowPrecip;	   // inches of rain per month
float NowTemp;		   // temperature this month
float NowHeight = 5.0; // grain height in inches
int NowNumDeer = 2;	   // number of deer

unsigned int seed = 0;

const float GRAIN_GROWS_PER_MONTH = 12.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0;
const float AVG_PRECIP_PER_MONTH = 7.0;
const float AMP_PRECIP_PER_MONTH = 6.0;
const float RANDOM_PRECIP = 2.0;
const float AVG_TEMP = 60.0;
const float AMP_TEMP = 20.0;
const float RANDOM_TEMP = 10.0;
const float MIDTEMP = 40.0;
const float MIDPRECIP = 10.0;
int FireEvent = 0; // 0 = no fire, 1-20 = fire happened

omp_lock_t Lock;
volatile int NumInThreadTeam;
volatile int NumAtBarrier;
volatile int NumGone;

void InitBarrier(int n)
{
	NumInThreadTeam = n;
	NumAtBarrier = 0;
	omp_init_lock(&Lock);
}

void WaitBarrier()
{
	omp_set_lock(&Lock);
	{
		NumAtBarrier++;
		if (NumAtBarrier == NumInThreadTeam)
		{
			NumGone = 0;
			NumAtBarrier = 0;
			while (NumGone != NumInThreadTeam - 1)
				;
			omp_unset_lock(&Lock);
			return;
		}
	}
	omp_unset_lock(&Lock);

	while (NumAtBarrier != 0)
		;

#pragma omp atomic
	NumGone++;
}

float Ranf(unsigned int *seedp, float low, float high)
{
	float r = (float)rand_r(seedp); // [ 0 - RAND_MAX ]
	return (low + r * (high - low) / (float)RAND_MAX);
}

float SQR(float x)
{
	return x * x;
}

void Grain()
{
	while (NowYear < 2031)
	{
		float tempFactor = exp(-SQR((NowTemp - MIDTEMP) / 10.0));
		float precipFactor = exp(-SQR((NowPrecip - MIDPRECIP) / 10.0));

		float nextHeight = NowHeight;
		nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
		nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
		if (nextHeight < 0.)
			nextHeight = 0.;

		WaitBarrier();
		NowHeight = nextHeight;
		WaitBarrier();
		WaitBarrier();
	}
}

void Deer()
{
	while (NowYear < 2031)
	{
		int nextNumDeer = NowNumDeer;
		int carryingCapacity = (int)(NowHeight);

		if (nextNumDeer < carryingCapacity)
			nextNumDeer++;
		else if (nextNumDeer > carryingCapacity)
			nextNumDeer--;

		if (nextNumDeer < 0)
			nextNumDeer = 0;

		WaitBarrier();
		NowNumDeer = nextNumDeer;
		WaitBarrier();
		WaitBarrier();
	}
}

void Watcher()
{
	FILE *fp = fopen("output.csv", "w");
	fprintf(fp, "Month,TempC,PrecipCM,HeightCM,Deer,FireEvent\n");
	while (NowYear < 2031)
	{
		WaitBarrier();
		WaitBarrier();

		// Convert to metric units for output
		float tempC = (5. / 9.) * (NowTemp - 32.);
		float precipCM = NowPrecip * 2.54;
		float heightCM = NowHeight * 2.54;

		fprintf(fp, "%d,%.2f,%.2f,%.2f,%d,%d\n",
				(NowYear - 2025) * 12 + NowMonth,
				tempC,
				precipCM,
				heightCM,
				NowNumDeer,
				FireEvent);

		// update time
		NowMonth++;
		if (NowMonth == 12)
		{
			NowMonth = 0;
			NowYear++;
		}

		// update temp & precip
		float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);
		float temp = AVG_TEMP - AMP_TEMP * cos(ang);
		NowTemp = temp + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);
		float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang);
		NowPrecip = precip + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
		if (NowPrecip < 0.)
			NowPrecip = 0.;

		WaitBarrier();
	}
	fclose(fp);
}

void MyAgent()
{
	while (NowYear < 2031)
	{
		bool fire = false;

		if (NowMonth >= 5 && NowMonth <= 8 && NowTemp > 70 && NowPrecip < 5)
			fire = true;

		WaitBarrier();
		if (fire)
		{
			NowHeight *= 0.5;
			NowNumDeer -= 2;
			if (NowNumDeer < 0)
				NowNumDeer = 0;
				FireEvent = (int)Ranf(&seed, 5, 21); // random intensity between 5 and 20
		}
		else
		{
			FireEvent = 0;
		}
		WaitBarrier();
		WaitBarrier();
	}
}

// main program:
int main(int argc, char *argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "No OpenMP support!\n");
	return 1;
#endif

	omp_set_num_threads(4);
	InitBarrier(4);

	// set initial NowTemp, NowPrecip
	float ang = (30. * (float)NowMonth + 15.) * (M_PI / 180.);
	NowTemp = AVG_TEMP - AMP_TEMP * cos(ang) + Ranf(&seed, -RANDOM_TEMP, RANDOM_TEMP);
	NowPrecip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin(ang) + Ranf(&seed, -RANDOM_PRECIP, RANDOM_PRECIP);
	if (NowPrecip < 0.)
		NowPrecip = 0.;

#pragma omp parallel sections
	{
#pragma omp section
		Grain();

#pragma omp section
		Deer();

#pragma omp section
		Watcher();

#pragma omp section
		MyAgent();
	}
	return 0;
}