#include <stdlib.h>
#include <stdio.h>
#include "apsp.h"

extern "C"
void computeGoldCol( float*, const float*,  int);

inline float min(float a, float b)
{
	return (a < b? a: b);
}

inline float combine(float a,float b)
{
	if(a == FLOATINF || b == FLOATINF)
		return FLOATINF;
	else
		return a+b;
}

void
computeGoldCol(float * C, const float * A, int hA)
{
	//first copy A to C as result
    for (int j = 0; j < hA; ++j)
	for (int i = 0; i < hA; ++i)
		C[j*hA + i] = A[j*hA + i];

    for (int k = 0; k < hA; ++k)
	for (int j = 0; j < hA; ++j)
		for (int i = 0; i < hA; ++i)
			C[j*hA + i] = min (C[j*hA + i], combine(C[k*hA + i],C[j*hA + k]));
}
