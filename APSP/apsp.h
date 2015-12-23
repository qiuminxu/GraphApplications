#ifndef _APSP_H_
#define _APSP_H_

#include <float.h>


// IMPORTANT !!! NEVER MAKE BLOCK_SIZE > 16, more than 512 threads are not allowed !!! 
// It won't give you a warning, but run incorrectly

//#define VERIFY 1
#define FLOATINF FLT_MAX

// Parameters for simple GEMM kernel
#define BLOCK_SIZE 16	  	// each block is of size (BLOCK_SIZE x BLOCK_SIZE)
#define FAST_GEMM 64		// for matrices of dimensions > (FAST_GEMM x FAST_GEMM), execute Volkov's code

// Parameters for Volkov's kernel
#define UNROLL 4		// inner blocking dimension
#define ABLOCK 16
#define BBLOCK 64   		// multiplications are between (ABLOCK x UNROLL for A) and (UNROLL  x BBLOCK for B)

#define WA 4096			// matrix dimension (problem size), should be = BLOCK_SIZE * BLOCK_DIM 

#endif
