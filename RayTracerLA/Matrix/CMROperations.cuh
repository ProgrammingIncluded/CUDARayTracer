#ifndef CMROPERATIONS_H
#define CMROPERATIONS_H

#include "CUDADef.h"

/* Cuda Matrix Related Operations */
/**
* Adds two CUDAMR objects together given their values.
*/
CUDA_MFUNCTION void addCUDAMR(float* mA, float* mB, int vectorSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// Safety if statement.
	if (index < vectorSize)
	{
		mA[index] += mB[index];
	}
}

/**
* Subtracts two CUDAMR objects together given their values.
*/
CUDA_MFUNCTION void subCUDAMR(float* mA, float* mB, int vectorSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// Safety if statement.
	if (index < vectorSize)
	{
		mA[index] -= mB[index];
	}
}

#endif // CMROPERATIONS_H