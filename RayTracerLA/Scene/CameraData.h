#ifndef CAMERADATA_H
#define CAMERADATA_H

#include <vector_types.h>

#include "Matrix/MatrixN.cuh"
#include "Matrix/VectN.cuh"

// Simple structure to store info related to device.
// Used by CUDA calls to prevent massive class given to Kernel.
struct CameraData
{
	//mat::MatrixN vpMatrix;	
	//mat::VectN origin;
	float3 ViewToWorldMatrixR0;
	float3 ViewToWorldMatrixR1;
	float3 ViewToWorldMatrixR2;

	float3 Origin;


	float tanFovXDiv2;
	float tanFovYDiv2;
};

#endif // CAMERADATA_H