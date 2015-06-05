#include "CUDAMR.cuh"

/* Cuda Matrix Related Operations */

/**
* Adds two CUDAMR objects together given their values.
*/


/**
* Subtracts two CUDAMR objects together given their values.
*/
__global__ void subCUDAMR(float* mA, float* mB, uint vectorSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// Safety if statement.
	if (index < vectorSize)
	{
		mA[index] -= mB[index];
	}
}

/**
* Multiplies one matrix by another matrix. Assumes square.
*/
__global__ void multCUDAMR(float* mA, float* mB, float* result, uint matrixDim)
{
	float value = 0;
	for (int x = 0; x < matrixDim; ++x)
	{
		value += mA[threadIdx.y * matrixDim + x] * mB[x * matrixDim + threadIdx.x];
	}
	result[threadIdx.y*matrixDim + threadIdx.x] = value;
}

__global__ void addCUDAMR(float* mA, float* mB, uint vectorSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// Safety if statement.
	if (index < vectorSize)
	{
		mA[index] += mB[index];
	}
}

namespace mat
{
	CUDAMR::CUDAMR(uint x, uint y)
	{
		if (x < 0)
			x = -x;
		if (y < 0)
			y = -y;

		value = nullptr;
		d_value = nullptr;

		size = sf::Vector2u(x, y);
		byteSize = size.x * size.y *sizeof(float);
		value = (float*)malloc(byteSize);
	}

	CUDAMR::~CUDAMR()
	{
		if (value != NULL)
			free(value);
		value = NULL;
		deallocateGPUMemory();
	}

	bool CUDAMR::setValue(float values[], uint size)
	{
		if (size != this->size.x*this->size.y)
			return false;

		std::memcpy(this->value, values, sizeof(float)*size);
		return true;
	}

	void CUDAMR::copyGPUValue()
	{
		cudaDeviceSynchronize();
		// You have to copy the result or use friends.
		cudaMemcpy(this->value, this->d_value, byteSize, cudaMemcpyDeviceToHost);
	}

	bool CUDAMR::allocateGPUMemory()
	{
		if (d_value != nullptr)
			return false;

		// Allocate!
		cudaMalloc(&d_value, byteSize);
		// Give it the info. Careful of interdeterminant values.
		cudaMemcpy(d_value, value, byteSize, cudaMemcpyHostToDevice);

		return true;
	}


	bool CUDAMR::deallocateGPUMemory()
	{
		if (d_value == nullptr)
			return false;
		cudaFree(d_value);
		return true;
	}

	void CUDAMR::add(CUDAMR* value)
	{
		if (value->size.x != this->size.x || value->size.y != this->size.y)
			return;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || value->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		// Kelvyne++
		addCUDAMR << <1, size.x*size.y >> > (d_value, value->d_value, size.x*size.y);
	}

	void CUDAMR::sub(CUDAMR* value)
	{
		if (value->size.x != this->size.x || value->size.y != this->size.y)
			return;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || value->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		// Kelvyne++
		subCUDAMR << <1, size.x*size.y >> > (d_value, value->d_value, size.x*size.y);
	}

	void CUDAMR::mult(CUDAMR* value, CUDAMR* result)
	{
		if (value->size.x != this->size.y || result->size.x != this->size.x || result->size.y != value->size.y)
			return;
		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || value->d_value == nullptr || result->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		dim3 grid(1, 1);
		dim3 thread(size.x, size.y);

		multCUDAMR << <grid, thread >> > (d_value, value->d_value, result->d_value, size.x);
	}
}