#include "MatrixN.h"

namespace mat
{
	/* Cuda Related Code */
	/**
	* Adds NxN SQUARE matrices. Adds matrix B to matrix A. Overwrites matrix A.
	*/
	__global__ void addMatrixN(float* mA, float* mB, int matrixSize)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		// Safety if statement.
		if (index < matrixSize)
		{
			mA[index] += mB[index];
		}
	}

	/**
	* Subtracts NxN SQUARE matrices. Subtracts matrix B to matrix A. Overwrites matrix A.
	* Will not use a negate matrix due to copying an extra matrix which is slow.
	* Hard code is faster.
	*/
	__global__ void subMatrixN(float* mA, float* mB, int matrixSize)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		// Safety if statement.
		if (index < matrixSize)
		{
			mA[index] -= mB[index];
		}
	}

	/**
	* Multiplies one matrix by another matrix.
	*/
	__global__ void multMatrixN(float* mA, float* mB, float* result, int matrixDim)
	{
		float value = 0;
		for (int x = 0; x < matrixDim; ++x)
		{
			value += mA[threadIdx.y * matrixDim + x] * mB[x * matrixDim + threadIdx.x];
		}
		result[threadIdx.y*matrixDim + threadIdx.x] = value;
	}

	MatrixN::MatrixN(uint size)
	{
		if (size == 0)
			size = 1;
		else if (size < 0)
			size = -size;

		this->size = size;
		// Set location to null by def.
		d_value = nullptr;
		value = (float*)malloc(size*size*sizeof(float)); // Create empty array.
	}

	MatrixN::~MatrixN()
	{
		free(value);
		deallocateGPUMemory();
	}

	bool MatrixN::setValue(float values[], uint matrixDim)
	{
		// YOLO, no safety.
		if (size != this->size)
		{
			return false;
		}
		std::memcpy(this->value, values, sizeof(float) * matrixDim * matrixDim);
		return true;
	}

	void MatrixN::add(MatrixN* matrix)
	{
		if (matrix->size != this->size)
			return;

		size_t byteSize = sizeof(float) * this->size * this->size;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || matrix->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		// Kelvyne++
		addMatrixN <<<1,size*size>>> (d_value, matrix->d_value, size*size);
	}

	// UGLY, check TODO
	void MatrixN::sub(MatrixN* matrix)
	{
		if (matrix->size != this->size)
			return;

		size_t byteSize = sizeof(float) * this->size * this->size;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || matrix->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		// Kelvyne++
		subMatrixN <<<1, size*size >>> (d_value, matrix->d_value, size*size);
	}

	// Check TODO
	void MatrixN::mult(MatrixN* matrix, MatrixN* result)
	{
		if (matrix->size != this->size)
			return;
		
		size_t byteSize = sizeof(float) * this->size * this->size;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || matrix->d_value == nullptr || result->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		dim3 grid(1, 1);
		dim3 thread(size, size);

		multMatrixN <<<grid, thread>>> (d_value, matrix->d_value, result->d_value, this->size);
	}

	void MatrixN::copyGPUValue()
	{
		cudaDeviceSynchronize();
		// You have to copy the result or use friends.
		cudaMemcpy(this->value, this->d_value, sizeof(float) * this->size * this->size,
			cudaMemcpyDeviceToHost);
	}

	bool MatrixN::allocateGPUMemory()
	{
		if (d_value != nullptr)
			return false;

		// Size is given in constructor, will not change dynamically.
		size_t byteSize = sizeof(float) * this->size * this->size;
		// Allocate!
		cudaMalloc(&d_value, byteSize);
		// Give it the info. Careful of interdeterminant values.
		cudaMemcpy(d_value, value, byteSize, cudaMemcpyHostToDevice);

		return true;
	}

	bool MatrixN::deallocateGPUMemory()
	{
		if (d_value == nullptr)
			return false;
		cudaFree(d_value);
		return true;
	}

	/*Operator Overloads*/
	std::ostream& operator<<(std::ostream& os, const MatrixN& matN)
	{
		for (int x = 0; x < matN.size; ++x)
		{
			for (int i = 0; i < matN.size; ++i)
			{
				os << matN.value[x * matN.size + i];
				os << " ";
			}
			os << std::endl;
		}
		return os;
	}
}