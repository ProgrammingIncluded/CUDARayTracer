#include "MatrixN.cuh"

/* Cuda Matrix Related Operations */

/**
* Adds two CUDAMR objects together given their values.
*/
__global__ void addCUDAMR(float* mA, float* mB, int vectorSize)
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
__global__ void subCUDAMR(float* mA, float* mB, int vectorSize)
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
__global__ void multMatrixN(float* mA, float* mB, float* result, uint matrixDim)
{
	float value = 0;
	for (int x = 0; x < matrixDim; ++x)
	{
		value += mA[threadIdx.y * matrixDim + x] * mB[x * matrixDim + threadIdx.x];
	}
	result[threadIdx.y*matrixDim + threadIdx.x] = value;
}

namespace mat
{

	MatrixN::MatrixN(uint dim) : CUDAMR(dim, dim)
	{
		if (dim == 0)
			dim = 1;
		else if (dim < 0)
			dim = -dim;
	}

	MatrixN::~MatrixN()
	{
	}

	void MatrixN::add(MatrixN* matrix)
	{
		if (matrix->size.x != this->size.x)
			return;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || matrix->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		// Kelvyne++
		addCUDAMR <<<1,size.x*size.y>>> (d_value, matrix->d_value, size.x*size.y);
	}

	// UGLY, check TODO
	void MatrixN::sub(MatrixN* matrix)
	{
		if (matrix->size.x != this->size.x)
			return;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || matrix->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		// Kelvyne++
		subCUDAMR <<<1, size.x*size.y >>> (d_value, matrix->d_value, size.x*size.y);
	}

	// Check TODO
	void MatrixN::mult(MatrixN* matrix, MatrixN* result)
	{
		if (matrix->size != this->size)
			return;
		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || matrix->d_value == nullptr || result->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		dim3 grid(1, 1);
		dim3 thread(size.x, size.x);

		multMatrixN <<<grid, thread>>> (d_value, matrix->d_value, result->d_value, this->size.x);
	}

	/*Operator Overloads*/
	std::ostream& operator<<(std::ostream& os, const MatrixN& matN)
	{

		for (int x = 0; x < matN.size.x; ++x)
		{
			for (int i = 0; i < matN.size.y; ++i)
			{
				os << matN.value[x * matN.size.x + i];
				os << " ";
			}
			os << std::endl;
		}
		return os;
	}
}