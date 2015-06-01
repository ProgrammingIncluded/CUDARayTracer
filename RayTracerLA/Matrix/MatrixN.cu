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
	__global__ void multMatrixN(float* mA, float* mB, float* mR, int matrixDim)
	{
		float value = 0;
		for (int x = 0; x < matrixDim; ++x)
		{
			value += mA[threadIdx.y * matrixDim + x] * mB[x * matrixDim + threadIdx.x];
		}
		mR[threadIdx.y*matrixDim + threadIdx.x] = value;
	}

	MatrixN::MatrixN(uint size)
	{
		if (size == 0)
			size = 1;
		else if (size < 0)
			size = -size;

		this->size = size;
		value = (float*)malloc(size*size*sizeof(float)); // Create empty array.
	}

	MatrixN::~MatrixN()
	{
		free(value);
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
		// Cuda memory pointer.
		float* d_value;
		float* d_adder;

		size_t byteSize = sizeof(float) * this->size * this->size;

		// Allocate memory in GPU (like new or malloc)
		cudaMalloc(&d_value, byteSize);
		cudaMalloc(&d_adder, byteSize);

		// Copy our values to GPU.
		cudaMemcpy(d_value, this->value, byteSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_adder, matrix->value, byteSize, cudaMemcpyHostToDevice);

		addMatrixN <<<1, size*size>>> (d_value, d_adder, this->size*this->size);
		// Wait for the adding to finish.
		cudaDeviceSynchronize();

		// Copy value from GPU to CPU.
		cudaMemcpy(this->value, d_value, byteSize, cudaMemcpyDeviceToHost);

		// Free Variables in GPU
		cudaFree(d_value);
		cudaFree(d_adder);
	}

	// UGLY, check TODO
	void MatrixN::sub(MatrixN* matrix)
	{
		if (matrix->size != this->size)
			return;
		// Cuda memory pointer.
		float* d_value;
		float* d_adder;

		size_t byteSize = sizeof(float) * this->size * this->size;

		// Allocate memory in GPU (like new or malloc)
		cudaMalloc(&d_value, byteSize);
		cudaMalloc(&d_adder, byteSize);

		// Copy our values to GPU.
		cudaMemcpy(d_value, this->value, byteSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_adder, matrix->value, byteSize, cudaMemcpyHostToDevice);

		subMatrixN <<<1, size*size >> > (d_value, d_adder, this->size*this->size);
		// Wait for the adding to finish.
		cudaDeviceSynchronize();

		// Copy value from GPU to CPU.
		cudaMemcpy(this->value, d_value, byteSize, cudaMemcpyDeviceToHost);

		// Free Variables in GPU
		cudaFree(d_value);
		cudaFree(d_adder);
	}

	// Check TODO
	void MatrixN::mult(MatrixN* matrix)
	{
		if (matrix->size != this->size)
			return;

		size_t byteSize = sizeof(float) * this->size * this->size;

		float* result = (float*)malloc(byteSize);

		// Cuda memory pointer.
		float* d_value;
		float* d_mult;
		float* d_result;

		// Allocate memory in GPU (like new or malloc)
		cudaMalloc(&d_value, byteSize);
		cudaMalloc(&d_mult, byteSize);
		cudaMalloc(&d_result, byteSize);

		// Copy our values to GPU.
		cudaMemcpy(d_value, this->value, byteSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_mult, matrix->value, byteSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_result, result, byteSize, cudaMemcpyHostToDevice);

		dim3 grid(1, 1);
		dim3 thread(size, size);

		multMatrixN <<<grid, thread>>> (d_value, d_mult, d_result, this->size);
		// Wait for the adding to finish.
		cudaDeviceSynchronize();

		free(result);

		// Copy value from GPU to CPU.
		cudaMemcpy(this->value, d_result, byteSize, cudaMemcpyDeviceToHost);

		// Free Variables in GPU
		cudaFree(d_value);
		cudaFree(d_mult);
		cudaFree(d_result);
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