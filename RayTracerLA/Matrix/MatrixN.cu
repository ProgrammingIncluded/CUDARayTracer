#include "MatrixN.cuh"

namespace mat
{
	/**
	* Multiplies one matrix by another matrix. Assumes square.
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

	

	MatrixN::MatrixN(uint size) : CUDAMR(size, size)
	{
		if (size == 0)
			size = 1;
		else if (size < 0)
			size = -size;

		this->size.x = size;
		this->size.y = size;
		// Set location to null by def.
		d_value = nullptr;
		value = (float*)malloc(size*size*sizeof(float)); // Create empty array.
	}

	MatrixN::~MatrixN()
	{
		free(value);
		deallocateGPUMemory();
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
		subCUDAMR <<<1, size*size >>> (d_value, matrix->d_value, size.x*size.y);
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