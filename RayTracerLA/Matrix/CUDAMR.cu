#include "CUDAMR.cuh"

/* Cuda Matrix Related Operations */



/**
* Subtracts two CUDAMR objects together given their values.
*/
__global__ void kernelSubCUDAMR(float* mA, float* mB, uint vectorSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// Safety if statement.
	if (index < vectorSize)
	{
		mA[index] -= mB[index];
	}
}

CUDA_MFUNCTION void subCUDAMR(float* mA, float* mB, uint vectorSize)
{
	for (uint index = 0; index < vectorSize; ++index)
		mA[index] -= mB[index];
}

/**
* Multiplies one matrix by another matrix. Assumes square.
*/
__global__ void kernelMultCUDAMR(float* A, float* B, float* C, uint aRow, uint bRow, uint cRow) {
	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < aRow; ++e)
		Cvalue += A[row * aRow + e] *
		B[e * bRow + col];
	C[row * cRow + col] = Cvalue;
}

CUDA_MFUNCTION void multCUDAMR(float* A, float* B, float* C, uint aRow, uint bRow, uint cRow)
{
	float Cvalue = 0;
	for (uint row = 0; row < aRow; ++row){
		for (uint col = 0; row < bRow; ++row){
			for (int e = 0; e < aRow; ++e)
				Cvalue += A[row * aRow + e] *
				B[e * bRow + col];
			C[row * cRow + col] = Cvalue;
		}
	}
}

/**
* Adds two CUDAMR objects together given their values.
*/
__global__ void kernelAddCUDAMR(float* mA, float* mB, uint vectorSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// Safety if statement.
	if (index < vectorSize)
	{
		mA[index] += mB[index];
	}
}

CUDA_MFUNCTION void addCUDAMR(float* mA, float* mB, uint vectorSize)
{
	for (uint index = 0; index < vectorSize; ++index)
		mA[index] += mB[index];
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
		if (this->d_value == nullptr)
			return;
		//cudaDeviceSynchronize();
		// You have to copy the result or use friends.
		cudaMemcpy(this->value, this->d_value, byteSize, cudaMemcpyDeviceToHost);
	}

	void CUDAMR::copyValueToGPU()
	{
		if (this->d_value == nullptr)
		return;

		cudaMemcpy(this->d_value, this->value, byteSize, cudaMemcpyHostToDevice);
	}

	bool CUDAMR::allocateGPUMemory()
	{
		if (d_value != nullptr)
			return false;

		// Allocate!
		cudaMalloc(&d_value, byteSize);
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
		kernelAddCUDAMR << <1, size.x*size.y >> > (d_value, value->d_value, size.x*size.y);
	}

	void CUDAMR::sub(CUDAMR* value)
	{
		if (value->size.x != this->size.x || value->size.y != this->size.y)
			return;

		// Check if GPU memory is allocated.
		if (this->d_value == nullptr || value->d_value == nullptr)
			return; // Add code to allocate or just silently return?

		// Kelvyne++
		kernelSubCUDAMR << <1, size.x*size.y >> > (d_value, value->d_value, size.x*size.y);
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

		kernelMultCUDAMR << <grid, thread >> > (d_value, value->d_value, result->d_value, size.y, 
			value->size.y, result->size.y);
	}

	std::ostream& operator<<(std::ostream& os, const CUDAMR& obj)
	{

		for (int x = 0; x < obj.size.x; ++x)
		{
			for (int i = 0; i < obj.size.y; ++i)
			{
				os << obj.value[x * obj.size.y + i];
				os << " ";
			}
			os << std::endl;
		}
		return os;
	}

}