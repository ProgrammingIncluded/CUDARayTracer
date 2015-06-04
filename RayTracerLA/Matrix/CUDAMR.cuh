#ifndef CUDAMR_H
#define CUDAMR_H

#include <SFML/Graphics.hpp>

#include "cuda_runtime.h"
#include "vector_types.h"
#include "GeneralTypedef.h"

namespace mat
{
	/**
	* Class to extended by any matrix or vector related functions.
	* Allows the class to work with CUDA.
	*/
	template <typename T>
	class CUDAMR
	{
		public:

			bool setValue(float values[], uint size)
			{
				if (size != this->size.x*this->size.y)
					return false;

				std::memcpy(this->value, values, sizeof(T)*size);
				return true;
			}

			/**
			* Copies the value from GPU to CPU memory.
			*/
			void copyGPUValue()
			{
				cudaDeviceSynchronize();
				// You have to copy the result or use friends.
				cudaMemcpy(this->value, this->d_value, byteSize,cudaMemcpyDeviceToHost);
			};

			/**
			* Call this function when you want to allocate memory into the GPU.
			* Mainly used internally. Returns false if space already allocated.
			* Short answer to our slowness problem. Nothing too complicated like a
			* memory manager.
			*/
			bool allocateGPUMemory()
			{
				if (d_value != nullptr)
					return false;

				// Allocate!
				cudaMalloc(&d_value, byteSize);
				// Give it the info. Careful of interdeterminant values.
				cudaMemcpy(d_value, value, byteSize, cudaMemcpyHostToDevice);

				return true;
			};

			/**
			* For whatever reason, perhaps a melt down, use this to delete allocation
			* in GPU. Returns false if internal ptr is null.
			*/
			bool deallocateGPUMemory()
			{
				if (d_value == nullptr)
					return false;
				cudaFree(d_value);
				return true;
			};
		protected:
			CUDAMR(uint x, uint y)
			{
				if (x < 0)
					x = -x;
				if (y < 0)
					y = -y;

				value = nullptr;
				d_value = nullptr;

				size = sf::Vector2u(x,y);
				byteSize = size.x * size.y *sizeof(T);
				value = (T*) malloc(byteSize);
			}

			virtual ~CUDAMR()
			{
				if (value != NULL)
					free(value);
				value = NULL;
				deallocateGPUMemory();
			}


			// Pointer to 1D array storing all the matrix data.
			// Converted to 2D within functions.
			T* value;
			// Pointer to allocation in GPU. Null by default.
			T* d_value;

			// Stores the byte size of the allocated space for reference.
			size_t byteSize;

			// Represents the number of elements inside the array.
			sf::Vector2u size;
	
	};

}

#endif // CUDAMR_H