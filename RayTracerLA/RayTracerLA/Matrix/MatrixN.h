#ifndef MATRIXN_H
#define MATRIXN_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "vector_types.h"

#include <iostream>
#include <vector>

#include "../GeneralTypedef.h"

namespace mat
{
	/**
	* Base class for Matrices and designed with CUDA support in mind.
	* Basic square dimensional matrix. Square matrices only.
	*
	* TODO: Add better function usage with mem allocation in specific function?
	*/
	class MatrixN
	{
		public:
			MatrixN(uint size);
			~MatrixN();

			bool setValue(float values[], uint matrixSize);

			/**
			* Adds the give matrix to this matrix. Assumes values have been allocated.
			*/
			void add(MatrixN* matrix);

			/**
			* Subtracts this matrix with given matrix. Internally does not call
			* negate due to negate overwrites.
			*/
			void sub(MatrixN* matrix);

			/**
			* Mutliplies one matrix by another.
			*/
			// For when you have a memory location to spare.
			void mult(MatrixN* matrix, MatrixN* result);

			/**
			* Makes all values the opposite in the matrix.
			* Wipes previous data, use with caution.
			*/
			//void negate();

			/**
			* Adjusts the matrix to Row Reduced Echelon Form.
			*/
			//void rref();

			//MatrixN copy();

			// Use RREF w/ Augumented Matrix. Thank you Prof. Van Lingen!
			// float det();

			/*Operator Overloads*/
			friend std::ostream& operator<<(std::ostream& os, const MatrixN& matN);

			/**
			* Copies the value from GPU to CPU memory.
			*/
			void copyGPUValue();

			/**
			* Call this function when you want to allocate memory into the GPU.
			* Mainly used internally. Returns false if space already allocated.
			* Short answer to our slowness problem. Nothing too complicated like a
			* memory manager.
			*/
			bool allocateGPUMemory();

			/**
			* For whatever reason, perhaps a melt down, use this to delete allocation
			* in GPU. Returns false if internal ptr is null. 
			*/
			bool deallocateGPUMemory();

		protected:
			// Pointer to 1D array storing all the matrix data.
			// Converted to 2D within functions.
			float* value;
			// Pointer to allocation in GPU. Null by default.
			float* d_value;

			//MatrixN createSubMatrix(uint size);

		private:
			uint size;
	};
}
#endif