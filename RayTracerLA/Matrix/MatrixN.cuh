#ifndef MATRIXN_H
#define MATRIXN_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "vector_types.h"

#include <iostream>
#include <vector>

#include "CUDAMR.cuh"
#include "CMROperations.cuh"
#include "GeneralTypedef.h"

namespace mat
{
	/**
	* Base class for Matrices and designed with CUDA support in mind.
	* Basic square dimensional matrix. Square matrices only.
	*
	* TODO: Add better function usage with mem allocation in specific function?
	*/
	class MatrixN : public CUDAMR<float>
	{
		public:
			MatrixN(uint size);
			~MatrixN();

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

			//MatrixN createSubMatrix(uint size);
	};
}
#endif