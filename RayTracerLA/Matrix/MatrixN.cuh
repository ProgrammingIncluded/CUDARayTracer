#ifndef MATRIXN_H
#define MATRIXN_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "vector_types.h"

#include <iostream>
#include <vector>

#include "CUDAMR.cuh"
#include "GeneralTypedef.h"

namespace mat
{
	/**
	* Base class for Matrices and designed with CUDA support in mind.
	* Basic square dimensional matrix. Square matrices only.
	*
	* TODO: Add better function usage with mem allocation in specific function?
	*/
	class MatrixN : public CUDAMR
	{
		public:
			MatrixN(uint dim);
			~MatrixN();

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

			//MatrixN createSubMatrix(uint size);
	};
}
#endif