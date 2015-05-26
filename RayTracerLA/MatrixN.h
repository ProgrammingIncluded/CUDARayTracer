#ifndef MATRIXN_H
#define MATRIXN_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"

#include <iostream>
#include <vector>

#include "GeneralTypedef.h"

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
			* Adds the give matrix to this matrix.
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
			void mult(MatrixN* matrix);

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

			// Use RREF
			// float det();

			/*Operator Overloads*/
			friend std::ostream& operator<<(std::ostream& os, const MatrixN& matN);

		protected:
			// Pointer to 1D array storing all the matrix data.
			// Converted to 2D within functions.
			float* value;
			//MatrixN createSubMatrix(uint size);

		private:
			uint size;
	};
}
#endif