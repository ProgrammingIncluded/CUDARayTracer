#include "MatrixN.cuh"

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
}