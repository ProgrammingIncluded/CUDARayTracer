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