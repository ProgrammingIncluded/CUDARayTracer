#ifndef VECTN_H
#define VECTN_H

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "vector_types.h"

#include <iostream>

#include "CUDAMR.cuh"
#include "Core/GeneralTypedef.h"

namespace mat
{
	/**
	* Class for vectors. Does not require to be a column or row vector.
	*/
	class VectN : public CUDAMR
	{
		public:
			// By default, size.x holds size of vector.
			VectN(uint size);
			~VectN();
	};
}
#endif // VECTN_H