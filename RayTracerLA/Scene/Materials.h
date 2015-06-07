#ifndef MATERIALS_H
#define MATERIALS_H

#include <vector_types.h>

namespace Scene{
	enum MaterialType {
		DIFFUSE,
		SPECULAR
	};

	struct LambertMaterial {
		MaterialType MaterialType;
		float3 MainColor;
		float3 EmmisiveColor;
	};
}
#endif