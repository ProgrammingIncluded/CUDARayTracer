#ifndef SCMATH_H
#define SCMATH_H

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

inline float3 make_float3(sf::Vector3f vect)
{
	float3 val;
	val.x = vect.x;
	val.y = vect.y;
	val.z = vect.z;
	return val;
}

inline sf::Vector3f make_vector3f(float3 val)
{
	return sf::Vector3f(val.x, val.y, val.z);
}

#endif // SCMATH_H