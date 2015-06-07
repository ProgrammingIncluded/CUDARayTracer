#include "PathTracer.cuh"

//#define IMPORTANCE_SAMPLE

#define PI 3.14159265359f
#define TWO_PI 6.283185307f
#define PI_DIV_TWO 1.57079632679f
#define INV_SQRT_THREE 0.5773502691896257645091487805019574556476f 

__global__ void pathTraceKernel(uchar4* textureData, uint width, uint height, CameraData *g_camera, Scene::SceneObjects *g_sceneObjects, uint hashedFrameNumber) {
	// Create a local copy of the arguments
	CameraData camera = *g_camera;
	Scene::SceneObjects sceneObjects = *g_sceneObjects;

	// Global threadId
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// Create random number generator
	curandState randState;
	curand_init(hashedFrameNumber + threadId, 0, 0, &randState);


	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Calculate the first ray for this pixel
	Scene::Ray ray = { camera.Origin, calculateRayDirectionFromPixel(x, y, width, height, camera, &randState) };


	float3 pixelColor = make_float3(0.0f, 0.0f, 0.0f);
	float3 accumulatedMaterialColor = make_float3(1.0f, 1.0f, 1.0f);

	// Bounce the ray around the scene
	for (uint bounces = 0; bounces < 10; ++bounces) {
		// Initialize the intersection variables
		float closestIntersection = FLT_MAX;
		float3 normal;
		Scene::LambertMaterial material;

		testSceneIntersection(ray, sceneObjects, &closestIntersection, &normal, &material);

		// Find out if we hit anything
		if (closestIntersection < FLT_MAX) {
			// We hit an object

			// Add the emmisive light
			pixelColor += accumulatedMaterialColor * material.EmmisiveColor;


			// Shoot a new ray

			// Set the origin at the intersection point
			ray.Origin = ray.Origin + ray.Direction * closestIntersection;
			// Offset the origin to prevent self intersection
			ray.Origin += normal * 0.001f;

			// Choose the direction based on the material
			if (material.MaterialType == Scene::DIFFUSE) {
#ifdef IMPORTANCE_SAMPLE
				ray.Direction = createCosineWeightedDirectionInHemisphere(normal, &randState);

				// Accumulate the diffuse/specular color
				accumulatedMaterialColor *= material.MainColor; // * dot(ray.Direction, normal) / PI     // Cancels with pdf

				// Divide by the pdf
				//accumulatedMaterialColor /= dot(ray.Direction, normal) / PI
#else
				ray.Direction = createUniformDirectionInHemisphere(normal, &randState);

				// Accumulate the diffuse/specular color
				accumulatedMaterialColor *= material.MainColor /* * (1 / PI)  <- this cancels with the PI in the pdf */ * dot(ray.Direction, normal);

				// Divide by the pdf
				accumulatedMaterialColor *= 2.0f; // pdf == 1 / (2 * PI)
#endif
			}
			else if (material.MaterialType == Scene::SPECULAR) {
				ray.Direction = reflect(ray.Direction, normal);

				// Accumulate the diffuse/specular color
				accumulatedMaterialColor *= material.MainColor;
			}


			// Russian Roulette
			if (bounces > 3) {
				float p = max(accumulatedMaterialColor.x, max(accumulatedMaterialColor.y, accumulatedMaterialColor.z));
				if (curand_uniform(&randState) > p) {
					break;
				}
				accumulatedMaterialColor *= 1 / p;
			}
		}
		else {
			// We didn't hit anything, return the sky color
			pixelColor += accumulatedMaterialColor * make_float3(0.846f, 0.933f, 0.949f);

			break;
		}
	}


	if (x < width && y < height) {
		// Get a pointer to the pixel at (x,y)
		uint index = x + y * width;

		// Write pixel data
		textureData[index].x += pixelColor.x;
		textureData[index].y += pixelColor.y;
		textureData[index].z += pixelColor.z;
		textureData[index].w = 255.0f;
	}
}

__device__ float3 calculateRayDirectionFromPixel(uint x, uint y, uint width, uint height, CameraData &camera, curandState *randState) {
	float3 viewVector = make_float3((((x + curand_uniform(randState)) / width) * 2.0f - 1.0f) * camera.tanFovXDiv2,
		-(((y + curand_uniform(randState)) / height) * 2.0f - 1.0f) * camera.tanFovYDiv2,
		1.0f);

	// Matrix multiply
	return normalize(make_float3(dot(viewVector, camera.ViewToWorldMatrixR0),
		dot(viewVector, camera.ViewToWorldMatrixR1),
		dot(viewVector, camera.ViewToWorldMatrixR2)));
}

/**
* Creates a uniformly random direction in the hemisphere defined by the normal
*
* Based on http://www.rorydriscoll.com/2009/01/07/better-sampling/
*
* @param normal        The normal that defines the hemisphere
* @param randState     The random state to use for internal random number generation
* @return              A uniformly random direction in the hemisphere
*/
__device__ float3 createUniformDirectionInHemisphere(float3 normal, curandState *randState) {
	// Create a random coordinate in spherical space
	// Then calculate the cartesian equivalent
	float z = curand_uniform(randState);
	float r = sqrt(1.0f - z * z);
	float phi = TWO_PI * curand_uniform(randState);
	float x = cos(phi) * r;
	float y = sin(phi) * r;

	// Find an axis that is not parallel to normal
	float3 majorAxis;
	if (abs(normal.x) < INV_SQRT_THREE) {
		majorAxis = make_float3(1, 0, 0);
	}
	else if (abs(normal.y) < INV_SQRT_THREE) {
		majorAxis = make_float3(0, 1, 0);
	}
	else {
		majorAxis = make_float3(0, 0, 1);
	}

	// Use majorAxis to create a coordinate system relative to world space
	float3 u = normalize(cross(majorAxis, normal));
	float3 v = cross(normal, u);
	float3 w = normal;

	// Transform from spherical coordinates to the cartesian coordinates space
	// we just defined above, then use the definition to transform to world space
	return normalize(u * x +
		v * y +
		w * z);
}

/**
* Creates a random direction in the hemisphere defined by the normal, weighted by a cosine lobe
*
* Based on http://www.rorydriscoll.com/2009/01/07/better-sampling/
*
* @param normal        The normal that defines the hemisphere
* @param randState     The random state to use for internal random number generation
* @return              A cosine weighted random direction in the hemisphere
*/
__device__ float3 createCosineWeightedDirectionInHemisphere(float3 normal, curandState *randState) {
	// Create random coordinates in the local coordinate system
	float rand = curand_uniform(randState);
	float r = sqrt(rand);
	float theta = curand_uniform(randState) * TWO_PI;

	float x = r * cos(theta);
	float y = r * sin(theta);


	// Find an axis that is not parallel to normal
	float3 majorAxis;
	if (abs(normal.x) < INV_SQRT_THREE) {
		majorAxis = make_float3(1, 0, 0);
	}
	else if (abs(normal.y) < INV_SQRT_THREE) {
		majorAxis = make_float3(0, 1, 0);
	}
	else {
		majorAxis = make_float3(0, 0, 1);
	}

	// Use majorAxis to create a coordinate system relative to world space
	float3 u = normalize(cross(majorAxis, normal));
	float3 v = cross(normal, u);
	float3 w = normal;


	// Transform from local coordinates to world coordinates
	return normalize(u * x +
		v * y +
		w * sqrt(fmaxf(0.0f, 1.0f - rand)));
}


uint32 WangHash(uint32 a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

/**
* Any code that uses CUDA semantics has to be a .cu/.cuh file. Instead of forcing
* the entire project to be .cu/.cuh files, we can just wrap each cuda kernel launch
* in a function. We forward declare the function in other parts of the project and
* implement the functions here, in this .cu file.
*/
cudaError pathTraceNextFrame(uchar4* buffer, uint width, uint height, CameraData *camera, Scene::SceneObjects *sceneObjects, uint frameNumber) {
	dim3 Db = dim3(16, 16);   // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3((width + Db.x - 1) / Db.x, (height + Db.y - 1) / Db.y);

	// The actual kernel launch call
	pathTraceKernel <<<Dg, Db >>>(buffer, width, height, camera, sceneObjects, WangHash(frameNumber));

	return cudaGetLastError();
}