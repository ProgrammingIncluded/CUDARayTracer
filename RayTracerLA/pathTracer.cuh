#ifndef PATHTRACER_CUH
#define PATHTRACER_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "ObjectIntersection.cuh"

#include "Scene/SceneObjects.h"
#include "Scene/Materials.h"
#include "Scene/CameraData.h"
#include "Core/helper_math.h"

#include "Core/GeneralTypedef.h"

__global__ void pathTraceKernel(unsigned char *textureData, uint width, uint height, CameraData *g_camera, Scene::SceneObjects *g_sceneObjects, uint hashedFrameNumber);
__device__ float3 calculateRayDirectionFromPixel(uint x, uint y, uint width, uint height, CameraData &camera, curandState *randState);
__device__ float3 createUniformDirectionInHemisphere(float3 normal, curandState *randState);
__device__ float3 createCosineWeightedDirectionInHemisphere(float3 normal, curandState *randState);

uint32 WangHash(uint32 a);

cudaError pathTraceNextFrame(uchar4* buffer, uint width, uint height, CameraData *camera, Scene::SceneObjects *sceneObjects, uint frameNumber);

#endif // PATHTRACER_CUH