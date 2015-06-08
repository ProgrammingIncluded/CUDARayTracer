#include "BallState.cuh"


void BallState::end()
{
	cutilSafeCall(cudaFree(d_materials));
	cutilSafeCall(cudaFree(d_planes));
	cutilSafeCall(cudaFree(d_spheres));
	cutilSafeCall(cudaFree(d_sceneObjects));
	cutilSafeCall(cudaFree(d_cameraData));
}

void BallState::setUpScene()
{
	#define NUM_MATERIALS 6
	Scene::LambertMaterial materials[NUM_MATERIALS];
	materials[0] = { Scene::DIFFUSE, make_float3(0.9f, 0.9f, 0.9f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[1] = { Scene::DIFFUSE, make_float3(0.408f, 0.741f, 0.467f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[2] = { Scene::DIFFUSE, make_float3(0.392f, 0.584f, 0.929f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[3] = { Scene::DIFFUSE, make_float3(1.0f, 0.498f, 0.314f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[4] = { Scene::DIFFUSE, make_float3(1.0f, 1.0f, 1.0f), make_float3(3.0f, 3.0f, 3.0f) };
	materials[5] = { Scene::SPECULAR, make_float3(0.95f, 0.95f, 0.95f), make_float3(0.0f, 0.0f, 0.0f) };

	cutilSafeCall(cudaMalloc(&d_materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial)));
	cutilSafeCall(cudaMemcpy(d_materials, &materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial), cudaMemcpyHostToDevice));


	#define NUM_PLANES 1
	Scene::Plane planes[NUM_PLANES];
	planes[0] = { make_float3(0.0f, -6.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 0u }; // Front

	cutilSafeCall(cudaMalloc(&d_planes, NUM_PLANES * sizeof(Scene::Plane)));
	cutilSafeCall(cudaMemcpy(d_planes, &planes, NUM_PLANES * sizeof(Scene::Plane), cudaMemcpyHostToDevice));


	#define NUM_RECTANGLES 0
	#define NUM_CIRCLES 0


	#define NUM_SPHERES 9
	Scene::Sphere spheres[NUM_SPHERES];
	spheres[0] = { make_float3(0.0f, 0.0f, 0.0f), 4.0f, 4u };
	spheres[1] = { make_float3(-4.0f, -4.0f, -4.0f), 4.0f, 1u };
	spheres[2] = { make_float3(-4.0f, -4.0f, 4.0f), 4.0f, 2u };
	spheres[3] = { make_float3(-4.0f, 4.0f, -4.0f), 4.0f, 3u };
	spheres[4] = { make_float3(-4.0f, 4.0f, 4.0f), 4.0f, 5u };
	spheres[5] = { make_float3(4.0f, -4.0f, -4.0f), 4.0f, 5u };
	spheres[6] = { make_float3(4.0f, -4.0f, 4.0f), 4.0f, 3u };
	spheres[7] = { make_float3(4.0f, 4.0f, -4.0f), 4.0f, 2u };
	spheres[8] = { make_float3(4.0f, 4.0f, 4.0f), 4.0f, 1u };

	cutilSafeCall(cudaMalloc(&d_spheres, NUM_SPHERES * sizeof(Scene::Sphere)));
	cutilSafeCall(cudaMemcpy(d_spheres, &spheres, NUM_SPHERES * sizeof(Scene::Sphere), cudaMemcpyHostToDevice));

	Scene::SceneObjects sceneObjects;
	sceneObjects.Materials = d_materials;
	sceneObjects.Planes = d_planes;
	sceneObjects.NumPlanes = NUM_PLANES;
	sceneObjects.Rectangles = d_rectangles;
	sceneObjects.NumRectangles = NUM_RECTANGLES;
	sceneObjects.Circles = d_circles;
	sceneObjects.NumCircles = NUM_CIRCLES;
	sceneObjects.Spheres = d_spheres;
	sceneObjects.NumSpheres = NUM_SPHERES;

	cutilSafeCall(cudaMalloc(&d_sceneObjects, sizeof(Scene::SceneObjects)));
	cutilSafeCall(cudaMemcpy(d_sceneObjects, &sceneObjects, sizeof(Scene::SceneObjects), cudaMemcpyHostToDevice));

	camera = Camera(-3.14, 3.14 / 2, 30.0f);
}