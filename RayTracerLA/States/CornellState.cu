#include "CornellState.cuh"


void CornellState::end()
{
	cutilSafeCall(cudaFree(d_materials));
	cutilSafeCall(cudaFree(d_spheres));
	cutilSafeCall(cudaFree(d_rectangles));
	cutilSafeCall(cudaFree(d_circles));
	cutilSafeCall(cudaFree(d_sceneObjects));
	cutilSafeCall(cudaFree(d_cameraData));
}

void CornellState::setUpScene()
{
#define NUM_MATERIALS 5
	Scene::LambertMaterial materials[NUM_MATERIALS];
	materials[0] = { Scene::DIFFUSE, make_float3(1.0f, 1.0f, 1.0f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[1] = { Scene::DIFFUSE, make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[2] = { Scene::DIFFUSE, make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[3] = { Scene::SPECULAR, make_float3(0.95f, 0.95f, 0.95f), make_float3(0.0f, 0.0f, 0.0f) };
	materials[4] = { Scene::DIFFUSE, make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 10.0f, 10.0f) };

	cutilSafeCall(cudaMalloc(&d_materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial)));
	cutilSafeCall(cudaMemcpy(d_materials, &materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial), cudaMemcpyHostToDevice));


#define NUM_PLANES 0


#define NUM_RECTANGLES 6
	Scene::Rectangle rectangles[NUM_RECTANGLES];
	rectangles[0] = { make_float3(-10.0f, -10.0f, -10.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(20.0f, 0.0f, 0.0f), make_float3(0.0f, 20.0f, 0.0f), 0u }; // Front
	rectangles[1] = { make_float3(-10.0f, -10.0f, -10.0f), make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 20.0f), make_float3(0.0f, 20.0f, 0.0f), 1u }; // Left
	rectangles[2] = { make_float3(-10.0f, -10.0f, -10.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(20.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 20.0f), 0u }; // Bottom
	rectangles[3] = { make_float3(10.0f, 10.0f, 10.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(-20.0f, 0.0f, 0.0f), make_float3(0.0f, -20.0f, 0.0f), 0u }; // Back
	rectangles[4] = { make_float3(10.0f, 10.0f, 10.0f), make_float3(-1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, -20.0f), make_float3(0.0f, -20.0f, 0.0f), 2u }; // Right
	rectangles[5] = { make_float3(10.0f, 10.0f, 10.0f), make_float3(0.0f, -1.0f, 0.0f), make_float3(-20.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, -20.0f), 0u }; // Top

	cutilSafeCall(cudaMalloc(&d_rectangles, NUM_RECTANGLES * sizeof(Scene::Rectangle)));
	cutilSafeCall(cudaMemcpy(d_rectangles, &rectangles, NUM_RECTANGLES * sizeof(Scene::Rectangle), cudaMemcpyHostToDevice));


#define NUM_CIRCLES 1
	Scene::Circle circles[NUM_CIRCLES];
	circles[0] = { make_float3(0.0f, 9.99f, 0.0f), make_float3(0.0f, -1.0f, 0.0f), 8.0f, 4u }; // Light

	cutilSafeCall(cudaMalloc(&d_circles, NUM_CIRCLES * sizeof(Scene::Circle)));
	cutilSafeCall(cudaMemcpy(d_circles, &circles, NUM_CIRCLES * sizeof(Scene::Circle), cudaMemcpyHostToDevice));


#define NUM_SPHERES 2
	Scene::Sphere spheres[NUM_SPHERES];
	spheres[0] = { make_float3(-5.0f, -7.0f, -3.0f), 9.0f, 0u };
	spheres[1] = { make_float3(3.0f, -7.0f, 5.0f), 9.0f, 3u };

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

	camera = Camera(-3.14f, 3.14f / 2, 45.0f);
}