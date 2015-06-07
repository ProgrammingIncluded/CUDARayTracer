#include "TraceState.cuh"

State* TraceState::createInternal(StateManager *sm, sf::RenderWindow *rw)
{
	return new TraceState(sm, rw);
}

__global__ void titleDraw(uchar4* pos, unsigned int width, unsigned int height, float time)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int x = index%width;
	unsigned int y = index / width;

	if (index < width*height) {
		unsigned char r = (x + (int)time) & 0xff;
		unsigned char g = (y + (int)time) & 0xff;
		unsigned char b = ((x + y) + (int)time) & 0xff;

		// Each thread writes one pixel location in the texture (textel)
		pos[index] = make_uchar4(r, g, b, 255.0f);
	}
}

void TraceState::input(sf::Event &event)
{
	if (event.type == sf::Event::Closed)
	{
		stateManager->popState();
	}
	else if (event.type == sf::Event::KeyPressed)
	{
		if (event.key.code == sf::Keyboard::Escape)
			stateManager->popState();
		else if (event.key.code == sf::Keyboard::P)
			pauseRender = !pauseRender;
	}
}

void TraceState::update()
{

}

void TraceState::draw(uchar4* canvas, float time)
{
	if (pauseRender == true)
		return;
	sf::Vector2u windowSize = window->getSize();
	uint totalPixels = windowSize.x * windowSize.y;
	pathTraceNextFrame(canvas, windowSize.x, windowSize.y, d_cameraData, d_sceneObjects, time);
}

void TraceState::pause()
{

}

void TraceState::resume()
{

}

void TraceState::setUp()
{
	cameraMoved = false;
	pauseRender = false;
	frameCount = 0;

	setUpScene();
	cutilSafeCall(cudaMalloc(&d_cameraData, sizeof(CameraData)));
	sf::Vector2u size = window->getSize();
	camera.setProjection(1.04719755f, (float)size.x/ size.y);
	updateCameraData(d_cameraData);
}

void TraceState::end()
{
	cutilSafeCall(cudaFree(d_materials));
	//cutilSafeCall(cudaFree(d_planes));
	cutilSafeCall(cudaFree(d_spheres));
	//cutilSafeCall(cudaFree(d_rectangles));
	//cutilSafeCall(cudaFree(d_circles));
	cutilSafeCall(cudaFree(d_sceneObjects));
	cutilSafeCall(cudaFree(d_cameraData));
}

void TraceState::updateCameraData(CameraData* dataPtr)
{
	float3 worldUp = make_float3(0.0f, camera.getUp(), 0.0f);
	float3 origin = make_float3(camera.getPosition());

	float3 zAxis = normalize(make_float3(camera.getTarget()) - origin);
	float3 xAxis = normalize(cross(worldUp, zAxis));
	float3 yAxis = cross(zAxis, xAxis);

	/*float value[9] =
	{
	xAxis.x, yAxis.x, zAxis.x,
	xAxis.y, yAxis.y, zAxis.y,
	xAxis.z, yAxis.z, zAxis.z
	};

	data.vpMatrix.setValue(value, 9);
	value[0] = origin.x;
	value[1] = origin.y;
	value[2] = origin.z;
	data.origin.setValue(value, 3);
	*/

	CameraData data;
	data.ViewToWorldMatrixR0 = make_float3(xAxis.x, yAxis.x, zAxis.x);
	data.ViewToWorldMatrixR1 = make_float3(xAxis.y, yAxis.y, zAxis.y);
	data.ViewToWorldMatrixR2 = make_float3(xAxis.z, yAxis.z, zAxis.z);

	data.Origin = origin;

	data.tanFovXDiv2 = camera.getTanFovXDiv2();
	data.tanFovYDiv2 = camera.getTanFovYDiv2();
	cutilSafeCall(cudaMemcpy(dataPtr, &data, sizeof(CameraData), cudaMemcpyHostToDevice));
}
/*
void TraceState::setUpScene()
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


	// Store the representation of the scene in a single object
	// CUDA only allows 256 bytes of data to be passed as arguments in the kernel launch
	// We can save some room by bundling all the scene variables together
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


	// Set the camera
	camera = Camera(-3.14f, 3.14f/2, 45.0f);


	// Set the exposure
	exposure = 1.0f;
}
*/
void TraceState::setUpScene()
{
	#define NUM_MATERIALS 6
	Scene::LambertMaterial materials[NUM_MATERIALS];
	materials[0] = {Scene::DIFFUSE, make_float3(0.9f, 0.9f, 0.9f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[1] = {Scene::DIFFUSE, make_float3(0.408f, 0.741f, 0.467f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[2] = {Scene::DIFFUSE, make_float3(0.392f, 0.584f, 0.929f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[3] = {Scene::DIFFUSE, make_float3(1.0f, 0.498f, 0.314f), make_float3(0.0f, 0.0f, 0.0f)};
	materials[4] = {Scene::DIFFUSE, make_float3(1.0f, 1.0f, 1.0f), make_float3(3.0f, 3.0f, 3.0f)};
	materials[5] = {Scene::SPECULAR, make_float3(0.95f, 0.95f, 0.95f), make_float3(0.0f, 0.0f, 0.0f)};

	cutilSafeCall(cudaMalloc(&d_materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial)));
	cutilSafeCall(cudaMemcpy(d_materials, &materials, NUM_MATERIALS * sizeof(Scene::LambertMaterial), cudaMemcpyHostToDevice));


	#define NUM_PLANES 1
	Scene::Plane planes[NUM_PLANES];
	planes[0] = {make_float3(0.0f, -6.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), 0u}; // Front

	cutilSafeCall(cudaMalloc(&d_planes, NUM_PLANES * sizeof(Scene::Plane)));
	cutilSafeCall(cudaMemcpy(d_planes, &planes, NUM_PLANES * sizeof(Scene::Plane), cudaMemcpyHostToDevice));


	#define NUM_RECTANGLES 0
	#define NUM_CIRCLES 0


	#define NUM_SPHERES 9
	Scene::Sphere spheres[NUM_SPHERES];
	spheres[0] = {make_float3(0.0f, 0.0f, 0.0f), 4.0f, 4u};
	spheres[1] = {make_float3(-4.0f, -4.0f, -4.0f), 4.0f, 1u};
	spheres[2] = {make_float3(-4.0f, -4.0f, 4.0f), 4.0f, 2u};
	spheres[3] = {make_float3(-4.0f, 4.0f, -4.0f), 4.0f, 3u};
	spheres[4] = {make_float3(-4.0f, 4.0f, 4.0f), 4.0f, 5u};
	spheres[5] = {make_float3(4.0f, -4.0f, -4.0f), 4.0f, 5u};
	spheres[6] = {make_float3(4.0f, -4.0f, 4.0f), 4.0f, 3u};
	spheres[7] = {make_float3(4.0f, 4.0f, -4.0f), 4.0f, 2u};
	spheres[8] = {make_float3(4.0f, 4.0f, 4.0f), 4.0f, 1u};

	cutilSafeCall(cudaMalloc(&d_spheres, NUM_SPHERES * sizeof(Scene::Sphere)));
	cutilSafeCall(cudaMemcpy(d_spheres, &spheres, NUM_SPHERES * sizeof(Scene::Sphere), cudaMemcpyHostToDevice));


	// Store the representation of the scene in a single object
	// CUDA only allows 256 bytes of data to be passed as arguments in the kernel launch
	// We can save some room by bundling all the scene variables together
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


	// Set the camera
	camera = Camera(-3.14,3.14/2, 30.0f);


	// Set the exposure
	exposure = 1.0f;
}
