#include "TraceState.cuh"

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

__global__ void clearTexture(uchar4* pos)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	pos[index].x = 0;
	pos[index].y = 0;
	pos[index].z = 0;
	pos[index].w = 255.0f;
}

void TraceState::input(sf::Event &event)
{
	if (event.type == sf::Event::Closed)
	{
		stateManager->exit();
		stateManager->popState();
	}
	else if (event.type == sf::Event::KeyPressed)
	{
		cameraMoved = true;
		if (event.key.code == sf::Keyboard::Escape)
		{
			stateManager->exit();
			stateManager->popState();
		}
		else if (event.key.code == sf::Keyboard::O)
			isWeighted = !isWeighted;
		else if (event.key.code == sf::Keyboard::N)
			stateManager->popState();
		else if (event.key.code == sf::Keyboard::P)
			pauseRender = !pauseRender;
		else if (event.key.code == sf::Keyboard::W)
			camera.zoom(1.0f);
		else if (event.key.code == sf::Keyboard::A)
			camera.pan(-1.0f, 0.0f);
		else if (event.key.code == sf::Keyboard::D)
			camera.pan(1.0f, 0);
		else if (event.key.code == sf::Keyboard::S)
			camera.zoom(-1.0f);
		else if (event.key.code == sf::Keyboard::Up)
			camera.rotate(0.0f, 3.14f / 100);
		else if (event.key.code == sf::Keyboard::Down)
			camera.rotate(0.0f, -3.14f / 100);
		else if (event.key.code == sf::Keyboard::Left)
			camera.rotate(-1.07f / 100, 0);
		else if (event.key.code == sf::Keyboard::Right)
			camera.rotate(1.07f / 100, 0);

	}
	else if (event.type == sf::Event::MouseButtonPressed && mouseBuffer == false)
	{
		mouseBuffer = true;
		prevMousePos = sf::Mouse::getPosition();
	}
	else if (event.type == sf::Event::MouseButtonReleased && mouseBuffer == true)
	{
		sf::Vector2i mousePos = sf::Mouse::getPosition();
		mouseBuffer = false;
		float dPhi = ((float)(prevMousePos.x - mousePos.x) / 300);
		float dTheta = ((float)(prevMousePos.y - mousePos.y) / 300);

		camera.rotate(-dTheta, dPhi);
		cameraMoved = true;
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
	if (cameraMoved == true)
	{
		cudaDeviceSynchronize();
		updateCameraData(d_cameraData);
		clearTexture << <(windowSize.x*windowSize.y) / 256, 256 >> >(canvas);
		cameraMoved = false;
	}
	pathTraceNextFrame(canvas, windowSize.x, windowSize.y, d_cameraData, d_sceneObjects,isWeighted, ++frameCount);
}

void TraceState::pause()
{

}

void TraceState::resume()
{

}

void TraceState::setUp()
{
	cameraMoved = true;
	pauseRender = false;
	mouseBuffer = false;
	isWeighted = false;
	prevMousePos = sf::Vector2i(0,0);
	frameCount = 0;

	setUpScene();
	cutilSafeCall(cudaMalloc(&d_cameraData, sizeof(CameraData)));
	sf::Vector2u size = window->getSize();
	camera.setProjection(1.04719755f, (float)size.x/ size.y);
	updateCameraData(d_cameraData);
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