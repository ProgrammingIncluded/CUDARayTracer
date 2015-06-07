#include "LARE.h"


LARE::LARE()
{
	window = new sf::RenderWindow(sf::VideoMode(256,256), "OpenGL", sf::Style::Close);
	window->resetGLStates();
	glewInit();
	renderCanvas = new RCS(window->getSize());
}


LARE::~LARE()
{
	delete renderCanvas;
}

void LARE::run()
{
	sf::Clock time;
	while (isEmpty() != true)
	{
		sf::Event event;
		while (window->pollEvent(event))
		{
			if (event.type == sf::Event::Resized)
			{
				sf::Vector2u size = window->getSize();
				renderCanvas->resize(size.x, size.y);
			}
			// Call every time as input may change event etc. 
			this->input(event);
		}
		// Update any logic.
		this->update();

		window->clear();

		renderCanvas->renderScene();

		// Draw to canvas.
		renderCanvas->startCUDALockCanvas();
		this->draw(renderCanvas->getDrawCanvasPointer(), time.getElapsedTime().asMilliseconds());
		renderCanvas->endCUDALockCanvas();
		renderCanvas->drawFrame();
		
		window->display();
		this->garbageCollect();
	}
}

sf::RenderWindow* LARE::getWindow()
{
	return window;
}