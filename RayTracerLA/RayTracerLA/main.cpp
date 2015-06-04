#include <GL/glew.h>
#include <windows.h>
#include <GL/GL.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/System.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include "GeneralTypedef.h"
#include "OpenGLTest.cuh"


int main()
{
	// create the window
	sf::RenderWindow window(sf::VideoMode(1024, 1024), "OpenGL", sf::Style::Close);
	//window.setVerticalSyncEnabled(true);
	sf::Vector2u windowSize;

	windowSize = sf::Vector2u(window.getSize());

	bool running = true;
	window.resetGLStates();
	glewInit();
	std::printf("OpenGL: %s:", glGetString(GL_VERSION));
	// We will not be using SFML's gl states.

	OpenGLTest* test = new OpenGLTest(window.getSize());

	sf::Clock clock;

	while (running)
	{
		// handle events
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				// end the program
				running = false;
			}
			else if (event.type == sf::Event::Resized)
			{
				windowSize = window.getSize();
				test->resize(windowSize.x, windowSize.y);
			}
		}

		// clear the buffers
		window.clear();
		test->renderScene();
		test->createFrame(clock.getElapsedTime().asMilliseconds()/10);
		test->drawFrame();
		window.display();
	}

	// release resources...
	delete test;

	return 0;
}

