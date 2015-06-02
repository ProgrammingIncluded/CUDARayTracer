#pragma once
#include <GL/glew.h>
#include <windows.h>
#include <GL/GL.h>
#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/System.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "GeneralTypedef.h"

uchar4* image;
GLuint gltexture;
GLuint pbo;
cudaGraphicsResource_t cudaPBO;
uchar4* d_textureBufferData;

sf::Vector2u windowSize;

void drawFrame();
void createFrame();
void setupOpenGL();


int main()
{
	// create the window
	sf::RenderWindow window(sf::VideoMode(800, 800), "OpenGL", sf::Style::Default, sf::ContextSettings(32));
	window.setVerticalSyncEnabled(true);

	windowSize = sf::Vector2u(window.getSize());

	bool running = true;
	glewInit();
	std::printf("OpenGL: %s:", glGetString(GL_VERSION));
	setupOpenGL();
	// We will not be using SFML's gl states.
	window.resetGLStates();
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
				// adjust the viewport when the window is resized
				glViewport(0, 0, event.size.width, event.size.height);
				windowSize = window.getSize();
			}
		}

		// clear the buffers
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		createFrame();
		drawFrame();
		window.display();
	}

	// release resources...

	return 0;
}