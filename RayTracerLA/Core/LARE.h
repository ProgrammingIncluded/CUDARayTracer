#ifndef	LARE_H
#define LARE_H

#include <GL/glew.h>
#include <windows.h>
#include <GL/GL.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "Core/RCS.cuh"
#include "Core/States/StateManager.h"

class LARE : public StateManager
{
	public:
		LARE();
		// Screen Size Constructor
		LARE(int size);
		~LARE();

		/**
		* Call function in order to run the application.
		*/
		void run();

		sf::RenderWindow* getWindow();

	private:
		// Render Window from SFML.
		sf::RenderWindow* window;
		// Since we only want one instance, it is much more safer
		// to in dynamic memory.
		RCS* renderCanvas;
};

#endif // LARE_H