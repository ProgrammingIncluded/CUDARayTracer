#pragma once

#include <SFML/Graphics.hpp>

class Camera
{
	public:
		Camera();
		~Camera();
	
	protected:
		
	private:
		sf::Vector3f pos;
		
		// Angles
		float theta;
		float phi;
		
};

