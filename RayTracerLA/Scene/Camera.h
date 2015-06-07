#ifndef CAMERA_H
#define CAMERA_H

#include <SFML/Graphics.hpp>
#include <math.h>  

#include "Core/helper_math.h"
#include "Core/SCMath.h"
#include "CameraData.h"
// TODO, use Eigen matrix library.
class Camera
{
	public:
		Camera();
		Camera(float theta, float phi, float radius, sf::Vector3f target = sf::Vector3f(0, 0, 0));
		~Camera();

		// Easy function to call all setters.
		void setCamera(float theta, float phi, float radius, sf::Vector3f target = sf::Vector3f(0,0,0));

		void rotate(float deltaTheta, float deltaPhi);

		bool setTheta(float theta);
		bool setPhi(float phi);
		bool setRadius(float radius);

		bool setProjection(float fov, float aspectRatio);
		bool setTarget(sf::Vector3f target);

		float getTheta();
		float getPhi();
		float getRadius();

		sf::Vector3f getTarget();
		sf::Vector3f getPosition();

		// Put this somewhere else? Call matrix update manually.
		void updateCameraData(CameraData &data);

	private:
		void setUpFromPhi();

		// Angles
		float theta;
		float phi;
		float radius;
		float up; // Calculated from phi.

		float tanFovXDiv2;
		float tanFovYDiv2;

		sf::Vector3f target;

};

#endif // CAMERA_H
