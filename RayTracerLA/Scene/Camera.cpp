#include "Camera.h"

#define PI 3.141592654f
#define TWO_PI 6.283185307f

Camera::Camera()
{
	setCamera(theta, phi, radius);
}

Camera::Camera(float theta, float phi, float radius, sf::Vector3f target)
{
	setCamera(theta, phi, radius, target);
}


Camera::~Camera()
{

}

void Camera::setCamera(float theta, float phi, float radius, sf::Vector3f target)
{
	setTheta(theta);
	setPhi(phi);
	setRadius(radius);
	setTarget(target);
}

bool Camera::setPhi(float theta)
{
	this->theta += std::fmod(theta, TWO_PI) * TWO_PI;
	setUpFromPhi();
	return true;
}

bool Camera::setTheta(float theta)
{
	this->theta = theta;
	return true;
}

bool Camera::setTarget(sf::Vector3f target)
{
	this->target = target;
	return true;
}

bool Camera::setRadius(float radius)
{
	this->radius = radius;
	return true;
}

void Camera::rotate(float deltaTheta, float deltaPhi)
{
	if (up > 0.0f)
		theta += deltaTheta;
	else
		theta -= deltaTheta;
	setPhi(phi + deltaPhi);
	setUpFromPhi();
}

bool Camera::setProjection(float fov, float aspectRatio)
{
	tanFovXDiv2 = tan(fov * 0.5f);
	tanFovYDiv2 = tan(fov * 0.5f) / aspectRatio;
	return true;
}

void Camera::setUpFromPhi()
{
	if ((phi > 0 && phi < PI) || (phi < -PI && phi > -TWO_PI)) {
		up = 1.0f;
	}
	else {
		up = -1.0f;
	}
}

float Camera::getTheta()
{
	return theta;
}

float Camera::getPhi()
{
	return phi;
}

float Camera::getRadius()
{
	return radius;
}

sf::Vector3f Camera::getTarget()
{
	return target;
}

sf::Vector3f Camera::getPosition()
{
	float x = radius * sinf(phi) * sinf(theta);
	float y = radius * cosf(phi);
	float z = radius * sinf(phi) * cosf(theta);

	return sf::Vector3f(x,y,z) + target;
}

float Camera::getUp()
{
	return up;
}

float Camera::getTanFovXDiv2()
{
	return tanFovXDiv2;
}

float Camera::getTanFovYDiv2()
{
	return tanFovYDiv2;
}