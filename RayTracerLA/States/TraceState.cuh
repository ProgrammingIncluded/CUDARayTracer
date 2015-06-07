#ifndef	TRACESTATE_H
#define TRACESTATE_H
#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

#include "Core/States/StateFactory.h"
#include "Core/States/StateManager.h"
#include "Core/States/State.h"

#include "Core/GeneralTypedef.h"
#include "Core/cutil.h"
#include "Core/cutil_inline_runtime.h"

#include "Scene/Materials.h"
#include "Scene/SceneObjects.h"
#include "Scene/Camera.h"

#include "PathTracer.cuh"


class TraceState : public StateFactory<TraceState, State>, public State
{
	public:
		/// Function that is called whenever an input is given.
		/// Input will not be none. Functionc called first before update().
		void input(sf::Event &event);

		/// Function that is called every update loop.
		void update();

		void draw(uchar4* canvas, float time);

		/// Function that is called when a new state is pushed uptop.
		/// Essentially when state is put on hold.
		void pause();

		/// Function that is called when state goes to the top of the stack
		/// after being pushed down.
		void resume();

		/// Function that is called once and first.
		void setUp();

		/// Function called when state is exiting, i.e. popped.
		void end();

		/// Function that should be called to instantiate the class.
		static State* createInternal(StateManager *sm, sf::RenderWindow *rw);

	private:
		void updateCameraData(CameraData* dataPtr);
		void setUpScene();

		Camera camera;
		float exposure;
		float frameCount;
		bool cameraMoved;
		bool pauseRender;

		// Pointer to GPU objects.
		Scene::SceneObjects* d_sceneObjects;

		Scene::LambertMaterial* d_materials;
		Scene::Plane* d_planes;
		Scene::Sphere* d_spheres;
		Scene::Rectangle* d_rectangles;
		Scene::Circle* d_circles;

		CameraData* d_cameraData;
	protected:
		TraceState(StateManager* sm, sf::RenderWindow* window) : State(sm, window){};
		TraceState(const TraceState& other) : State(other){};
		TraceState& operator =(const State& other){ return *this; };
};


#endif // TRACESTATE_H