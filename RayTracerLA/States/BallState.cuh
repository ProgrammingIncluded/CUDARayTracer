#ifndef	BALLSTATE_H
#define BALLSTATE_H

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

#include "Core/States/StateFactory.h"
#include "Core/States/StateManager.h"
#include "States/TraceState.cuh"

#include "Core/GeneralTypedef.h"
#include "Core/cutil.h"
#include "Core/cutil_inline_runtime.h"

#include "Scene/Materials.h"
#include "Scene/SceneObjects.h"
#include "Scene/Camera.h"

#include "PathTracer.cuh"

class BallState : public StateFactory<BallState, State>, public TraceState
{
public:
	/// Function that should be called to instantiate the class.
	static State* createInternal(StateManager *sm, sf::RenderWindow *rw)
	{
		return new BallState(sm, rw);
	};

	void end();

private:
	void setUpScene();

protected:
	BallState(StateManager* sm, sf::RenderWindow* window) : TraceState(sm, window){};
	BallState(const BallState& other) : TraceState(other){};
	BallState& operator =(const State& other){ return *this; };
};

#endif // BALLSTATE_H