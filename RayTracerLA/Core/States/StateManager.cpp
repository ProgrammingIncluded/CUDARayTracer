#include "StateManager.h"

StateManager::StateManager()
{

}

void StateManager::update()
{
	if (states.empty())
		return;

	// Update most recent object. Notice, if input deletes, 
	// next state will have update called first...
	if (!states.empty())
		states.top()->update();
}

void StateManager::input(sf::Event event)
{
	if (states.empty())
		return;

	states.top()->input(event);
}

void StateManager::draw(uchar4* canvasTexture, float time)
{
	if (states.empty())
		return;

	states.top()->draw(canvasTexture, time);
}

bool StateManager::pushState(State* state)
{
	// Null state, return and exist.
	if (state == 0)
		return false;

	// First object, setUp the state.
	if (!states.empty())
		states.top()->pause();

	states.push(state);
	state->setUp();

	return true;
}

bool StateManager::popState()
{
	int capacity = states.size();;

	if (capacity <= 0)
		return false;

	State* oldState = states.top();
	states.pop();
	oldState->end();
	deletedStates.push(oldState);

	// Resume the previous state if applicable.
	if (capacity > 1)
		states.top()->resume();

	return true;
}

bool StateManager::isEmpty()
{
	return states.empty();
}

void StateManager::garbageCollect()
{
	while (!deletedStates.empty())
	{
		delete deletedStates.top();
		deletedStates.pop();
	}
}

int StateManager::stateCount()
{
	return states.size();
}