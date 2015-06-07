#ifndef	TITLESTATE_H
#define TITLESTATE_H

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

#include "Core/States/StateFactory.h"
#include "Core/States/StateManager.h"
#include "Core/States/State.h"

#include "GeneralTypedef.h"


class TitleState : public StateFactory<TitleState, State>, public State
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

	protected:
		TitleState(StateManager* sm, sf::RenderWindow* window) : State(sm, window){};
		TitleState(const TitleState& other) : State(other){};
		TitleState& operator =(const State& other){return *this;};
};

#endif // TITLESTATE_H