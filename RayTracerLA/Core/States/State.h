#ifndef STATE_H
#define STATE_H

class StateManager;
class sf::Event;
class sf::RenderWindow;

/**
* State that should be extended to create a generic state for StateManager.
* Must incorporate a factor class in order to create this class.
*/
class State
{
	public:
		/// Function that is called whenever an input is given.
		virtual void input(sf::Event &event) = 0;

		/// Function that is called every update loop.
		virtual void update() = 0;

		virtual void draw(uchar4* canvas, float time) = 0;

		/// Function that is called when a new state is pushed uptop.
		/// Essentially when state is put on hold.
		virtual void pause() = 0;

		/// Function that is called when state goes to the top of the stack
		/// after being pushed down.
		virtual void resume() = 0;

		/// Function that is called once and first.
		virtual void setUp() = 0;

		/// Function called when state is exiting, i.e. popped.
		virtual void end() = 0;

	protected:
		StateManager* stateManager;
		sf::RenderWindow* window;

		// Must use a Factory design method.
		State(StateManager* sm, sf::RenderWindow* rw)
		{
			stateManager = sm;
			window = rw;
		};
		State(const State& other)
		{
			stateManager = other.stateManager;
			window = other.window;
		};
		State& operator =(const State& other)
		{
			return *this;
		};
};
#endif // STATE_H 