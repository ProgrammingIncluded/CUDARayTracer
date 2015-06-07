#ifndef STATEMANAGER_H
#define STATEMANAGER_H

#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>

#include <stack>
#include "State.h"

typedef unsigned int GLuint;

/**
* Generic StateManager class. Used for managing a State class.
*/
class StateManager
{
	public:
		StateManager();

		// Function that should be called to update.
		void update();

		void input(sf::Event event);

		void draw(uchar4* canvasTexture, float time);

		/// Pushes the State into the stack. If first State, starts automatically.
		/// Returns false if null.
		bool pushState(State* state);
		/// Returns false if empty. Pops current state and starts the next one
		/// if applicable.
		bool popState();
		/// Returns true if empty.
		bool isEmpty();
		/// Returns amount of states in manager.
		int stateCount();

		/// Call to collect and delete the states. Alternative to heavy smart pointers.
		void garbageCollect();

	private:
		std::stack < State* > states;
		// Smart pointer alternative by garbage collecting.
		// Deleted each update cycle. 
		// TODO: Implement better auto collect. I.e. smarter pointers.
		std::stack < State* > deletedStates;
};


#endif // STATEMANAGER_H 

