#ifndef STATEFACTORY_H
#define STATEFACTORY_H

#include <SFML/Graphics.hpp>

class StateManager;

/**
* The factor of the Factor Design method.
* Allows instantiation of States without Constructor
* dependency.
*/
template <class implementClass, class interfaceClass>
class StateFactory
{
	public:
		/// For BootState/State related classes.
		static interfaceClass* create(StateManager* sm, sf::RenderWindow* rw)
		{
			return implementClass::createInternal(sm, rw);
		};
};
#endif // STATEFACTORY_H 

