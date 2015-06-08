#include <iostream>
#include "Core/LARE.h"
#include "States/Titlestate.cuh"

int main()
{
	std::cout << "Please type display size." << std::endl;
	int value;
	std::cin >> value;
	LARE app(value);
	app.pushState(TitleState::create(&app, app.getWindow()));
	app.run();

	return 0;
}