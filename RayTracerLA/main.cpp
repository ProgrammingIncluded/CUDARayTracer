#include "Core/LARE.h"
#include "States/Titlestate.cuh"

int main()
{
	LARE app;
	app.pushState(TitleState::create(&app, app.getWindow()));
	app.run();

	return 0;
}