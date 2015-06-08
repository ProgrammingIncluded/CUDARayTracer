#include "TitleState.cuh"

void TitleState::input(sf::Event &event)
{
	
}

void TitleState::update()
{
	if (switcher == false)
	{
		stateManager->pushState(CornellState::create(stateManager, window));
	}
	else
	{
		stateManager->pushState(BallState::create(stateManager,window));
	}
	switcher = !switcher;
}

void TitleState::draw(uchar4* canvas, float time)
{
	
}

void TitleState::pause()
{

}

void TitleState::resume()
{

}

void TitleState::setUp()
{
	switcher = false;
}

void TitleState::end()
{

}

State* TitleState::createInternal(StateManager *sm, sf::RenderWindow *rw)
{
	return new TitleState(sm, rw);
}