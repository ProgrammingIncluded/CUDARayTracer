#include "TitleState.cuh"

void TitleState::input(sf::Event &event)
{
	
}

void TitleState::update()
{
	stateManager->popState();
	stateManager->pushState(TraceState::create(stateManager, window));
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

}

void TitleState::end()
{

}

State* TitleState::createInternal(StateManager *sm, sf::RenderWindow *rw)
{
	return new TitleState(sm, rw);
}