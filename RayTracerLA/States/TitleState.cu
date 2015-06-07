#include "TitleState.cuh"

__global__ void titleDraw(uchar4* pos, unsigned int width, unsigned int height, float time)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int x = index%width;
	unsigned int y = index / width;

	if (index < width*height) {
		unsigned char r = (x + (int)time) & 0xff;
		unsigned char g = (y + (int)time) & 0xff;
		unsigned char b = ((x + y) + (int)time) & 0xff;

		// Each thread writes one pixel location in the texture (textel)
		pos[index] = make_uchar4(r, g, b, 255.0f);
	}
}

void TitleState::input(sf::Event &event)
{
	if (event.type == sf::Event::Closed)
	{
		stateManager->popState();
	}
}

void TitleState::update()
{

}

void TitleState::draw(uchar4* canvas, float time)
{
	sf::Vector2u windowSize = window->getSize();
	uint totalPixels = windowSize.x * windowSize.y;
	titleDraw <<<(totalPixels) / 256, 256 >> >(canvas, windowSize.x, windowSize.y, time/10);
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