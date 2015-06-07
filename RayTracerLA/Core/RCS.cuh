#ifndef	RCS_CUH
#define RCS_CUH

#include <GL/glew.h>
#include <windows.h>
#include <GL/GL.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <iostream>

#include "cutil.h"
#include "cutil_inline_runtime.h"
#include "GeneralTypedef.h"

class RCS
{
	public:
		RCS(sf::Vector2u windowSize);
		~RCS();

		void resize(int width, int height);

		void drawFrame();
		void renderScene();

		uchar4* getDrawCanvasPointer();

		// Call these functions to allow cuda calls on canvas.
		void startCUDALockCanvas();
		void endCUDALockCanvas();

	private:
		// Canvas pointer values.
		cudaArray* cudaTexArraySource;
		cudaArray* cudaTexArrayResult;

		GLuint glRenderTexture;
		GLuint glFinalTexture;
		GLuint glpbo;
		GLuint gldbo;
		GLuint glfbo;
		// Cuda
		cudaGraphicsResource_t* resources;
		cudaGraphicsResource_t cudaRenderTexture;
		cudaGraphicsResource_t cudaFinalTexture;

		uchar4* cudaAllocResultTexture;
		size_t cudaAllocResultTexture_size;

		sf::Vector2u windowSize;

		void createTexture(GLuint &texture, unsigned int width, unsigned int height);
		void deleteTexture(GLuint &texture);

		void createPBO(GLuint &bufferID, size_t size);
		void deletePBO(GLuint& bufferID);

		void createDepthBuffer(GLuint& depthBuffer, unsigned int width, unsigned int height);
		void deleteDepthBuffer(GLuint& depthBuffer);

		void createFrameBuffer(GLuint& framebuffer, GLuint colorAttachment0, GLuint depthAttachment);
		void deleteFrameBuffer(GLuint& framebuffer);

		void createCUDAResource(cudaGraphicsResource_t& cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags);
		void deleteCUDAResource(cudaGraphicsResource_t& cudaResource);

		void setupOpenGL();
		void setupCUDA();
		void setupMatrix();
};

#endif // RCS_CUH