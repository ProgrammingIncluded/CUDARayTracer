#ifndef	OPENGLTEST_CUH
#define OPENGLTEST_CUH

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

class OpenGLTest
{
	public:
		GLuint glRenderTexture;
		GLuint glFinalTexture;
		GLuint glpbo;
		GLuint gldbo;
		GLuint glfbo;
		// Cuda
		cudaGraphicsResource_t cudaRenderTexture;
		cudaGraphicsResource_t cudaFinalTexture;

		uchar4* cudaAllocResultTexture;
		size_t cudaAllocResultTexture_size;

		sf::Vector2u windowSize;
	
		OpenGLTest(sf::Vector2u windowSize)
		{
			glRenderTexture = 0;
			glFinalTexture = 0;
			glpbo = 0;
			gldbo = 0;
			glfbo = 0;
			cudaRenderTexture = 0;
			cudaFinalTexture = 0;
			cudaAllocResultTexture = 0;
			cudaAllocResultTexture_size = 0;

			this->windowSize = sf::Vector2u(windowSize);
			this->setupCUDA();
			this->setupOpenGL();
			//this->setupMatrix();
			resize(windowSize.x, windowSize.y);
		};

		~OpenGLTest()
		{
			deletePBO(glpbo);
			deleteFrameBuffer(glfbo);
			deleteDepthBuffer(gldbo);
			deleteTexture(glRenderTexture);
			deleteTexture(glFinalTexture);
			deleteCUDAResource(cudaRenderTexture);
			deleteCUDAResource(cudaFinalTexture);
		}

		void resize(int width, int height);

		void drawFrame();
		void createFrame(float time);
		void renderScene();
	private:
		//OpenGL Functions.
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
#endif //OPENGLTEST_CUH