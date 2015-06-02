#include "OpenGLTest.cuh"

__global__ void createGPUTexture(uchar4* d_texture)
{
	uint pixelID = blockIdx.x*blockDim.x + threadIdx.x;
	
	uchar4 value;
	value.x = 150;
	value.y = 0;
	value.z = 0;
	value.w = 0;
		
	d_texture[pixelID].x = value.x;
	d_texture[pixelID].y = value.y;
	d_texture[pixelID].z =  value.z;
	d_texture[pixelID].w = value.w;
}

void drawFrame()
{
	glColor3f(1.0f,1.0f,1.0f);
	glBindTexture(GL_TEXTURE_2D, gltexture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowSize.x, windowSize.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(1.0f, -1.0f);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glEnd();
	
	// Release
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void createFrame()
{
	cudaGraphicsMapResources(1, &cudaPBO, 0);
	size_t numBytes;
	cudaGraphicsResourceGetMappedPointer((void**)&d_textureBufferData, &numBytes, cudaPBO);
	
	// Run code here.
	createGPUTexture << <(windowSize.x + windowSize.y)/256,  256>> >(d_textureBufferData);

	// Unmap mapping to PBO so that OpenGL can access.
	cudaGraphicsUnmapResources(1, &cudaPBO, 0);
}

void setupOpenGL()
{
	image  = new uchar4[800*800];

	// Unbind any textures from previous.
	glBindTexture(GL_TEXTURE_2D, 0);

	// Create new textures.
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gltexture);
	glBindTexture(GL_TEXTURE_2D, gltexture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Create image with same resolution as window.
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowSize.x , windowSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

	// Unbind the texture
	glBindTexture(GL_TEXTURE_2D, 0);

	// Create buffer boject.
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, windowSize.x * windowSize.y * sizeof(uchar4), image, GL_STREAM_COPY);

	cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsWriteDiscard);
}