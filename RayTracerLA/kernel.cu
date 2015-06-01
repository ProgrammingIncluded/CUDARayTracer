
#include <iostream>
#include <ctime>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include "Matrix/MatrixN.h"

int main()
{
	float test[9] = 
	{   1.0f, 1.0f, 1.0f,
		1.0f, 2.0f, 1.0f,
		1.0f, 1.0f, 1.0f};


	// GLM TEST
	const clock_t tst_time = clock();
	glm::mat3 mat = glm::make_mat3(test);
	glm::mat3 mat2 = glm::make_mat3(test);

	glm::mat3 result = mat*mat2;
	std::cout << float(clock() - tst_time) << std::endl;

	// CUDA Matrix Test
	mat::MatrixN* matN = new mat::MatrixN(3);
	matN->setValue(test, 3);
	matN->allocateGPUMemory();

	mat::MatrixN* matB = new mat::MatrixN(3);
	matB->setValue(test,3);
	matB->allocateGPUMemory();

	mat::MatrixN* matR = new mat::MatrixN(3);
	matR->setValue(test, 3);
	matR->allocateGPUMemory();

	std::cout << "Before:" << std::endl;
	std::cout << *matN;
	const clock_t begin_time = clock();
	matN->mult(matB, matR);
	std::cout << float(clock() - begin_time) << std::endl;
	
	matR->copyGPUValue();
	std::cout << "After:" << std::endl;
	std::cout << *matR;

	// Delete testing.
	delete matN;
	delete matB;
	delete matR;

	std::cin.ignore();
}