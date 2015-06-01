
#include <iostream>
#include <ctime>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include "Matrix/MatrixN.h"

int main()
{
	float test[9] = { 8.0f, 3.0f, 44.0f,
		45.0f, 96.0f, 51.0f,
		22.0f, 55.0f, 5.0f };


	// GLM TEST
	const clock_t tst_time = clock();
	glm::mat3 mat = glm::make_mat3(test);
	glm::mat3 mat2 = glm::make_mat3(test);

	glm::mat3 result = mat*mat2;
	std::cout << float(clock() - tst_time) << std::endl;

	// CUDA Matrix Test
	const clock_t begin_time = clock();
	mat::MatrixN* matN = new mat::MatrixN(3);
	matN->setValue(test, 3);

	mat::MatrixN* matB = new mat::MatrixN(3);
	matB->setValue(test, 3);

	//std::cout << "Before:" << std::endl;
	//std::cout << *matN;
	


	// do something
	matN->mult(matB);
	std::cout << float(clock() - begin_time) << std::endl;

	std::cout << "After:" << std::endl;
	std::cout << *matN;

	std::cin.ignore();
}