
#include <iostream>
#include "MatrixN.h"

int main()
{
	mat::MatrixN* matN = new mat::MatrixN(3);
	float test[9] = { 8.0f, 3.0f, 44.0f, 
					  45.0f, 96.0f, 51.0f,
					  22.0f, 55.0f, 5.0f};
	matN->setValue(test, 3);

	mat::MatrixN* matB = new mat::MatrixN(3);
	matB->setValue(test, 3);



	std::cout << "Before:" << std::endl;
	std::cout << *matN;

	matN->mult(matB);

	std::cout << "After:" << std::endl;
	std::cout << *matN;

	std::cin.ignore();
}