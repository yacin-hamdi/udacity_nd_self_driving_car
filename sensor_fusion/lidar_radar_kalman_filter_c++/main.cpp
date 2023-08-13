#include<iostream>
#include "Eigen/Dense"
#include<vector>

using namespace Eigen;


int main() {

	VectorXd my_vector(2);
	my_vector << 0, 1;
	std::cout << my_vector << std::endl;

	MatrixXd my_matrix(2, 2);
	my_matrix << 0, 1,
		2, 5;
	std::cout << my_matrix << std::endl;

	my_matrix(0, 0) = 10;
	my_matrix(0, 1) = 3;
	std::cout << my_matrix << std::endl;
	std::cout << my_matrix.inverse() << std::endl;

	MatrixXd result_matrix = my_matrix * my_vector;
	std::cout << result_matrix << std::endl;
	return 0;
}