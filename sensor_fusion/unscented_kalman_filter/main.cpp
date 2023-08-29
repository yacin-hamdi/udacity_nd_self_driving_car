#include<iostream>
#include"ukf.h"
#include"Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main() {
	UKF ukf;
	MatrixXd Xsig(5, 11);
	ukf.AugmentedSigmaPoints(&Xsig);

	std::cout << "Xsig=" << std::endl << Xsig << std::endl;

	return 0;
}