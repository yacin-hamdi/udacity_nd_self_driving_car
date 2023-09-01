#include<iostream>
#include"ukf.h"
#include"Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main() {
	UKF ukf;
	VectorXd x(5);
	MatrixXd P(5, 5);
	ukf.UpdateState(&x, &P);


	return 0;
}