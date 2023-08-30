#include<iostream>
#include"ukf.h"
#include"Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

int main() {
	UKF ukf;
	MatrixXd Xsig(15, 5);
	ukf.SigmaPointPrediction(&Xsig);


	return 0;
}