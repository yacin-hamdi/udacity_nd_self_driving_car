#include<iostream>
#include "Eigen/Dense"
#include<vector>

//Kalman Filter Variables

Eigen::VectorXd x; // Object State
Eigen::MatrixXd P; // Object Covariance Matrix
Eigen::VectorXd u; // External Motion
Eigen::MatrixXd F; // State Transition Matrix
Eigen::MatrixXd H; // Measurements Matrix
Eigen::MatrixXd R; // Measurement Covariance Matrix
Eigen::MatrixXd I; // Identity Matrix
Eigen::MatrixXd Q; // Process Covariance Matrix

std::vector<Eigen::VectorXd> measurements;

void kalmanFilter(Eigen::VectorXd& x, Eigen::MatrixXd& P);


int main() {

	/*Eigen::VectorXd my_vector(3);
	my_vector << 1, 2, 3;
	std::cout << my_vector << std::endl;

	Eigen::MatrixXd my_matrix(3, 3);
	my_matrix << 1, 2, 3,
		4, 5, 6, 
		7, 8, 9;
	std::cout << my_matrix << std::endl;

	Eigen::MatrixXd result_matrix = my_matrix * my_vector;
	std::cout << result_matrix << std::endl;
	std::cout << my_matrix.transpose() << std::endl;
	std::cout << my_matrix.inverse() << std::endl;*/

	// Design kalman filter with 1D motion
	x = Eigen::VectorXd(2);
	x << 0, 0;

	P = Eigen::MatrixXd(2, 2);
	P << 1000, 0, 
		0, 1000;

	u = Eigen::VectorXd(2);
	u << 0, 0;

	F = Eigen::MatrixXd(2, 2);
	F << 1, 1,
		0, 1;

	H = Eigen::MatrixXd(1, 2);
	H << 1, 0;
	
	I = Eigen::MatrixXd(2, 2);
	I << 1, 0,
		0, 1;

	R = Eigen::MatrixXd(1, 1);
	R << 1;

	Q = Eigen::MatrixXd(2, 2);
	Q << 0, 0,
		0, 0;
	Eigen::VectorXd single_meas(1);
	single_meas << 1;
	measurements.push_back(single_meas);
	single_meas << 2;
	measurements.push_back(single_meas);
	single_meas << 3;
	measurements.push_back(single_meas);
	single_meas << 3;
	measurements.push_back(single_meas);
	single_meas << 3;
	measurements.push_back(single_meas);
	single_meas << 3;
	measurements.push_back(single_meas);

	kalmanFilter(x, P);
	
	
	return 0;
}


void kalmanFilter(Eigen::VectorXd& x, Eigen::MatrixXd& P) {

	for (unsigned int i = 0; i < measurements.size(); i++) {
		Eigen::VectorXd z = measurements[i];
		//YOUR CODE HERE
		
		// KF Measurement update step
		Eigen::VectorXd y = z - (H * x);
		Eigen::MatrixXd Ht = H.transpose();
		Eigen::MatrixXd S = H * P * Ht + R;
		Eigen::MatrixXd Si = S.inverse();
		Eigen::MatrixXd K = P * Ht * Si;

		
		// new state
		x = x + (K * y);
		P = (I - (K * H)) * P;
		

		// KF Prediction step

		x = F * x + u;
		P = (F * P) * F.transpose() + Q;

		
		std::cout << "x=" << std::endl << x << std::endl;
		std::cout << "P=" << std::endl << P << std::endl;
	}

}