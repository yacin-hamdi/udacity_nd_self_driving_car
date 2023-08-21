#include<iostream>
#include "Eigen/Dense"


Eigen::MatrixXd calculateJacobian(const Eigen::VectorXd& x_state);

int main() {
	Eigen::VectorXd x_predicted(4);
	x_predicted << 1, 2, 0.2, 0.4;
	Eigen::MatrixXd Hj = calculateJacobian(x_predicted);

	std::cout << "Hj=" << std::endl << Hj;

	

	return 0;
}

Eigen::MatrixXd calculateJacobian(const Eigen::VectorXd& x_state) {
	Eigen::MatrixXd Hj(3, 4);

	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float c = px * px + py * py;
	if (fabs(c) < 0.0001) {
		std::cout << "[-] error division by 0!!" << std::endl;
		return Hj;
	}

	
	float d_rho_px = px / sqrt(c);
	float d_rho_py = py / sqrt(c);

	float d_phi_px = -(py / (c));
	float d_phi_py = px / (c);

	float d_rho_dot_px = py * (vx * py - vy * px) / pow(c, 3 / 2);
	float d_rho_dot_py = px * (vy * px - vx * py) / pow(c, 3 / 2);
		
	Hj << d_rho_px, d_rho_py, 0, 0,
		d_phi_px, d_phi_py, 0, 0,
		d_rho_dot_px, d_rho_dot_py, d_rho_px, d_rho_py;

	

	return Hj;

}