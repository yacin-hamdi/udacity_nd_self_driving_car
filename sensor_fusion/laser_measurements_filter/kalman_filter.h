#ifndef KALMAN_FILTER_H
#define KALMNA_FILTER_H

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;

class KalmanFilter {
public:
	//state vector
	VectorXd x_;

	//state covariance vector
	MatrixXd P_;

	//state transition matrix
	MatrixXd F_;

	//measurement matrix
	MatrixXd H_;

	//measurement covariance matrix
	MatrixXd R_;

	//process covariance matrix
	MatrixXd Q_;

	KalmanFilter();
	virtual ~KalmanFilter();

	void predict();
	void update(const VectorXd& z);


};
#endif

