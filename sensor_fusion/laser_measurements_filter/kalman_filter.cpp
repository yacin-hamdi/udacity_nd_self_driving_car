#include "kalman_filter.h"


KalmanFilter::KalmanFilter() {

}

KalmanFilter::~KalmanFilter() {

}

void KalmanFilter::predict() {
	x_ = F_ * x_;
	MatrixXd F_t = F_.transpose();
	P_ = F_ * P_ * F_t + Q_;
}

void KalmanFilter::update(const VectorXd& z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd H_t = H_.transpose();
	MatrixXd S = H_ * P_ * H_t + R_;
	MatrixXd S_i = S.inverse();
	MatrixXd k = P_ * H_t * S_i;


	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);

	x_ = x_ + k * y;
	P_ = (I - k * H_) * P_;

}

