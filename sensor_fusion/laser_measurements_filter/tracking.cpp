#include "tracking.h"
#include<iostream>
#include<math.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;

Tracking::Tracking() {
	_is_initialized = false;
	_previous_timestamp = 0;

	//state vector
	kf_.x_ = VectorXd(4);

	//state covariance matrix
	kf_.P_ = MatrixXd(4, 4);

	kf_.P_ << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000;

	//measurement covariance
	kf_.R_ = MatrixXd(2, 2);
	kf_.R_ << 0.0225, 0,
		0, 0.0225;

	//measurement matrix
	kf_.H_ = MatrixXd(2, 4);
	kf_.H_ << 1, 0, 0, 0,
		0, 1, 0, 0;

	//state transition matrix
	kf_.F_ = MatrixXd(4, 4);
	kf_.F_ << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1;
	kf_.Q_ = MatrixXd(4, 4);

	//acceleration noise component
	_noise_ax = 5;
	_noise_ay = 5;

	
}


Tracking::~Tracking() {

}

void Tracking::processMeasurement(const MeasurementPackage& measurementPackage) {
	if (_is_initialized) {
		std::cout << "kalman filter initialization" << std::endl;

		kf_.x_ << measurementPackage.raw_measurements_[0], measurementPackage.raw_measurements_[1], 0, 0;
		_previous_timestamp = measurementPackage.timestamp_;
		_is_initialized = true;
		return;
	}

	float dt = (measurementPackage.timestamp_ - _previous_timestamp) / 1000000.0;
	_previous_timestamp = measurementPackage.timestamp_;

	// TODO: YOUR CODE HERE
	//1. Modify the F matrix so that the time is integrated
	
	kf_.F_(0, 2) = dt;
	kf_.F_(1, 3) = dt;
	//2. Set the process covariance matrix Q

	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;
	
	kf_.Q_ << _noise_ax * dt_4 / 4, 0, _noise_ax* dt_3 / 2, 0,
		0, _noise_ay* dt_4 / 4, 0, _noise_ay* dt_3 / 2,
		_noise_ax* dt_3 / 2, 0, _noise_ax* dt_2, 0,
		0, _noise_ay* dt_3 / 2, 0, _noise_ay* dt_2;

	//3. Call the Kalman Filter predict() function
	kf_.predict();
	//4. Call the Kalman Filter update() function
	// with the most recent raw measurements_
	kf_.update(measurementPackage.raw_measurements_);

	std::cout << "x_= " << kf_.x_ << std::endl;
	std::cout << "P_= " << kf_.P_ << std::endl;
}