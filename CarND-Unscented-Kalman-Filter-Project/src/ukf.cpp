#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
#include<iostream>
/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
	std::cout << "constructor" << std::endl;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.0;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  

  //state dimension
  n_x_ = 5;
  
  //augmented state dimension
  n_aug_ = 7;

  //define spreading parameter
  lambda_ = 3 - n_aug_;

  //set weights vector
  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
	  double weight = 0.5 / (n_aug_ + lambda_);
	  weights_(i) = weight;
  }

  //sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  is_initialized_ = false;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
	std::cout << "start process" << std::endl;
	if (!is_initialized_) {
		
		P_ << 0.3, 0, 0, 0, 0,
			0, 0.01, 0, 0, 0,
			0, 0, 0.7, 0, 0,
			0, 0, 0, 0.3, 0,
			0, 0, 0, 0, 0.7;


		if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;

			
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			double rho = meas_package.raw_measurements_[0];
			double phi = meas_package.raw_measurements_[1];

			double px = rho * cos(phi);
			double py = rho * sin(phi);

			if (px < 0.0001)
				px = 0.0001;
			if (py < 0.0001)
				py = 0.0001;
			x_ << px, py, 0, 0, 0;

			
		}

		std::cout << "x_ = " << x_ << std::endl;
		std::cout << "P_ = " << P_ << std::endl;
		is_initialized_ = true;
		time_us_ = meas_package.timestamp_;
	}

	float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;
	std::cout << "prediction" << std::endl;
	Prediction(delta_t);
	std::cout << "f prediction" << std::endl;
	if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
		UpdateLidar(meas_package);
	}
	else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		UpdateRadar(meas_package);
	}
	std::cout << "x_ = " << x_ << std::endl;
	std::cout << "P_ = " << P_ << std::endl;

}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
	VectorXd x_aug(n_aug_);
	MatrixXd P_aug(n_aug_, n_aug_);
	MatrixXd Xsig_aug(n_aug_, 2 * n_aug_ + 1);
	x_aug.fill(0.0);
	P_aug.fill(0.0);
	x_aug.head(5) = x_;
	P_aug.topLeftCorner(5, 5) = P_;
	P_aug(5, 5) = std_a_ * std_a_;
	P_aug(6, 6) = std_yawdd_ * std_yawdd_;
	MatrixXd A = P_aug.llt().matrixL();
	MatrixXd lam_A = sqrt(lambda_ + n_aug_) * A;
	Xsig_aug.col(0) = x_aug;
	for (int i = 0; i < n_aug_; i++) {
		Xsig_aug.col(i + 1) = x_aug + lam_A.col(i);
		Xsig_aug.col(i + n_aug_ + 1) = x_aug - lam_A.col(i);
	}
	

	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double px = Xsig_aug(0, i);
		double py = Xsig_aug(1, i);
		double v = Xsig_aug(2, i);
		double yaw = Xsig_aug(3, i);
		double yawd = Xsig_aug(4, i);
		double nu_a = Xsig_aug(5, i);
		double nu_yawdd = Xsig_aug(6, i);

		double px_pred;
		double py_pred;
		double pred_v = v;
		double pred_yaw = yaw + yawd * delta_t;
		double pred_yawd = yawd;
		if (fabs(yawd) < 0.001) {
			px_pred = px + v * cos(yaw) * delta_t;
			py_pred = py + v * sin(yaw) * delta_t;
		}
		else {
			px_pred = px + (v / yawd) * (sin(yaw + yawd * delta_t) - sin(yaw));
			py_pred = py + (v / yawd) * (cos(yaw) - cos(yaw + yawd * delta_t));
		}


		px_pred += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
		py_pred += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
		pred_v += delta_t * nu_a;
		pred_yaw += 0.5 * delta_t * delta_t * nu_yawdd;
		pred_yawd += delta_t * nu_yawdd;

		Xsig_pred_(0, i) = px_pred;
		Xsig_pred_(1, i) = py_pred;
		Xsig_pred_(2, i) = pred_v;
		Xsig_pred_(3, i) = pred_yaw;
		Xsig_pred_(4, i) = pred_yawd;

	}
	x_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		x_ += weights_(i) * Xsig_pred_.col(i);
		
	}
	P_.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		while (x_diff(3) > M_PI) {
			x_diff(3) -= 2 * M_PI;
			std::cout << "normilize" << std::endl;
		}
		while (x_diff(3) < -M_PI) {
			x_diff(3) += 2 * M_PI;
			std::cout << "normilize" << std::endl;
		}
		P_ += weights_(i) * x_diff * x_diff.transpose();
		
		
	}

	
	

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
	/**
	 * TODO: Complete this function! Use lidar data to update the belief
	 * about the object's position. Modify the state vector, x_, and
	 * covariance, P_.
	 * You can also calculate the lidar NIS, if desired.
	 */
	int n_z = 2;

	VectorXd z_(n_z);
	z_ << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1];
	MatrixXd Zsig(n_z, n_aug_ * 2 + 1);
	Zsig.fill(0.0);
	for (int i = 0; i < n_aug_ * 2 + 1; i++) {
		Zsig(0, i) = Xsig_pred_(0, i);
		Zsig(1, i) = Xsig_pred_(1, i);
	}



	VectorXd z_pred(n_z);
	z_pred.fill(0.0);

	//predict measurement mean
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred += weights_(i) * Zsig.col(i);
	}

	//predict covariance
	MatrixXd S(n_z, n_z);
	S.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;
		S += weights_(i) * z_diff * z_diff.transpose();
	}

	MatrixXd R(n_z, n_z);
	R << std_laspx_, 0,
		0, std_laspy_;

	S += R;

	//cross correlation matrix
	MatrixXd T(n_x_, n_z);
	T.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		VectorXd z_diff = Zsig.col(i) - z_pred;

		while (x_diff(3) > M_PI) {
			x_diff(3) -= 2 * M_PI;
			std::cout << "lidar normilize" << std::endl;
		}
		while (x_diff(3) < -M_PI) {
			x_diff(3) += 2 * M_PI;
			std::cout << "lidar normilize" << std::endl;
		}

		T += weights_(i) * x_diff * z_diff.transpose();
	}

	

	//kalman gain K
	MatrixXd K = T * S.inverse();

	//update state

	x_ += K * (z_ - z_pred);
	
	//update covariance matrix;
	P_ = P_ - (K * S * K.transpose());

	
	std::cout << "finish lidar" << std::endl;

}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
	int n_z = 3;
	VectorXd z_pred(n_z);
	MatrixXd Zsig(n_z, 2 * n_aug_ + 1);
	MatrixXd S(n_z, n_z);
	z_pred.fill(0.0);
	Zsig.fill(0.0);
	S.fill(0.0);

	//measurement model
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		double px = Xsig_pred_(0, i);
		double py = Xsig_pred_(1, i);
		double v = Xsig_pred_(2, i);
		double yaw = Xsig_pred_(3, i);
		double yawd = Xsig_pred_(4, i);

		double rho = sqrt(px * px + py * py);
		double phi = atan2(py, px);

		double rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v) / rho;

		Zsig(0, i) += rho;
		Zsig(1, i) += phi;
		Zsig(2, i) += rho_dot;

	}

	//predict measurement mean
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		z_pred += weights_(i) * Zsig.col(i);
	}

	//predict covariance
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd z_diff = Zsig.col(i) - z_pred;
		while (z_diff(1) > M_PI) z_diff(1) -= 2 * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2 * M_PI;

		S += weights_(i) * z_diff * z_diff.transpose();
	}

	MatrixXd R(3, 3);
	R << std_radr_ * std_radr_, 0, 0,
		0, std_radphi_* std_radphi_, 0,
		0, 0, std_radrd_* std_radrd_;

	S += R;

	//cross correlation matrix
	MatrixXd T(n_x_, n_z);
	T.fill(0.0);
	for (int i = 0; i < 2 * n_aug_ + 1; i++) {
		VectorXd x_diff = Xsig_pred_.col(i) - x_;
		VectorXd z_diff = Zsig.col(i) - z_pred;

		while (x_diff(3) > M_PI) x_diff(3) -= 2 * M_PI;
		while (x_diff(3) < -M_PI) x_diff(3) += 2 * M_PI;
		while (z_diff(1) > M_PI) z_diff(1) -= 2 * M_PI;
		while (z_diff(1) < -M_PI) z_diff(1) += 2 * M_PI;

		T += weights_(i) * x_diff * z_diff.transpose();
	}

	VectorXd z_(n_z);
	z_ << meas_package.raw_measurements_[0],
		meas_package.raw_measurements_[1],
		meas_package.raw_measurements_[2];

	//kalman gain
	MatrixXd K = T * S.inverse();

	//update state
	VectorXd z_diff = z_ - z_pred;
	while (z_diff(1) > M_PI) z_diff(1) -= 2 * M_PI;
	while (z_diff(1) < -M_PI) z_diff(1) += 2 * M_PI;

	x_ += K * z_diff;

	//covariance matrix update
	P_ = P_ - K * S * K.transpose();



	std::cout << "finish radar" << std::endl;
}