#ifndef MEASUREMENT_PACKAGE_H
#define MEASUREMENT_PACKAGE_H

#include "Eigen/Dense"
class MeasurementPackage {
public:

	int64_t timestamp_;
	enum SensorType {
		LASER,
		RADAR
	} sensor_type_;

	Eigen::VectorXd raw_measurements_;
};


#endif MEASUREMENT_PACKAGE_H
