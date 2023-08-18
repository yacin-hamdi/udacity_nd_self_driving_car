#ifndef TRACKING_H
#define TRACKING_H

#include "kalman_filter.h"
#include "measurement_package.h"



class Tracking {
public:
	Tracking();
	virtual ~Tracking();
	void processMeasurement(const MeasurementPackage& measurementPackage);
	KalmanFilter kf_;

private:
	bool _is_initialized;
	int64_t _previous_timestamp;

	float _noise_ax;
	float _noise_ay;

};


#endif TRACKING_H