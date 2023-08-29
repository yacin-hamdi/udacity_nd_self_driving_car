#ifndef UKF_H
#define UKF_H
#include"Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
class UKF {
public:
	UKF();
	virtual ~UKF();

	void init();

	void GenerateSigmaPoints(MatrixXd* Xsig_out);
};




#endif
