#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

    
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    VectorXd temp(4);
    temp << 0, 0, 0, 0;
    if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
        std::cout << "[-] error check the estimations or ground_truth vector" << std::endl;
        return rmse;
    }
    for (int i = 0; i < estimations.size(); i++) {
        temp = estimations[i] - ground_truth[i];
        temp = temp.array().pow(2);
        rmse += temp;
    }

    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();


    return rmse;

}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */

    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);

    double c = px * px + py * py;
    
    MatrixXd Hj(3, 4);

    if (fabs(c) < 0.0001) {
        std::cout << "division by zero" << std::endl;
        return Hj;
    }
    double d_rho_px = px / sqrt(c);
    double d_rho_py = py / sqrt(c);
    double d_phi_px = -(py / c);
    double d_phi_py = px / c;
    double d_rho_dot_px = py * (vx * py - vy * px) / pow(c, 1.5);
    double d_rho_dot_py = px * (vy * px - vx * py) / pow(c, 1.5);

    
    Hj << d_rho_px, d_rho_py, 0, 0,
        d_phi_px, d_phi_py, 0, 0,
        d_rho_dot_px, d_rho_dot_py, d_rho_px, d_rho_py;

    

    
   


    return Hj;
}
