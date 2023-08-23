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
    VectorXd temp(4);
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

    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    float c = px * px + py * py;
    float d_rho_px = px / sqrt(c);
    float d_rho_py = py / sqrt(c);
    float d_phi_px = -py / c;
    float d_phi_py = px / c;
    float d_rho_dot_px = py * (vx * py - vy * px) / pow(c, 3 / 2);
    float d_rho_dot_py = px * (vy * px - vx * py) / pow(c, 3 / 2);

    MatrixXd Hj(3, 4);
    Hj << d_rho_px, d_rho_py, 0, 0,
        d_phi_px, d_phi_px, 0, 0,
        d_rho_dot_px, d_rho_dot_py, d_rho_px, d_rho_py;
    


    return Hj;
}
