#include "tools.h"
#include<iostream>
using Eigen::VectorXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */

    VectorXd rmse(4);
    rmse.fill(0.0);
    VectorXd temp(4);
    temp.fill(0.0);
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