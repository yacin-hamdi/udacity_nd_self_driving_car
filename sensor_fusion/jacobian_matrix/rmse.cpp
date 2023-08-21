#include<iostream>
#include"Eigen/Dense"
#include<vector>

Eigen::VectorXd calculateRMSE(const std::vector<Eigen::VectorXd>& estimations,
	const std::vector<Eigen::VectorXd>& ground_truth);

int main() {

	std::vector<Eigen::VectorXd> estimations;
	std::vector<Eigen::VectorXd> ground_truth;

	//the input list of estimations
	Eigen::VectorXd e(4);
	e << 1, 1, 0.2, 0.1;
	estimations.push_back(e);
	e << 2, 2, 0.3, 0.2;
	estimations.push_back(e);
	e << 3, 3, 0.4, 0.3;
	estimations.push_back(e);

	//the corresponding list of ground truth values
	Eigen::VectorXd g(4);
	g << 1.1, 1.1, 0.3, 0.2;
	ground_truth.push_back(g);
	g << 2.1, 2.1, 0.4, 0.3;
	ground_truth.push_back(g);
	g << 3.1, 3.1, 0.5, 0.4;
	ground_truth.push_back(g);

	//call the CalculateRMSE and print out the result
	std::cout << calculateRMSE(estimations, ground_truth) << std::endl;

	return 0;
}

Eigen::VectorXd calculateRMSE(const std::vector<Eigen::VectorXd>& estimations,
	const std::vector<Eigen::VectorXd>& ground_truth) {
	Eigen::VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	if (estimations.size() == 0) {
		std::cout << "the estimations size shoude not be of size 0" << std::endl;
		return rmse;
	}
	if(estimations.size() != ground_truth.size()) {
		std::cout << "the size of estimations vector and ground_truth should be of the same size" << std::endl;
		return rmse;
	}
	Eigen::VectorXd temp_sum(4);
	temp_sum << 0, 0, 0, 0;
	for (int i = 0; i < estimations.size(); i++) {
		temp_sum = estimations[i] - ground_truth[i];
		temp_sum = temp_sum.array().pow(2);
		rmse += temp_sum;
	}

	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();
	return rmse;
}