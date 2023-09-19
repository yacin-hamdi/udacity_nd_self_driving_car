#include<iostream>
#include<random>


// @param  gps_x	gps provided x position
// @param  gps_y	gps provided y position
// @param  theta	gps provided yaw

void printSample(double gps_x, double gps_y, double theta);

int main() {
	double gps_x = 4983;
	double gps_y = 5029;
	double theta = 1.201;

	printSample(gps_x, gps_y, theta);

	return 0;
}



void printSample(double gps_x, double gps_y, double theta) {
	std::default_random_engine gen;
	double sample_x, sample_y, sample_theta;

	double std_x = 2;
	double std_y = 2;
	double std_theta = 0.05;

	std::normal_distribution<double> dist_x(gps_x, std_x);
	std::normal_distribution<double> dist_y(gps_y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < 3; i++) {
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		std::cout << "sample" << i + 1 << " " << sample_x << " " << sample_y << " " << sample_theta << std::endl;

	}

}