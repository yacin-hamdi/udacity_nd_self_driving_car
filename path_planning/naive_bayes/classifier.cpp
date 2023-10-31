#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {
	left_means = Eigen::ArrayXd(4);
	left_means << 0, 0, 0, 0;
	left_stds = Eigen::ArrayXd(4);
	left_stds << 0, 0, 0, 0;	
	left_prior = 0;

	right_means = Eigen::ArrayXd(4);
	right_means << 0, 0, 0, 0;
	right_stds = Eigen::ArrayXd(4);
	right_stds << 0, 0, 0, 0;
	right_prior = 0;

	keep_means = Eigen::ArrayXd(4);
	keep_means << 0, 0, 0, 0;
	keep_stds = Eigen::ArrayXd(4);
	keep_stds << 0, 0, 0, 0;
	keep_prior = 0;
}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d,
			s_dot and d_dot.
		  - Example : [
				[3.5, 0.1, 5.9, -0.02],
				[8.0, -0.3, 3.0, 2.2],
				...
			]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
	


	float left_size = 0;
	float right_size = 0;
	float keep_size = 0;

	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] == "left") {
			left_means += ArrayXd::Map(data[i].data(), data[i].size()); //conversion of data[i] to ArrayXd
			left_size += 1;
		}
		else if (labels[i] == "keep") {
			keep_means += ArrayXd::Map(data[i].data(), data[i].size());
			keep_size += 1;
		}
		else if (labels[i] == "right") {
			right_means += ArrayXd::Map(data[i].data(), data[i].size());
			right_size += 1;
		}
	}

	

	left_means /= left_size;
	right_means /= right_size;
	keep_means /= keep_size;

	
	Eigen::ArrayXd data_point;
	for (int i = 0; i < labels.size(); i++) {
		data_point = Eigen::ArrayXd::Map(data[i].data(), data[i].size());
		if (labels[i] == "left") {
			left_stds += (data_point - left_means) * (data_point - left_means);
		}
		else if (labels[i] == "right") {
			right_stds += (data_point - right_means) * (data_point - right_means);
		}
		else if (labels[i] == "keep") {
			keep_stds += (data_point - keep_means) * (data_point - keep_means);
		}
	}

	

	left_stds = (left_stds / left_size).sqrt();
	right_stds = (right_stds / right_size).sqrt();
	keep_stds = (keep_stds / keep_size).sqrt();

	left_prior = left_size / labels.size();
	right_prior = right_size / labels.size();
	keep_prior = keep_size / labels.size();
	
}

double GNB::pdf(double mu, double std, double x) {
	double term1 = 1 / (sqrt(2 * M_PI * pow(std, 2)));
	double term2 = exp(-pow(x - mu, 2) / (2 * pow(std, 2)));
	return term1 * term2;
}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
		# TODO - complete this
	*/

	double left_prob = 1.0;
	double right_prob = 1.0;
	double keep_prob = 1.0;

	for (int i = 0; i < 4; i++) {
		left_prob *= pdf(left_means[i], left_stds[i], sample[i]);
		keep_prob *= pdf(keep_means[i], keep_stds[i], sample[i]);
		right_prob *= pdf(right_means[i], right_means[i], sample[i]);

	}

	


	left_prob *= left_prior;
	right_prob *= right_prior;
	keep_prob *= keep_prior;


	double probs[3] = { left_prob, keep_prob, right_prob };
	double max_prob = left_prob;
	int max_index = 0;
	for (int i = 1; i < 3; i++) {
		if (probs[i] > max_prob) {
			max_prob = probs[i];
			max_index = i;
		}
	}
	return this->possible_labels[max_index];

}