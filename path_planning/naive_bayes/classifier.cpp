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
	int N = data.size();
	double mean_s = 0;
	double mean_d = 0;
	double mean_s_dot = 0;
	double mean_d_dot = 0;
	

	for (int i = 0; i < N; i++) {
		mean_s += data[i][0];
		mean_d += data[i][1];
		mean_s_dot += data[i][2];
		mean_d_dot += data[i][3];
	}

	mean_s /= N;
	mean_d /= N;
	mean_s_dot /= N;
	mean_d_dot /= N;

	double var_s = 0;
	double var_d = 0;
	double var_s_dot = 0;
	double var_d_dot = 0;

	for (int i = 0; i < N; i++) {
		var_s += pow(data[i][0], 2) - pow(mean_s, 2);
		var_d += pow(data[i][1], 2) - pow(mean_d, 2);
		var_s_dot += pow(data[i][2], 2) - pow(mean_s_dot, 2);
		var_d_dot += pow(data[i][3], 2) - pow(mean_d_dot, 2);
	}

	var_s /= N;
	var_d /= N;
	var_s_dot /= N;
	var_d_dot /= N;
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

	return this->possible_labels[1];

}