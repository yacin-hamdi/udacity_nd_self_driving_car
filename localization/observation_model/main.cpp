#include<iostream>
#include<vector>
#include"helpers.h"
#include<algorithm>


std::vector<float> landmark_positions{ 5, 10, 12, 20 };
std::vector<float> pseudo_range_estimator(std::vector<float>landmarks_positions, 
	float pseudo_position);

float observation_model(std::vector<float>landmark_positions, std::vector<float>observations,
	std::vector<float> pseudo_range, float max_distance,
	float observation_std);
int main() {

	//set observation standard deviation:
	float observation_stdev = 1.0f;

	//number of x positions on map
	int map_size = 25;

	//set distance max
	float distance_max = map_size;

	//define landmarks
	std::vector<float> landmark_positions{ 5, 10, 12, 20 };

	//define observations
	std::vector<float> observations{ 5.5, 13, 15 };
	//step through each pseudo position x (i)
	for (unsigned int i = 0; i < map_size; ++i) {
		float pseudo_position = float(i);

		//get pseudo ranges
		std::vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions,
			pseudo_position);

		//get observation probability
		float observation_prob = observation_model(landmark_positions, observations,
			pseudo_ranges, distance_max,
			observation_stdev);

		//print to stdout
		std::cout << observation_prob << endl;
	}


	return 0;
}

float observation_model(std::vector<float>landmark_positions, std::vector<float>observations,
	std::vector<float> pseudo_ranges, float max_distance,
	float observation_std) {
	std::vector<float> pdfs;
	float distance_prob = 1.0;
	for (unsigned int i = 0; i < observations.size(); i++) {
		float min_pseudo_range;
		if (pseudo_ranges.size() > 0) {
			min_pseudo_range = pseudo_ranges[0];
			pseudo_ranges.erase(pseudo_ranges.begin());
		}
		else {
			min_pseudo_range = max_distance;
		}
		distance_prob = distance_prob * Helpers::normpdf(observations[i], min_pseudo_range,observation_std);
		
	
	}

	return distance_prob;


}

std::vector<float> pseudo_range_estimator(std::vector<float>landmarks_positions,
	float pseudo_position) {
	std::vector<float>pseudo_ranges;
	float pseudo_range;
	for (int i = 0; i < landmarks_positions.size(); i++) {
		pseudo_range = landmarks_positions[i] - pseudo_position;
		if (pseudo_range > 0) {
			pseudo_ranges.push_back(pseudo_range);
		}
	}
	sort(pseudo_ranges.begin(), pseudo_ranges.end());

	return pseudo_ranges;
}