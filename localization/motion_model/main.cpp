#include<iostream>
#include<vector>
#include "help_functions.h"

std::vector<float> initialize_prior(int map_size, std::vector<float> landmark_position,
	float position_stdev);

float motion_model(float pseudo_position, float movement, std::vector<float> priors,
	int map_size, int control_stdev);
int main() {
	//set standard deviation control
	float control_stdev = 1.0;

	//set standard deviation position
	float position_stdev = 1.0;

	//map size
	int map_size = 25;

	//meters vehicle moves per timestep
	float movement_per_timestep = 1.0;

	//landmark position
	std::vector<float> landmark_position{ 5, 10, 20 };

	//initial belief
	std::vector<float> priors = initialize_prior(map_size, landmark_position, position_stdev);

	for (unsigned int i = 0; i < map_size; ++i) {
		float pseudo_position = float(i);
		float motion_prob = motion_model(pseudo_position, movement_per_timestep, 
			priors, map_size, control_stdev);
		std::cout << motion_prob << std::endl;
	}
	return 0;
}

float motion_model(float pseudo_position, float movement, std::vector<float> priors,
	int map_size, int control_stdev) {
	float position_prob = 0.0f;
	for (int i = 0; i < map_size; ++i) {
		float next_pseudo_position = float(i);
		float delta_pos = pseudo_position - next_pseudo_position;
		float p_trans = Helpers::normpdf(delta_pos, movement, control_stdev);
		position_prob += p_trans * priors[i];
	}

	return position_prob;
}

std::vector<float> initialize_prior(int map_size, std::vector<float> landmark_position,
	float position_stdev) {
	std::vector<float> priors(map_size, 0.0);
	float initial_prob = 1 / (landmark_position.size() * (2 * position_stdev + 1));
	for (int i = 0; i < landmark_position.size(); i++) {
		priors[landmark_position[i]] = initial_prob;
		priors[landmark_position[i - 1]] = initial_prob;
		priors[landmark_position[i + 1]] = initial_prob;
	}

	return priors;
}

