#include<iostream>
#include<vector>


std::vector<float> initialize_prior(int map_size,
										std::vector<float> landmarks, 
											float position_std);

int main() {

	int map_size = 25;
	std::vector<float> landmarks_position{ 5, 10, 20 };
	float position_std = 1.0;

	std::vector<float> prior = initialize_prior(map_size, landmarks_position, position_std);
	for (int i = 0; i < prior.size(); i++) {
		std::cout << prior[i] << " ";
	}
	std::cout << std::endl;

	return 0;
}

std::vector<float> initialize_prior(int map_size,
	std::vector<float> landmarks,
	float position_std) {

	std::vector<float> priors(map_size, 0.0);
	int position_range = 2 * position_std + 1;
	float initial_prob = 1 / float(position_range * landmarks.size());
	std::cout << initial_prob << std::endl;

	for (int i = 0; i < landmarks.size(); i++) {
		priors[landmarks[i]] = initial_prob;
		priors[landmarks[i] + 1] = initial_prob;
		priors[landmarks[i] - 1] = initial_prob;
	}

	return priors;

}
