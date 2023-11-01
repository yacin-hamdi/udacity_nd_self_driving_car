#ifndef COST_H
#define COST_H

#include<vector>

float goal_distance_cost(int goal_lane, int intended_lane, 
	int final_lane, float distance_to_goal);

float inefficiency_cost(int target_speed, int intended_lane,
	int final_lane, std::vector<int> lane_speeds);


#endif 

