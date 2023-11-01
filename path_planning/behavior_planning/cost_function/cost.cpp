#include"cost.h"
#include<math.h>
float goal_distance_cost(int goal_lane, int intended_lane, int final_lane,
	float distance_to_goal) {
    /*
    The cost increases with both the distance of intended lane from the goal
    and the distance of the final lane from the goal. The cost of being out of the
    goal lane also becomes larger as vehicle approaches the goal.
    */
    int delta_d = abs(2.0 * goal_lane - intended_lane - final_lane);

    float cost = 1 - exp(- delta_d / distance_to_goal);

    return cost;
}


float inefficiency_cost(int target_speed, int intended_lane,
    int final_lane, std::vector<int> lane_speeds) {
    /*
   Cost becomes higher for trajectories with intended lane and final lane that 
   have traffic slower than target_speed.
   */

   //TODO: Replace cost = 0 with an appropriate cost function.
    int intended_speed = lane_speeds[intended_lane];
    int final_speed = lane_speeds[final_lane];

    float cost = (2.0 * target_speed - intended_speed - final_speed) / target_speed;
    
     

    return cost;
}