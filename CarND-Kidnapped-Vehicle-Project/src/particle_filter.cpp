/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
    
    if (!is_initialized) {
        num_particles = 100;  // TODO: Set the number of particles
        std::default_random_engine gen;
        std::normal_distribution<double> dist_x(x, std[0]);
        std::normal_distribution<double> dist_y(y, std[1]);
        std::normal_distribution<double> dist_theta(theta, std[2]);



        for (unsigned int i = 0; i < num_particles; i++) {
            Particle particle;
            particle.id = i;
            particle.x = dist_x(gen);
            particle.y = dist_y(gen);
            particle.theta = dist_theta(gen);
            particle.weight = 1;
            particles.push_back(particle);
            weights.push_back(particle.weight);

        }
        is_initialized = true;
    }
 

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

    std::default_random_engine gen;
    double x, y, theta;
    for (unsigned int i = 0; i < num_particles; i++) {
        x = particles[i].x;
        y = particles[i].y;
        theta = particles[i].theta;

        if (fabs(yaw_rate) > 0.0001) {
            x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
            y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
            theta += yaw_rate * delta_t;
        }
        else {
            x += velocity * cos(theta) * delta_t;
            y += velocity * sin(theta) * delta_t;
        }

        

        
        

        std::normal_distribution<double> dist_x(x, std_pos[0]);
        std::normal_distribution<double> dist_y(y, std_pos[1]);
        std::normal_distribution<double> dist_theta(theta, std_pos[2]);

        x = dist_x(gen);
        y = dist_y(gen);
        theta = dist_theta(gen);

        particles[i].x = x;
        particles[i].y = y;
        particles[i].theta = theta;
    }


}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations, double sensor_range) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    double x1, x2, y1, y2;
    int idx;
    double min_dist;
    double current_dist;
    int pred_id;
    for (unsigned int i = 0; i < observations.size(); i++) {
        min_dist = sensor_range * sqrt(2);
        idx = -1;
        x1 = observations[i].x;
        y1 = observations[i].y;
        for (unsigned int j = 0; j < predicted.size(); j++) {
            x2 = predicted[j].x;
            y2 = predicted[j].y;
            pred_id = predicted[j].id;
            current_dist = dist(x1, y1, x2, y2);
            if (current_dist < min_dist) {
                min_dist = current_dist;
                idx = pred_id;
            }
        }

        observations[i].id = idx;
    }



    

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    double xp, yp, theta;
    std::vector<LandmarkObs> observations_vec;
    std::vector<LandmarkObs> predicted;
    LandmarkObs temp;
    double weight_sum = 0;
    for (unsigned int i = 0; i < num_particles; i++) {
        observations_vec.clear();
        predicted.clear();
        xp = particles[i].x;
        yp = particles[i].y;
        theta = particles[i].theta;
      

        for (unsigned int ob_i = 0; ob_i < observations.size(); ob_i++) {
            temp.id = observations[ob_i].id;
            temp.x = observations[ob_i].x;
            temp.y = observations[ob_i].y;
            observations_vec.push_back(temp);
        }

        for (unsigned int lm_i = 0; lm_i < map_landmarks.landmark_list.size(); lm_i++) {
            Map::single_landmark_s map_lm = map_landmarks.landmark_list[lm_i];
            if ((fabs(yp - map_lm.y_f) <= sensor_range) && (fabs(xp - map_lm.x_f) <= sensor_range)) {
                temp.id = map_lm.id_i;
                temp.x = map_lm.x_f;
                temp.y = map_lm.y_f;
                predicted.push_back(temp);
            }
        }

            toMapCoordinate(observations_vec, xp, yp, theta);
            dataAssociation(predicted, observations_vec, sensor_range);
            double std_x = std_landmark[0];
            double std_y = std_landmark[1];

            for (unsigned int ob_i = 0; ob_i < observations_vec.size(); ob_i++) {
                int obs_id = observations_vec[ob_i].id;
                double x = observations_vec[ob_i].x;
                double y = observations_vec[ob_i].y;
                for (unsigned int pred_i = 0; pred_i < predicted.size(); pred_i++) {
                    int pred_id = predicted[pred_i].id;
                    double mu_x = predicted[pred_i].x;
                    double mu_y = predicted[pred_i].y;
                    if(pred_id == obs_id)
                        particles[i].weight *= multivariateGauss(x, mu_x, y, mu_y, std_x, std_y);

                }
                

                
                
            }
            weight_sum += particles[i].weight;
            
    }
    //normalize weights
    std::cout << "main loop" << std::endl;
    for (unsigned int i = 0; i < num_particles; i++) {
        particles[i].weight /= weight_sum;
        weights[i] = particles[i].weight;
    }

    


}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

    vector<Particle> resampled_particles;

    // Create a generator to be used for generating random particle index and beta value
    std::default_random_engine gen;

    //Generate random particle index
    std::uniform_int_distribution<int> particle_index(0, num_particles - 1);

    int current_index = particle_index(gen);

    double beta = 0.0;

    double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());

    for (int i = 0; i < particles.size(); i++) {
        std::uniform_real_distribution<double> random_weight(0.0, max_weight_2);
        beta += random_weight(gen);

        while (beta > weights[current_index]) {
            beta -= weights[current_index];
            current_index = (current_index + 1) % num_particles;
        }
        resampled_particles.push_back(particles[current_index]);
    }
    particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}