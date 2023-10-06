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
        num_particles = 50;  // TODO: Set the number of particles
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

        if (yaw_rate > 0.001) {
            x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) + sin(theta));
            y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
        }
        else {
            x += velocity * cos(yaw_rate) * delta_t;
            y += velocity * sin(yaw_rate) * delta_t;
        }

        
        theta += yaw_rate * delta_t;

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
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    double x1, x2, y1, y2, distance;
    double idx;
    std::vector<std::vector<double>> distances;
    std::vector<double> temp;
    for (unsigned int i = 0; i < observations.size(); i++) {
        x1 = observations[i].x;
        y1 = observations[i].y;
        for (unsigned int j = 0; j < predicted.size(); j++) {
            x2 = predicted[j].x;
            y2 = predicted[j].y;
            distance = dist(x1, y1, x2, y2);
            temp.push_back(distance);
        }
        distances.push_back(temp);
        temp.clear();
    }


    for (unsigned int i = 0; i < observations.size(); i++) {
        idx = minIdx(distances[i]);
        lockIdx(distances, idx);
        observations[i].id = idx;

    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   vector<LandmarkObs> &observations, 
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
    for (unsigned int i = 0; i < num_particles; i++) {
            
            xp = particles[i].x;
            yp = particles[i].y;
            theta = particles[i].theta;
            toMapCoordinate(observations, xp, yp, theta);
            dataAssociation(map_landmarks.landmark_list, observations);
        
    }



}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

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