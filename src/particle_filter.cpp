/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <iostream>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Reference: Implementation of a Particle Filter > 5
  num_particles = 500;

  random_device rd;
  default_random_engine gen(rd());

  double std_x, std_y, std_psi; // Standard deviations for x, y, and psi

  // TODO: Set standard deviations for x, y, and psi
  std_x = std[0];
  std_y = std[1];
  std_psi = std[2];

  // This line creates a normal (Gaussian) distribution for x
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_psi(theta, std_psi);

  for (int i = 0; i < num_particles; ++i) {

    Particle out;
    out.id = i;
    out.x = dist_x(gen);
    out.y = dist_y(gen);
    out.theta = dist_psi(gen);
    out.weight = 1.0;

    particles.push_back(out);

    // add weight index
    weights.push_back(1.0);

    // Print your samples to the terminal.
    cout << "x: " << out.x << endl;
    cout << "y: " << out.y << endl;
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  random_device rd;
  default_random_engine gen(rd());

  for (int i = 0; i < num_particles; ++i) {

    double x, y, theta;
    Particle item = particles[i];

    if (yaw_rate == 0) {
      x = item.x + velocity * delta_t * cos(item.theta);
      y = item.y + velocity * delta_t * sin(item.theta);
      theta = item.theta;
    } else {
      // Reference: Implementation of a Particle Filter > 8
      x = item.x + velocity / yaw_rate * (sin(item.theta + yaw_rate * delta_t) - sin(item.theta));
      y = item.y + velocity / yaw_rate * (cos(item.theta) - cos(item.theta + yaw_rate * delta_t));
      theta = item.theta + yaw_rate * delta_t;
    }

    // same as init
    double std_x, std_y, std_psi; // Standard deviations for x, y, and psi
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_psi = std_pos[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_psi(theta, std_psi);

    // update item
    item.x = dist_x(gen);
    item.y = dist_y(gen);
    item.theta = dist_psi(gen);

    particles[i] = item;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
  //   for the fact that the map's y-axis actually points downwards.)
  //   http://planning.cs.uiuc.edu/node99.html

  // loop particles
  for (int i = 0; i < num_particles; i++) {
    Particle item = particles[i];
    item.weight = 1.0;

    // loop observations for each
    for (int y = 0; y < observations.size(); y++) {
      LandmarkObs current = observations[y];
      double x_o, y_o;
      y_o = item.y + (current.x * sin(item.theta) + current.y * cos(item.theta));
      x_o = item.x + (current.x * cos(item.theta) - current.y * sin(item.theta));

      double hit_distance = sensor_range; // sensor_range Range [m] of sensor
      int hit_idx = -1;

      // loop landmarks for finding the closest
      for (int x = 0; x < map_landmarks.landmark_list.size(); x++) {

        // nearest neighbor, look for closest distance between two points
        double distance = sqrt(
            pow(x_o - map_landmarks.landmark_list[x].x_f, 2.0) + pow(y_o - map_landmarks.landmark_list[x].y_f, 2.0));
        if (distance < hit_distance) {
          hit_idx = x;
          hit_distance = distance;
        }
      }

      // only if we have a hit
      if (hit_idx >= 0) {
        Map::single_landmark_s hit_mk = map_landmarks.landmark_list[hit_idx];

        // Reference: Particle Filters > 14 & 23
        // Refenence: Implementation of a Particle Filter > 11
        // exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
        long double exp_formula = exp(-0.5 * ((pow(x_o - hit_mk.x_f, 2.0) / (2 * pow(std_landmark[0], 2.0))
            + pow(y_o - hit_mk.y_f, 2.0) / (2 * pow(std_landmark[1], 2.0)))));
        long double final_multi = exp_formula / (2 * M_PI * std_landmark[0] * std_landmark[1]);

        item.weight *= final_multi;
      } else {
        // cout << "no hit" << endl;
      }
    }

    particles[i] = item;
    weights[i] = item.weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  //  Example from http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  //  std::random_device rd;
  //  std::mt19937 gen(rd());
  //  std::discrete_distribution<> d({40, 10, 10, 40});
  //  std::map<int, int> m;
  //  for(int n=0; n<10000; ++n) {
  //    ++m[d(gen)];
  //  }

  // Reference: Particle Filters > 20
  vector<Particle> out;
  random_device rd;
  mt19937 gen(rd());
//  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++) {
    //  while w[index] < beta:
    //    beta = beta - w[index]
    //    index = index + 1
    //  select p[index]

    out.push_back(particles[distribution(gen)]);
  }

  particles = out;
}

void ParticleFilter::write(std::string filename) {
  // You don't need to modify this file.
  std::ofstream dataFile;
  dataFile.open(filename, std::ios::app);
  for (int i = 0; i < num_particles; ++i) {
    dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
  }
  dataFile.close();
}
