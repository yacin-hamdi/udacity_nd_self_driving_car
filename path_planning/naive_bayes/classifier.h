#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using Eigen::ArrayXd;

class GNB {
public:

    vector<string> possible_labels = { "left","keep","right" };

    Eigen::ArrayXd left_means;
    Eigen::ArrayXd left_stds;
    double left_prior;

    Eigen::ArrayXd right_means;
    Eigen::ArrayXd right_stds;
    double right_prior;

    Eigen::ArrayXd keep_means;
    Eigen::ArrayXd keep_stds;
    double keep_prior;


    /**
    * Constructor
    */
    GNB();

    /**
    * Destructor
    */
    virtual ~GNB();

    void train(vector<vector<double> > data, vector<string>  labels);
    double pdf(double means, double std, double x);
    string predict(vector<double>);

};

#endif


