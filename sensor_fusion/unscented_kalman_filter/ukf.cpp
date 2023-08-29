#include "ukf.h"
#include<iostream>
UKF::UKF() {

}

UKF::~UKF() {

}

void UKF::init() {

}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out) {

    //define state dimension
	int n_x = 5;

    //define spreading parameter
	double lambda = 3 - n_x;
    VectorXd x = VectorXd(n_x);
    x << 5.7441,
        1.3800,
        2.2049,
        0.5015,
        0.3528;

    //set example covariance matrix
    MatrixXd P = MatrixXd(n_x, n_x);
    P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
        -0.0013, 0.0077, 0.0011, 0.0071, 0.0060,
        0.0030, 0.0011, 0.0054, 0.0007, 0.0008,
        -0.0022, 0.0071, 0.0007, 0.0098, 0.0100,
        -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;

    MatrixXd Xsig(n_x, 2 * n_x + 1);
    MatrixXd A = P.llt().matrixL();
    Xsig.col(0) = x;
    
    
    MatrixXd lam_p = sqrt(lambda + n_x) * A;
    std::cout << lam_p << std::endl;
    for (int i = 0; i < n_x*2; i++) {
        if(i < n_x)
            Xsig.col(i + 1) = x + lam_p.col(i);
        else 
            Xsig.col(i + 1) = x - lam_p.col(i % n_x);
    }

    
    *Xsig_out = Xsig;

}