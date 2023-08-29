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

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
    //state dimension
    int n_x = 5;
    
    // augmented state dimension
    int n_aug = 7;

    //process noise standard deviation longtitude acceleration
    float std_a = 0.2;

    //process noise standard deviation yaw acceleration
    float std_yawdd = 0.2;

    double lambda = 3 - n_aug;

    //set example state
    VectorXd x = VectorXd(n_x);
    x << 5.7441,
        1.3800,
        2.2049,
        0.5015,
        0.3528;

    //create example covariance matrix
    MatrixXd P = MatrixXd(n_x, n_x);
    P << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
        -0.0013, 0.0077, 0.0011, 0.0071, 0.0060,
        0.0030, 0.0011, 0.0054, 0.0007, 0.0008,
        -0.0022, 0.0071, 0.0007, 0.0098, 0.0100,
        -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;

    VectorXd x_aug(7);
    MatrixXd p_aug(7, 7);
    MatrixXd Xsig_aug(n_aug, n_aug * 2 + 1);

    x_aug.head(n_x) = x;
    p_aug.topLeftCorner(5, 5) = P;
    p_aug(5, 5) = std_a * std_a;
    p_aug(6, 6) = std_yawdd * std_yawdd;
    
    MatrixXd A_aug = p_aug.llt().matrixL();
    Xsig_aug.col(0) = x_aug;

    MatrixXd lam_p_aug = sqrt(lambda + n_aug) * A_aug;
    for (int i = 0; i < n_aug; i++) {
        Xsig_aug.col(i + 1) = x_aug + lam_p_aug.col(i);
        Xsig_aug.col(i + 1 + n_aug) = x_aug - lam_p_aug.col(i);
    }

    *Xsig_out = Xsig_aug;
    

}