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

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {
    //state dimensions
    int n_x = 5;

    //augmented state dimension
    int n_aug = 7;

    //create example sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
    Xsig_aug <<
        5.7441, 5.85768, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.63052, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441,
        1.38, 1.34566, 1.52806, 1.38, 1.38, 1.38, 1.38, 1.38, 1.41434, 1.23194, 1.38, 1.38, 1.38, 1.38, 1.38,
        2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.2049, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.2049, 2.2049,
        0.5015, 0.44339, 0.631886, 0.516923, 0.595227, 0.5015, 0.5015, 0.5015, 0.55961, 0.371114, 0.486077, 0.407773, 0.5015, 0.5015, 0.5015,
        0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.3528, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.3528,
        0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0, 0, -0.34641, 0,
        0, 0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0, 0, -0.34641;

    MatrixXd Xsig_pred(n_x, n_aug * 2 + 1);
    double d_t = 0.1;

    VectorXd x_pred(n_x);
    for (int i = 0; i < n_aug * 2 + 1; i++) {
        double px = Xsig_aug(0, i);
        double py = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double noise_a = Xsig_aug(5, i);
        double noise_y = Xsig_aug(6, i);

        double px_pred;
        double py_pred;
        double yaw_pred = yaw + yawd * d_t;
        double v_pred = v;
        double yawd_pred = yawd;
        
        if (fabs(yawd) > 0.0001) {
             px_pred = px + v/yawd * (sin(yaw + yawd * d_t) - sin(yaw));
             py_pred = py + v / yawd * (-cos(yaw + yawd * d_t) + cos(yaw));     
        }
        else {
            px_pred = px + v * cos(yaw) * d_t;
            py_pred = py + v * sin(yaw) * d_t;
        }

       

        double px_noise = 0.5 * pow(d_t, 2) * cos(yaw) * noise_a;
        double py_noise = 0.5 * pow(d_t, 2) * sin(yaw) * noise_a;
        double v_noise = d_t * noise_a;
        double yaw_noise = 0.5 * pow(d_t, 2) * noise_y;
        double yawd_noise = d_t * noise_y;

        x_pred << px_pred + px_noise, py_pred + py_noise, v_pred + v_noise,
            yaw_pred + yaw_noise, yawd_pred + yawd_noise;
        Xsig_pred.col(i) = x_pred;

       


    }

    std::cout << "Xsig_pred = " << std::endl << Xsig_pred << std::endl;
    *Xsig_out = Xsig_pred;

}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {

    //state dimension
    int n_x = 5;

    //augmented state dimension
    int n_aug = 7;

    //define spreading parameter
    double lambda = 3 - n_aug;

    //create example matrix with predicted sigma points
    MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);
    Xsig_pred <<
        5.9374, 6.0640, 5.925, 5.9436, 5.9266, 5.9374, 5.9389, 5.9374, 5.8106, 5.9457, 5.9310, 5.9465, 5.9374, 5.9359, 5.93744,
        1.48, 1.4436, 1.660, 1.4934, 1.5036, 1.48, 1.4868, 1.48, 1.5271, 1.3104, 1.4787, 1.4674, 1.48, 1.4851, 1.486,
        2.204, 2.2841, 2.2455, 2.2958, 2.204, 2.204, 2.2395, 2.204, 2.1256, 2.1642, 2.1139, 2.204, 2.204, 2.1702, 2.2049,
        0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337, 0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188, 0.5367, 0.535048,
        0.352, 0.29997, 0.46212, 0.37633, 0.4841, 0.41872, 0.352, 0.38744, 0.40562, 0.24347, 0.32926, 0.2214, 0.28687, 0.352, 0.318159;


    VectorXd weights(2 * n_aug + 1);
    VectorXd x(n_x);
    MatrixXd P(n_x, n_x);

    weights(0) = lambda / (lambda + n_aug);
    for (int i = 1; i < 2*n_aug + 1; i++) {
        weights(i) = 0.5 / (lambda + n_aug);
    }

    //std::cout << weights << std::endl;
    for (int i = 0; i < 2 * n_aug + 1; i++) {
        x += Xsig_pred.col(i) * weights(i);
        
    }

    VectorXd x_diff;

    for (int i = 0; i < 2 * n_aug + 1; i++) {
        x_diff = Xsig_pred.col(i) - x;
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;
        P += weights(i) * x_diff * x_diff.transpose();
    }
    std::cout << "predicted state:" << std::endl << x << std::endl;
    std::cout << "predicted covariance matrix:" << std::endl << P << std::endl;

    *x_out = x;
    *P_out = P;



}