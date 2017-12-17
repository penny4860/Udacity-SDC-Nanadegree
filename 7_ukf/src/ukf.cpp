#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Todo : std_a_, std_yawdd_ parameter tuning
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.5;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.5;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
    */

    // initially set to false, set to true in first call of ProcessMeasurement
    is_initialized_ = false;
    // State dimension
    n_x_ = 5;
    // Augmented state dimension
    n_aug_ = 7;
    // Sigma point spreading parameter
    lambda_ = 3 - n_aug_;
    time_us_ = 0;

    // initializing matrices
    Xsig_pred_ = MatrixXd(n_aug_, n_aug_);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */

    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        // state components
        float px;
        float py;
        float v = 0;
        float theta;
        float theta_d = 0;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            float rho = meas_package.raw_measurements_[0];  // range: radial
            float phi = meas_package.raw_measurements_[1];  // bearing:
            px = rho * cos(phi);
            py = rho * sin(phi);
            theta = atan2(py, px);

        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            // init state
            px = meas_package.raw_measurements_[0];
            py = meas_package.raw_measurements_[1];
            theta = atan2(py, px);
        }
        x_ << px, py, v, theta, theta_d;
        P_.fill(0.0);
        P_(0, 0) = 1.0;
        P_(1, 1) = 1.0;
        P_(2, 2) = 1.0;
        P_(3, 3) = 1.0;
        P_(4, 4) = 1.0;

        time_us_ = meas_package.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    // compute the time elapsed between the current and previous measurements
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     */

    // Radar updates
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    }
    // Laser updates
    else {
        UpdateLidar(meas_package);
    }
}

VectorXd UKF::_create_augmented_state(void) {
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.head(5) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;
    return x_aug;
}

MatrixXd UKF::_create_augmented_covariance(void) {
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;
    return P_aug;
}

MatrixXd UKF::_generate_sigma_points(VectorXd x_aug, MatrixXd P_aug) {
    MatrixXd A = P_aug.llt().matrixL();
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
    Xsig_aug.col(0) = x_aug;
    for (int i = 0; i < n_aug_; i++) {
        Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
        Xsig_aug.col(i + 1 + n_aug_) =
            x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
    }
    return Xsig_aug;
}

MatrixXd UKF::_predict_sigma_points(MatrixXd Xsig_aug, double delta_t) {
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd x_sigma = Xsig_aug.col(i);
        VectorXd x_pred(5);

        double px = x_sigma(0);
        double py = x_sigma(1);
        double v = x_sigma(2);
        double theta = x_sigma(3);
        double theta_dot = x_sigma(4);

        double nu_long_acc = x_sigma(5);
        double nu_theta_acc = x_sigma(6);

        if (fabs(theta_dot) < 0.001) {
            x_pred(0) = px + v * cos(theta) * delta_t;
            x_pred(1) = py + v * sin(theta) * delta_t;
        } else {
            x_pred(0) =
                px +
                v / theta_dot * (sin(theta + theta_dot * delta_t) - sin(theta));
            x_pred(1) =
                py + v / theta_dot *
                         (-cos(theta + theta_dot * delta_t) + cos(theta));
        }
        x_pred(2) = v;
        x_pred(3) = theta + theta_dot * delta_t;
        x_pred(4) = theta_dot;

        // noise addition
        x_pred(0) += 0.5 * delta_t * delta_t * cos(theta) * nu_long_acc;
        x_pred(1) += 0.5 * delta_t * delta_t * sin(theta) * nu_long_acc;
        x_pred(2) += delta_t * nu_long_acc;
        x_pred(3) += 0.5 * delta_t * delta_t * nu_theta_acc;
        x_pred(4) += delta_t * nu_theta_acc;

        Xsig_pred.col(i) = x_pred;
    }
    return Xsig_pred;
}

void UKF::_predict(VectorXd *x_pred, MatrixXd *P_pred) {
    VectorXd weights = VectorXd(2 * n_aug_ + 1);
    VectorXd x = VectorXd(n_x_);
    MatrixXd P = MatrixXd(n_x_, n_x_);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        if (i == 0)
            weights(i) = lambda_ / (lambda_ + n_aug_);
        else
            weights(i) = 1 / (2 * (lambda_ + n_aug_));
    }

    // predict state mean
    x.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        x += weights(i) * Xsig_pred_.col(i);
    }
    // predict state covariance matrix
    P.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x;
        // angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        P = P + weights(i) * x_diff * x_diff.transpose();
    }
    // write result
    *x_pred = x;
    *P_pred = P;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance
    matrix.
    */

    //    cout << "\nPrediction() is called\n";
    //    cout << "	dt = " << delta_t << "\n";

    // 1. create augmented state, augmented covariance
    VectorXd x_aug = _create_augmented_state();
    MatrixXd P_aug = _create_augmented_covariance();

    // 2. generate sigma points
    MatrixXd Xsig_aug = _generate_sigma_points(x_aug, P_aug);

    // 3. predict sigma points
    // create matrix with predicted sigma points as columns
    Xsig_pred_ = _predict_sigma_points(Xsig_aug, delta_t);
    _predict(&x_, &P_);

    //    cout << "\npred state : \n"<< x_ << "\n";
    //    cout << P_;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the
    object's position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
    VectorXd z = meas_package.raw_measurements_;
    MatrixXd H = MatrixXd(2, 5);
    MatrixXd R = MatrixXd(2, 2);
    R << std_laspx_ * std_laspx_, 0, 0, std_laspy_ * std_laspy_;

    H << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0;

    VectorXd z_pred = H * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H.transpose();
    MatrixXd PHt = P_ * Ht;
    MatrixXd S = H * PHt + R;
    MatrixXd Si = S.inverse();
    MatrixXd K = PHt * Si;

    // new estimate
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H) * P_;

    MatrixXd yt = y.transpose();
    MatrixXd nis = yt * Si * y;
    nis_ = nis(0, 0);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the
    object's position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */

    // 1. Sigma Point
    // create matrix for sigma points in measurement space
    MatrixXd Zsig = _measurement_sigma_points();

    // 2. measurement space variable (z_pred, S)
    VectorXd z_pred = _pred_measurement(Zsig);
    MatrixXd S = _calc_measurement_cov(Zsig, z_pred);

    // 3. Posterior (state, covariance)
    VectorXd z = meas_package.raw_measurements_;
    _update_posterior_var(z, z_pred, S, Zsig);

    // 4. NIS
    VectorXd y = z - z_pred;
    MatrixXd yt = y.transpose();
    MatrixXd Si = S.inverse();
    MatrixXd nis = yt * Si * y;
    nis_ = nis(0, 0);
}

MatrixXd UKF::_measurement_sigma_points(void) {
    int n_z = 3;
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);

    // transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // 2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);  // r
        Zsig(1, i) = atan2(p_y, p_x);              // phi
        Zsig(2, i) =
            (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);  // r_dot
    }
    return Zsig;
}

VectorXd UKF::_get_sigma_weights(void) {
    VectorXd weights = VectorXd(2 * n_aug_ + 1);
    double weight_0 = lambda_ / (lambda_ + n_aug_);
    weights(0) = weight_0;
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {
        double weight = 0.5 / (n_aug_ + lambda_);
        weights(i) = weight;
    }
    return weights;
}

VectorXd UKF::_pred_measurement(MatrixXd Zsig) {
    // set weights
    VectorXd weights = _get_sigma_weights();
    // mean predicted measurement
    VectorXd z_pred = VectorXd(3);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights(i) * Zsig.col(i);
    }
    return z_pred;
}

MatrixXd UKF::_calc_measurement_cov(MatrixXd Zsig, VectorXd z_pred) {
    VectorXd weights = _get_sigma_weights();

    // measurement covariance matrix S
    int n_z = 3;
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // 2n+1 simga points
        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        S = S + weights(i) * z_diff * z_diff.transpose();
    }

    // add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);
    R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0,
        std_radrd_ * std_radrd_;
    S = S + R;
    return S;
}

void UKF::_update_posterior_var(VectorXd z, VectorXd z_pred, MatrixXd S,
                                MatrixXd Zsig) {
    // create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, 3);
    VectorXd weights = _get_sigma_weights();

    // calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  // 2n+1 simga points

        // residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // angle normalization
        while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
        while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        // angle normalization
        while (x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
        while (x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

        Tc = Tc + weights(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd z_diff = z - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    // update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
}
