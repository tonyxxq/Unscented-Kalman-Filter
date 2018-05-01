#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // 加速度越小曲线越光滑
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2;

  // DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
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
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.

  /**
   TODO:

   Complete the initialization. See ukf.h for other member properties.

   Hint: one or more values initialized above might be wildly off...
   */
  // 是否已经初始化
  is_initialized_ = false;

  // 状态向量维度
  n_x_ = 5;

  // 扩展状态向量维度
  n_aug_ = 7;

  // 扩展参数
  lambda_ = 3 - n_x_;

  // 初始化P矩阵
  P_ << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 1, 0,
      0, 0, 0, 0, 1;

  // 初始化上一次记录的时间，设为0
  time_us_ = 0;

  // 初始化sigma点集合预测值
  Xsig_pred_ = MatrixXd(n_x_, 2* n_aug_ +1);
}

UKF::~UKF() {
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /*****************************************************************************
   *  初始化，使用第一次的测量数据去更新状态且初始化需要用到的协方差矩阵
   *  注意需要先判断是否开启了雷达或激光雷达
   ****************************************************************************/

  // 判断是否已经初始化数据
  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      // 雷达测量数据是极坐标数据，需要转换为CTRV模型
      // 因为雷达测量的速度和CRTV的速度不是同一个概念，所以不能初始化速度
      float ro = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float ro_dot = meas_package.raw_measurements_[2];
      x_ << ro * cos(phi), ro * sin(phi), 0, 0, 0;
      // 初始化时间
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER
        && use_laser_) {
      // 激光雷达测量数据只有位移，没有速度，初始化速度为0
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      // 初始化时间
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
    }
    return;
  }

  // 计算两次测量之间的时间差delta_t
  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // 预测
  Prediction(delta_t);

  // 更新
  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (use_laser_
      && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
}

/**
 * 计算sigma点集，预测sigma点集, 计算预测的sigma点集合的均值和协方差
 * 参数： {double} delta_t 两次测量之间的间隔时间，单位为秒
 */
void UKF::Prediction(double delta_t) {

  /*****************************************************************************
   *  求sigma点集且扩展状态向量和P矩阵，因为噪声也是非线性的，需要计算在内
   *  分为四步
   ****************************************************************************/

  // 1. 扩展状态向量，在原状态向量的基础上增加了径向加速度和角速度加速度
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  // 因为噪声的均值为0，所以这个地方两个都设为0
  x_aug(5) = 0;
  x_aug(6) = 0;

  // 2. 扩展协方差P矩阵，在原有协方差P矩阵基础上增加了径向加速度和角速度加速度误差协方差
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // 3. 求得A矩阵
  MatrixXd A = P_aug.llt().matrixL();

  // 4. 计算sigma点集，这些都是按照求sigma点集公式来的
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  /*****************************************************************************
   *  将sigma点带入方程，预测sigma点集
   *  注意：此处输入的扩展向量的维度为7，计算的出的预测向量维度为5
   ****************************************************************************/

  // 遍历sigma点集中的每一个点，根据模型方程，计算sigma点集的预测值
  Xsig_pred_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    // 注意：计算px, py的时候，因为模型中角速度作为分母，所以需要特殊处理
    double px_p, py_p;
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // 加上噪声向量
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;
    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;

    // 把预测值，写入预测矩阵
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  /*****************************************************************************
   *  根据预测的点，计算均值，协方差
   ****************************************************************************/

  // 设置权重，因为在计算sigma点的时候乘以了lambda_ + n_aug_，所以这个地方需要再除一下
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  weights(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights(i) = weight;
  }


  // 遍历每一个sigma预测点，根据之上计算每个点的权重，计算均值
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    x_ = x_ + weights(i) * Xsig_pred_.col(i);
  }


  // 计算协方差矩阵P
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // 把偏航角范围限制在-PI和PI之间
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;
    P_ = P_ + weights(i) * x_diff * x_diff.transpose();
  }
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  // 把测量值放到一个数组
  VectorXd z_ = VectorXd(n_z);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights(i) = weight;
  }

  // 把sigma点集的预测值， 根据测量模型转换为测量空间的点集合
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    Zsig(0, i) = p_x;
    Zsig(1, i) = p_y;
  }

  // 计算均值
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  // 计算协方差矩阵
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;
    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  // 在S矩阵中加上测量噪声矩阵
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0, 0, std_laspx_ * std_laspx_;
  S = S + R;

  // 计算Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;
    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  // 计算卡尔曼增益
  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z_ - z_pred;
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // 更新状态均值和协方差矩阵
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  int n_z = 3;
  // 把测量值放到一个数组
  VectorXd z_ = VectorXd(n_z);
  z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package
      .raw_measurements_[2];

  VectorXd weights = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights(0) = weight_0;
  for (int i = 1; i < 2 * n_aug_ + 1; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights(i) = weight;
  }

  // 把sigma点集的预测值， 根据测量模型转换为测量空间的点集合
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                        //r
    Zsig(1, i) = atan2(p_y, p_x);                                 //phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y);   //r_dot
  }

  // 计算均值
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  // 计算协方差矩阵
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;
    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  // 在S矩阵中加上测量噪声矩阵
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_ * std_radphi_, 0, 0, 0, std_radrd_
      * std_radrd_;
  S = S + R;

  // 计算Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;
    Tc = Tc + weights(i) * x_diff * z_diff.transpose();
  }

  // 计算卡尔曼增益
  MatrixXd K = Tc * S.inverse();
  VectorXd z_diff = z_ - z_pred;
  while (z_diff(1) > M_PI)
    z_diff(1) -= 2. * M_PI;
  while (z_diff(1) < -M_PI)
    z_diff(1) += 2. * M_PI;

  // 更新状态均值和协方差矩阵
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
}
