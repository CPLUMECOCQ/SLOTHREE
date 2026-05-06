/**
 * @file Maths.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief
 * @version 0.1
 * @date 2025-05-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <cmath>

template <typename T>
class Maths {
 private:
 public:
  static T round(const T& x, const double& precision);
  static std::vector<T> round_vec(const std::vector<T>& vecx, const double& precision);
  static T compute_norm(const T& a, const T& b, const T& eps = static_cast<T>(1e-15));
  static T compute_norm_log(const T& a, const T& b, const T& eps = static_cast<T>(1e-15));

  static std::vector<T> simplex_to_hypercube(const std::vector<T>& coord_simplex);
  static std::vector<T> hypercube_to_simplex(const std::vector<T>& coord_hyper);

  Maths();
  ~Maths();
};
template <typename T>
Maths<T>::Maths() {}

template <typename T>
T Maths<T>::round(const T& x, const double& precision) {
  return std::round(x / precision) * precision;
}

template <typename T>
std::vector<T> Maths<T>::round_vec(const std::vector<T>& vecx, const double& precision) {
  std::size_t N = vecx.size();
  std::vector<T> res;
  res.resize(N);
  for (std::size_t i = 0; i < vecx.size(); i++) {
    res[i] = std::round(vecx[i] / precision) * precision;
  }

  return res;
}

template <typename T>
T Maths<T>::compute_norm(const T& a, const T& b, const T& eps) {
  // T factor = (std::abs(a) < 1e-20 || std::abs(b) < 1e-20) ? 1e30 : 1;
  // T a_ = a * factor;
  // T b_ = b * factor;
  // T diff = a_ - b_;
  // T res = std::abs(diff) / (std::abs(b_) + eps);
    T diff = a - b;
  T res = std::abs(diff) / (std::abs(b + eps));
  return res;
}

template <typename T>
T Maths<T>::compute_norm_log(const T& a, const T& b, const T& eps) {
  T diff = a - b;
  T res = std::exp(diff) - 1.0;
  return res;
}

template <typename T>
std::vector<T> Maths<T>::simplex_to_hypercube(const std::vector<T>& coord_simplex) {
  std::vector<T> coord_hyper(coord_simplex);
  T sum = static_cast<double>(0.);
  for (std::size_t i = 0; i < coord_simplex.size() - 1; i++) {
    sum += coord_simplex[i];
    coord_hyper[i] = coord_simplex[i + 1] / sum;
  }
  return coord_hyper;
}

template <typename T>
std::vector<T> Maths<T>::hypercube_to_simplex(const std::vector<T>& coord_hyper) {
  std::vector<T> coord_simplex(coord_hyper);
  T prod = static_cast<double>(1.);
}

template <typename T>
Maths<T>::~Maths() {}
