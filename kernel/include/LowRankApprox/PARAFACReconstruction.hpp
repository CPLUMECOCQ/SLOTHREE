/**
 * @file PARAFACReconstruction.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief
 * @version 0.1
 * @date 2026-05-21
 *
 * @copyright Copyright (c) 2026
 *
 */
#include <vector>

#include "IO/HDF54Sloth.hpp"
#include "Utils/UtilsForTensorialAlgebra.hpp"
#pragma once

template <std::size_t DIM>
class PARAFACReconstruction {
 private:
  std::array<FlattenedTensor<double>, DIM> cores;  // data structures
  std::array<double, DIM> shapes;                  // precompute
  std::vector<double> temp;
  std::vector<double> new_tmp;
  std::array<std::vector<double>, DIM> coord_grid;
  std::vector<double> normalization_factor;
  std::size_t rank;

 public:
  struct Weights {
    std::array<double, DIM> wk;
    std::array<double, DIM> wkp1;
  };
  Weights weights;

  PARAFACReconstruction();
  void import_cores_from_hdf5(std::string filename, std::string var,
                              const std::vector<std::string>& name_coord_grid);

  double get_tt_element(const std::vector<size_t>& point);

  double compute_TT_interpolation(const std::array<std::size_t, DIM>& indices);
  std::array<std::size_t, DIM> compute_weight_for_TT_interpolation(
      const std::array<double, DIM>& point_to_interpolate);
  double get_value_on_grid(const std::array<double, DIM>& point_to_interpolate);

  ~PARAFACReconstruction();
};

/**
 * @brief
 *
 * @tparam DIM
 * @param point
 * @return double
 */
template <std::size_t DIM>
double PARAFACReconstruction<DIM>::get_value_on_grid(
    const std::array<double, DIM>& point_to_interpolate) {
  std::array<std::size_t, DIM> indices =
      this->compute_weight_for_TT_interpolation(point_to_interpolate);
  return this->compute_TT_interpolation(indices);
};

/**
 * @brief
 *
 * @tparam DIM
 * @param point
 * @return double
 */
template <std::size_t DIM>
std::array<std::size_t, DIM> PARAFACReconstruction<DIM>::compute_weight_for_TT_interpolation(
    const std::array<double, DIM>& point_to_interpolate) {
  std::array<std::size_t, DIM> lower_indices;
  double x0, x1, dx, pd;
  for (std::size_t d = 0; d < DIM; ++d) {
    pd = std::clamp(point_to_interpolate[d], coord_grid[d].front(), coord_grid[d].back());
    auto it = std::lower_bound(coord_grid[d].begin(), coord_grid[d].end(), pd);
    // lower_indices[d] = it - coord_grid[d].begin();

    auto idx = it - coord_grid[d].begin();

    if (idx > 0) {
      --idx;
    }

    if (idx >= coord_grid[d].size() - 1) {
      idx = coord_grid[d].size() - 2;
      std::cout << "[TT interpolation] Clamp index on dim " << d
                << " : point = " << point_to_interpolate[d] << ", idx = " << idx
                << ", max = " << coord_grid[d].size() - 2 << std::endl;
    };
    lower_indices[d] = idx;
    // std::cout << d << "," << lower_indices[d] << std::endl;
    x0 = coord_grid[d][lower_indices[d]];
    x1 = coord_grid[d][lower_indices[d] + 1];
    // std::cout << x0 << " < " << point_to_interpolate[d] << " < " << x1 << std::endl;
    dx = x1 - x0;
    // weights.wkp1[d] = (pd - x0) / dx;
    weights.wkp1[d] = dx > 1e-10 ? ((pd - x0) / dx) : 0.;
    weights.wk[d] = 1. - weights.wkp1[d];
  }
  return lower_indices;
};

/**
 * @brief
 *
 * @tparam DIM
 * @param point
 * @return double
 */
template <std::size_t DIM>
double PARAFACReconstruction<DIM>::compute_TT_interpolation(
    const std::array<std::size_t, DIM>& indices) {
  double res = 0.;
  double tempres = 0.;
  for (std::size_t r = 0; r < this->rank; ++r) {
    tempres = 1.;
    for (std::size_t d = 0; d < DIM; ++d) {
      tempres *= weights.wk[d] * cores[d].evaluate({indices[d], r}) +
                 weights.wkp1[d] * cores[d].evaluate({indices[d] + 1, r});
    }
    res += this->normalization_factor[r] * tempres;
  }
  // std::cout << res << std::endl;

  return res;
};

/**
 * @brief
 *
 * @tparam DIM
 * @param filename
 * @param var
 */
template <std::size_t DIM>
void PARAFACReconstruction<DIM>::import_cores_from_hdf5(
    std::string filename, std::string var, const std::vector<std::string>& name_coord_grid) {
  std::size_t maxr = 0;

  std::string factorname = var + "/weights";
  HDF54Sloth<std::monostate>::get_data_from_HDF5(filename, factorname, this->normalization_factor);

  for (std::size_t i = 0; i < DIM; ++i) {
    std::string dataset_name = var + "/factor_" + std::to_string(i);
    HDF54Sloth<std::monostate>::get_data_from_HDF5(filename, dataset_name, cores[i]);
    shapes[i] = cores[i].get_shape()[0];  // calculer plusieurs fois mais pas grave
    rank = cores[i].get_shape()[1];       // calculer plusieurs fois mais pas grave

    std::string n = name_coord_grid[i];
    HDF54Sloth<std::monostate>::get_data_from_HDF5(filename, n, coord_grid[i]);
    // std::cout << DIM << " d = " << i << std::endl;

    // for (auto& el : coord_grid[i]) {
    //   std::cout << el << ",";
    // }
    // std::cout << std::endl;
  }
}

/**
 * @brief
 *
 * @tparam DIM
 * @param point
 * @return double
 */
template <std::size_t DIM>
double PARAFACReconstruction<DIM>::get_tt_element(const std::vector<std::size_t>& point) {
  double res = 0.;
  double tempres = 0.;
  for (std::size_t r = 0; r < this->rank; ++r) {
    tempres = 1.;
    for (std::size_t d = 0; d < DIM; ++d) {
      tempres *= cores[d].evaluate({point[d], r});
    }
    res += this->normalization_factor[r] * tempres;
  }
  std::cout << res << std::endl;
  return res;
};

/**
 * @brief Construct a new TTCores<DIM>::TTCores object
 *
 * @tparam DIM
 */
template <std::size_t DIM>
PARAFACReconstruction<DIM>::PARAFACReconstruction() {}

/**
 * @brief Destroy the TTCores<DIM>::TTCores object
 *
 * @tparam DIM
 */
template <std::size_t DIM>
PARAFACReconstruction<DIM>::~PARAFACReconstruction() {}