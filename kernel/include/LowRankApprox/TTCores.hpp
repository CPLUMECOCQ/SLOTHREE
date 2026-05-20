/**
 * @file TTCores.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief
 * @version 0.1
 * @date 2026-01-27
 *
 * @copyright Copyright (c) 2026
 *
 */
#include <vector>

#include "IO/HDF54Sloth.hpp"
#include "Utils/UtilsForTensorialAlgebra.hpp"
#pragma once

template <std::size_t DIM>
class TTCores {
 private:
  std::array<FlattenedTensor<double>, DIM> cores;    // data structures
  std::array<std::vector<std::size_t>, DIM> shapes;  // precompute
  std::vector<double> temp;
  std::vector<double> new_tmp;
  std::array<std::vector<double>, DIM> coord_grid;
  double normalization_factor;

 public:
  struct Weights {
    std::array<double, DIM> wk;
    std::array<double, DIM> wkp1;
  };
  Weights weights;

  TTCores();
  void import_cores_from_hdf5(std::string filename, std::string var,
                              const std::vector<std::string>& name_coord_grid);

  double get_tt_element(const std::vector<size_t>& point);

  double compute_interpolation(const std::array<double, DIM>& point_to_interpolate);
  double compute_TT_interpolation(const std::array<std::size_t, DIM>& indices);
  std::array<std::size_t, DIM> compute_weight_for_TT_interpolation(
      const std::array<double, DIM>& point_to_interpolate);
  double get_value_on_grid(const std::array<double, DIM>& point_to_interpolate);

  ~TTCores();
};

/**
 * @brief
 *
 * @tparam DIM
 * @param point
 * @return double
 */
template <std::size_t DIM>
double TTCores<DIM>::get_value_on_grid(const std::array<double, DIM>& point_to_interpolate) {
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
std::array<std::size_t, DIM> TTCores<DIM>::compute_weight_for_TT_interpolation(
    const std::array<double, DIM>& point_to_interpolate) {
  std::array<std::size_t, DIM> lower_indices;
  double x0, x1, dx, pd;
  for (std::size_t d = 0; d < DIM; d++) {
    pd = std::clamp(point_to_interpolate[d], coord_grid[d].front(), coord_grid[d].back());
    auto it = std::lower_bound(coord_grid[d].begin(), coord_grid[d].end(), pd);
    // lower_indices[d] = it - coord_grid[d].begin();
    auto idx = it - coord_grid[d].begin();
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
    dx = x1 - x0;
    weights.wkp1[d] = (pd - x0) / dx;
    // weights.wkp1[d] = dx > 1e-10 ? ((point_to_interpolate[d] - x0) / dx) : 0.;
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
double TTCores<DIM>::compute_TT_interpolation(const std::array<std::size_t, DIM>& indices) {
  std::size_t r0 = shapes[0][2];
  std::fill(temp.begin(), temp.end(), 0.0);
  temp[0] = 1.;
  double interpolated_core;
  for (std::size_t c = 0; c < DIM; c++) {  // loop on dimension (or core)
    // std::vector<std::size_t> shape = this->cores[c].get_shape();
    // std::size_t r_prev = shape[0];
    // std::size_t cu_dim = shape[1];
    // std::size_t r_next = shape[2];
    std::fill(new_tmp.begin(), new_tmp.end(), 0.0);

    for (std::size_t i = 0; i < shapes[c][0]; i++) {    // loop on r_prev
      for (std::size_t j = 0; j < shapes[c][2]; j++) {  // loop on r_next
        // G[i_k] = w_{i_k} * G[i_k] +  w_{i_k+1} * G[i_k+1]
        interpolated_core = weights.wk[c] * cores[c].evaluate({i, indices[c], j}) +
                            weights.wkp1[c] * cores[c].evaluate({i, indices[c] + 1, j});
        new_tmp[j] += temp[i] * interpolated_core;
      }
    }
    temp.swap(new_tmp);
  }
  return (normalization_factor * temp[0]);
};

/**
 * @brief
 *
 * @tparam DIM
 * @param filename
 * @param var
 */
template <std::size_t DIM>
void TTCores<DIM>::import_cores_from_hdf5(std::string filename, std::string var,
                                          const std::vector<std::string>& name_coord_grid) {
  std::size_t maxr = 0;
  for (std::size_t i = 0; i < DIM; ++i) {
    std::string dataset_name = var + "/core_" + std::to_string(i);
    HDF54Sloth<std::monostate>::get_data_from_HDF5(filename, dataset_name, cores[i]);
    shapes[i] = cores[i].get_shape();
    maxr = std::max(maxr, shapes[i][2]);

    std::string n = name_coord_grid[i];
    HDF54Sloth<std::monostate>::get_data_from_HDF5(filename, n, coord_grid[i]);

    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group base_grp = file.openGroup(var);
    int num_attrs = base_grp.getNumAttrs();
    for (int i = 0; i < num_attrs; ++i) {
      H5::Attribute attr = base_grp.openAttribute(i);
      std::string name_attr = attr.getName();
      if (name_attr == "vmax") {
        double val;
        attr.read(H5::PredType::NATIVE_DOUBLE, &val);
        this->normalization_factor = static_cast<double>(val);
        std::cout << "ICI FACTEUR" << this->normalization_factor << std::endl;
      }
    }
  }
  new_tmp.resize(maxr);
  temp.resize(maxr);
  std::cout << "Cores loaded \n";
}

/**
 * @brief
 *
 * @tparam DIM
 * @param point
 * @return double
 */
template <std::size_t DIM>
double TTCores<DIM>::get_tt_element(const std::vector<std::size_t>& point) {
  std::size_t r0 = shapes[0][2];
  std::fill(temp.begin(), temp.end(), 0.0);
  for (std::size_t a = 0; a < r0; ++a) {
    temp[a] = cores[0].evaluate({0, point[0], a});
  }
  for (std::size_t c = 1; c < DIM; c++) {  // loop on dimension (or core)
    // std::vector<std::size_t> shape = this->cores[c].get_shape();
    // std::size_t r_prev = shape[0];
    // std::size_t cu_dim = shape[1];
    // std::size_t r_next = shape[2];
    std::fill(new_tmp.begin(), new_tmp.end(), 0.0);

    for (std::size_t i = 0; i < shapes[c][0]; i++) {    // loop on r_prev
      for (std::size_t j = 0; j < shapes[c][2]; j++) {  // loop on r_next
        new_tmp[j] += temp[i] * cores[c].evaluate({i, point[c], j});
      }
    }
    temp.swap(new_tmp);
  }
  return temp[0];
};

/**
 * @brief
 *
 * @tparam DIM
 * @param point
 * @return double
 */
template <std::size_t DIM>
double TTCores<DIM>::compute_interpolation(const std::array<double, DIM>& point_to_interpolate) {
  std::array<std::size_t, DIM> lower_indices;
  std::array<double, DIM> alpha;
  for (std::size_t d = 0; d < DIM; d++) {
    auto it = std::lower_bound(coord_grid[d].begin(), coord_grid[d].end(), point_to_interpolate[d]);
    lower_indices[d] = it - coord_grid[d].begin();
    // std::cout << lower_indices[d] << std::endl;
    double x0 = coord_grid[d][lower_indices[d]];
    double x1 = coord_grid[d][lower_indices[d] + 1];
    double dx = x1 - x0;
    alpha[d] = (point_to_interpolate[d] - x0) / dx;
  }
  double val_interpolated = 0.0;
  std::vector<std::size_t> indices(DIM);

  for (std::size_t i = 0; i < (1 << DIM); ++i) {
    double weight = 1.0;
    for (std::size_t d = 0; d < DIM; d++) {
      bool condition = (i & (1 << d)) != 0;

      weight *= condition ? alpha[d] : (1.0 - alpha[d]);
      indices[d] = lower_indices[d] + condition;
    }
    val_interpolated += weight * this->get_tt_element(indices);
  }
  return val_interpolated;
};

/**
 * @brief Construct a new TTCores<DIM>::TTCores object
 *
 * @tparam DIM
 */
template <std::size_t DIM>
TTCores<DIM>::TTCores() {}

/**
 * @brief Destroy the TTCores<DIM>::TTCores object
 *
 * @tparam DIM
 */
template <std::size_t DIM>
TTCores<DIM>::~TTCores() {}