/**
 * @file HyperOctree.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief Structure for explicit (hyper-)octree
 * @version 0.1
 * @date 2025-05-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "MortonIndex.hpp"
#include "Utils/UtilsForData.hpp"
#include "Utils/UtilsForDebug.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Utils/Maths.hpp"
#include "Utils/UtilsforUnorderedMap.hpp"

const double PRECISION = 1e-15;
const double EPS = 1e-15;

template <typename T, std::size_t DIM>
class HyperOctree {
 private:
 public:
  using VectorisationIndex =
      MortonIndex<std::uint64_t, std::array<std::uint64_t, DIM>, std::uint_fast8_t, DIM>;
  bool isLeaf;
  std::uint_fast8_t current_level;
  std::uint_fast8_t max_depth;
  double half_size;
  double initial_size;

  std::array<double, DIM> hypercube_center;
  std::array<double, DIM> hypercube_center_init;

  std::optional<std::array<std::unique_ptr<HyperOctree<T, DIM>>, (1 << DIM)>> children;

  std::shared_ptr<std::unordered_map<std::array<int, DIM>, double, HashArrayIndex<DIM>>> map_values;
  std::shared_ptr<std::vector<double>> values;
  std::shared_ptr<std::vector<std::uint64_t>> index;

  // functions
  inline std::array<std::uint64_t, DIM> coord_to_index(const std::array<double, DIM>& coord);
  inline std::array<double, DIM> index_to_coord(const std::array<std::uint64_t, DIM>& index);

  void get_data_from_HDF5(const H5std_string& file_name, const std::string& dataset_name);

  void reconstruct_from_map();

  void export_tree_as_csv(std::ofstream& output_file);

  inline T search_point_in_tree(std::array<double, DIM> point);

  inline std::array<std::array<double, DIM>, (1 << DIM)> get_corners_coords();

  void apply_scalling(const std::function<double(double)>& f);

  inline T evaluate(const std::uint64_t& mortonIndex);
  inline T evaluate(const std::array<double, DIM>& point);

  HyperOctree(const std::uint_fast8_t& current_level_, const std::uint_fast8_t& max_depth_,
              const double& half_size_, const double& initial_size_,
              const std::array<double, DIM>& hypercube_center_,
              const std::array<double, DIM>& hypercube_center_init_,
              std::shared_ptr<std::vector<std::uint64_t>> index_ = nullptr,
              std::shared_ptr<std::vector<double>> values_ = nullptr);

  HyperOctree();
  HyperOctree(HyperOctree&& other) noexcept = default;
  HyperOctree& operator=(HyperOctree&& other) noexcept = default;
  HyperOctree(const HyperOctree<T, DIM>& other);
  ~HyperOctree();
};

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param mortonIndex
 * @return T
 */
template <typename T, std::size_t DIM>
T HyperOctree<T, DIM>::evaluate(const std::uint64_t& mortonIndex) {
  // Catch_Time_Section("HyperOctree::evaluate2");
  auto it = std::lower_bound(this->index->begin(), this->index->end(), mortonIndex);
  if (it == index->end() || *it != mortonIndex) {
    throw std::out_of_range("alors la bonne chance pour débugger");
  }
  std::uint64_t indx = std::distance(this->index->begin(), it);
  T res = (*this->values)[indx];
  return res;
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param point
 * @return T
 */
template <typename T, std::size_t DIM>
T HyperOctree<T, DIM>::evaluate(const std::array<double, DIM>& point) {
  // Catch_Time_Section("HyperOctree::evaluate");
  std::array<std::uint64_t, DIM> vec_indices = this->coord_to_index(point);
  std::uint64_t mortonIndex =
      VectorisationIndex::vect_indices_to_index(vec_indices, this->max_depth + 1);
  T res = this->evaluate(mortonIndex);
  return res;
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param point
 * @return T
 */
template <typename T, std::size_t DIM>
T HyperOctree<T, DIM>::search_point_in_tree(std::array<double, DIM> point) {
  // Catch_Time_Section("HyperOctree::search_point_in_tree");

  for (std::size_t i = 0; i < DIM; ++i) {
    double delta = point[i] - hypercube_center[i];
    if (std::abs(delta) > (half_size + 1e-15)) {
      point[i] = hypercube_center[i] + (delta < 0.0 ? -half_size : half_size);
    }
  }
  T val_interpolated;
  double alpha_i;

  if (!this->isLeaf) {
    std::size_t child_index = 0;
    for (std::size_t i = 0; i < DIM; ++i) {
      if (point[i] > hypercube_center[i]) {
        child_index |= (1 << (DIM - 1 - i));
      }
    }
    val_interpolated = (*children)[child_index]->search_point_in_tree(point);
  } else {
    std::array<double, DIM> alpha;

    for (std::size_t i = 0; i < DIM; ++i) {
      double x0 = this->hypercube_center[i] - this->half_size;
      double x1 = this->hypercube_center[i] + this->half_size;
      alpha[i] = (point[i] - x0) / (x1 - x0);
    }

    val_interpolated = 0.;
    std::array<std::array<double, DIM>, (1 << DIM)> corners_coords = this->get_corners_coords();
    double weight;

    for (auto& corner_coords : corners_coords) {
      T local_value = this->evaluate(corner_coords);
      weight = 1.;
      for (std::size_t i = 0; i < DIM; ++i) {
        bool condition = corner_coords[i] < this->hypercube_center[i];
        alpha_i = alpha[i];
        weight *= condition ? (1 - alpha_i) : alpha_i;
      }
      // val_interpolated += weight * (*map_values)[indices_corner];
      val_interpolated += weight * local_value;
    }
  }
  return val_interpolated;
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 */
template <typename T, std::size_t DIM>
std::array<std::array<double, DIM>, (1 << DIM)> HyperOctree<T, DIM>::get_corners_coords() {
  // Catch_Time_Section("HyperOctree::get_corners_coords");

  std::array<std::array<double, DIM>, 1 << DIM> corners;

  for (std::size_t i = 0; i < (1 << DIM); ++i) {
    std::array<double, DIM> corner(this->hypercube_center);
    for (std::size_t j = 0; j < DIM; ++j) {
      std::size_t bit = (i >> (DIM - 1 - j)) & 1;
      corner[j] += (bit ? this->half_size : -half_size);
      corner[j] = Maths<double>::round(corner[j], PRECISION);
    }
    corners[i] = std::move(corner);
  }
  return corners;
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param f
 */
template <typename T, std::size_t DIM>
void HyperOctree<T, DIM>::apply_scalling(const std::function<double(double)>& f) {
  for (auto& pair : *map_values) {
    pair.second = f(pair.second);
  }
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param output_file
 */
template <typename T, std::size_t DIM>
void HyperOctree<T, DIM>::export_tree_as_csv(std::ofstream& output_file) {
  // Catch_Time_Section("HyperOctree::export_tree_as_csv");

  output_file << static_cast<int>(this->current_level) << " , ";
  output_file << this->isLeaf << " , ";

  output_file << "[";
  for (size_t i = 0; i < hypercube_center.size(); ++i) {
    output_file << hypercube_center[i];
    if (i < hypercube_center.size() - 1) output_file << ", ";
  }
  output_file << "]" << " , " << this->half_size << std::endl;

  if (!this->isLeaf && this->current_level <= this->max_depth) {
    std::size_t number_of_children(1 << DIM);
    for (std::size_t i = 0; i < number_of_children; ++i) {
      (*children)[i]->export_tree_as_csv(output_file);
    }
  }
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 */
template <typename T, std::size_t DIM>
void HyperOctree<T, DIM>::reconstruct_from_map() {
  // Catch_Time_Section("HyperOctree::reconstruct_from_map");

  if (current_level >= max_depth) return;
  std::array<std::uint64_t, DIM> index_center = coord_to_index(this->hypercube_center);
  std::uint64_t mortonIndex =
      VectorisationIndex::vect_indices_to_index(index_center, this->max_depth + 1);

  auto it = std::lower_bound(this->index->begin(), this->index->end(), mortonIndex);
  if (it != index->end() && *it == mortonIndex) {
    if (it == index->end() || *it != mortonIndex) return;
    if (!children.has_value()) {
      children.emplace();
    }

    for (std::size_t i = 0; i < (1 << DIM); ++i) {
      std::array<double, DIM> child_hypercube_center(this->hypercube_center);
      double child_half_size = this->half_size / 2.;
      this->isLeaf = false;

      for (std::size_t j = 0; j < DIM; ++j) {
        std::size_t bit = (i >> (DIM - 1 - j)) & 1;
        child_hypercube_center[j] += (bit ? child_half_size : -child_half_size);
        child_hypercube_center[j] = Maths<double>::round(child_hypercube_center[j], PRECISION);
      }

      std::uint_fast8_t next_level = this->current_level + 1;
      auto child = std::make_unique<HyperOctree<T, DIM>>(
          next_level, this->max_depth, child_half_size, this->initial_size, child_hypercube_center,
          this->hypercube_center_init, this->index, this->values);
      (*children)[i] = std::move(child);
      (*children)[i]->reconstruct_from_map();
    }
  }
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param file_name
 * @param dataset_name
 */
template <typename T, std::size_t DIM>
void HyperOctree<T, DIM>::get_data_from_HDF5(const H5std_string& file_name,
                                             const std::string& dataset_name) {
  bool is_hdf5 = H5::H5File::isHdf5(file_name);
  H5::H5File file(file_name, H5F_ACC_RDONLY);
  std::cout << file_name << "  " << dataset_name << std::endl;

  H5::DataSet key_dataset = file.openDataSet(dataset_name + "/key");
  H5::DataSet val_dataset = file.openDataSet(dataset_name + "/values");

  H5::DataSpace key_space = key_dataset.getSpace();
  hsize_t dims[2];
  int rank = key_space.getSimpleExtentDims(dims);
  size_t num_entries = dims[0];
  size_t key_size = dims[1];

  std::vector<int> flat_keys(num_entries * key_size);
  std::vector<double> values(num_entries);

  key_dataset.read(flat_keys.data(), H5::PredType::NATIVE_INT);
  val_dataset.read(values.data(), H5::PredType::NATIVE_DOUBLE);

  for (size_t i = 0; i < num_entries; ++i) {
    std::array<int, DIM> key;
    for (size_t j = 0; j < key_size; ++j) {
      key[j] = flat_keys[i * key_size + j];
    }
    map_values->emplace(key, values[i]);
  }

  file.close();
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param coord
 * @return std::array<std::uint64_t, DIM>
 */
template <typename T, std::size_t DIM>
std::array<std::uint64_t, DIM> HyperOctree<T, DIM>::coord_to_index(
    const std::array<double, DIM>& coord) {
  // Catch_Time_Section("HyperOctree::coord_to_index");

  std::array<std::uint64_t, DIM> index;
  // double h = this->initial_size / (1ULL << (this->max_depth + 1) - 1);
  double h = this->initial_size / (1ULL << this->max_depth);

  std::uint64_t max_index = (1ULL << (this->max_depth + 1)) - 1;

  for (std::size_t i = 0; i < DIM; ++i) {
    double a = this->hypercube_center_init[i] - this->initial_size / 2.0;
    double b = this->hypercube_center_init[i] + this->initial_size / 2.0;
    index[i] = static_cast<std::uint64_t>(std::round((coord[i] - a) / h));
    index[i] = std::min(index[i], max_index);
  }
  return index;
}

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param index
 * @return std::array<double, DIM>
 */
template <typename T, std::size_t DIM>
std::array<double, DIM> HyperOctree<T, DIM>::index_to_coord(
    const std::array<std::uint64_t, DIM>& index) {
  // Catch_Time_Section("HyperOctree::index_to_coord");

  std::array<double, DIM> coord;
  // double h = this->initial_size / (1ULL << (this->max_depth + 1) - 1);
  double h = this->initial_size / (1ULL << this->max_depth);

  for (std::size_t i = 0; i < DIM; ++i) {
    double a = this->hypercube_center_init[i] - this->initial_size / 2.0;
    coord[i] = a + index[i] * h;
  }
  return coord;
}

/**
 * @brief Construct a new Hyper Octree< T,  D I M>:: Hyper Octree object
 *
 * @tparam T
 * @tparam DIM
 * @param current_level_
 * @param max_depth_
 * @param half_size_
 * @param hypercube_center_
 * @param shared_map
 */
template <typename T, std::size_t DIM>
HyperOctree<T, DIM>::HyperOctree(const std::uint_fast8_t& current_level_,
                                 const std::uint_fast8_t& max_depth_, const double& half_size_,
                                 const double& initial_size_,
                                 const std::array<double, DIM>& hypercube_center_,
                                 const std::array<double, DIM>& hypercube_center_init_,
                                 std::shared_ptr<std::vector<std::uint64_t>> index_,
                                 std::shared_ptr<std::vector<double>> values_)
    : current_level(current_level_),
      max_depth(max_depth_),
      half_size(half_size_),
      initial_size(initial_size_),
      hypercube_center(hypercube_center_),
      hypercube_center_init(hypercube_center_init_),
      index(index_),
      values(values_),
      isLeaf(true) {
  if (!index) {
    index = std::make_shared<std::vector<uint64_t>>();
  }
  if (!values) {
    values = std::make_shared<std::vector<double>>();
  }
}

/**
 * @brief Construct a new Hyper Octree< T,  D I M>:: Hyper Octree object
 *
 * @tparam T
 * @tparam DIM
 */
template <typename T, std::size_t DIM>
HyperOctree<T, DIM>::HyperOctree()
    : current_level(0),
      max_depth(0),
      half_size(0.),
      initial_size(0.),
      hypercube_center({}),
      hypercube_center_init({}),
      isLeaf(true),
      index(std::make_shared<std::vector<uint64_t>>()),
      values(std::make_shared<std::vector<double>>()) {}

/**
 * @brief Copy constructor for a new Hyper Octree<T,DIM>::HyperOctree object
 *
 * @tparam T
 * @tparam DIM
 * @param other
 */
template <typename T, std::size_t DIM>
HyperOctree<T, DIM>::HyperOctree(const HyperOctree<T, DIM>& other)
    : isLeaf(other.isLeaf),
      current_level(other.current_level),
      max_depth(other.max_depth),
      half_size(other.half_size),
      initial_size(other.initial_size),
      hypercube_center(other.hypercube_center),
      hypercube_center_init(other.hypercube_center_init),
      index(other.index),
      values(other.values) {
  if (other.children.has_value()) {
    children.emplace();
    for (std::size_t i = 0; i < (1 << DIM); ++i) {
      if ((*other.children)[i]) {
        (*children)[i] = std::make_unique<HyperOctree<T, DIM>>(*(*other.children)[i]);
      } else {
        (*children)[i] = nullptr;
      }
    }
  }
}

/**
 * @brief Destroy the Hyper Octree< T,  D I M>:: Hyper Octree object
 *
 * @tparam T
 * @tparam DIM
 */
template <typename T, std::size_t DIM>
HyperOctree<T, DIM>::~HyperOctree() {}
