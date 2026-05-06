/**
 * @file HyperOctreeGrid.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief Class for defining a forest of (hyper)-octrees
 * @version 0.1
 * @date 2025-07-24
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

#include "Utils/UtilsForData.hpp"
#include "Utils/UtilsForDebug.hpp"
#include "Utils/UtilsForTensorialAlgebra.hpp"
#include "HyperOctree.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Utils/Maths.hpp"
#include "Utils/UtilsforUnorderedMap.hpp"
#include "Utils/Utils.hpp"

template <template <typename, std::size_t> class HyperOctreeType, typename T, std::size_t DIM>
class HyperOctreeGrid {
 private:
  std::vector<HyperOctreeType<T, DIM>> vec_hoctree;

 public:
  void add_hyperoctree(HyperOctreeType<T, DIM>&& octree) {
    this->vec_hoctree.emplace_back(std::move(octree));
  }

  std::size_t search_octree_with_point(const std::array<double, DIM>& point_to_interpolate);
  std::vector<T> search_point_in_tree(const std::array<double, DIM>& point_to_interpolate);
};

/**
 * @brief
 *
 * @tparam HyperOctreeType
 * @tparam T
 * @tparam DIM
 * @param point_to_interpolate
 * @return std::size_t
 */
template <template <typename, std::size_t> class HyperOctreeType, typename T, std::size_t DIM>
std::size_t HyperOctreeGrid<HyperOctreeType, T, DIM>::search_octree_with_point(
    const std::array<double, DIM>& point_to_interpolate) {
  for (std::size_t i = 0; i < vec_hoctree.size(); i++) {
    auto& current_hoctree = this->vec_hoctree[i];
    double half_size = current_hoctree.initial_size / 2.;
    bool inside = true;
    std::array<double, DIM> hypercube_center = current_hoctree.hypercube_center;

    for (std::size_t j = 0; j < DIM; j++) {
      double lower = hypercube_center[j] - half_size;
      double upper = hypercube_center[j] + half_size;
      if (!(lower <= point_to_interpolate[j] && point_to_interpolate[j] <= upper)) {
        inside = false;
        break;
      }
    }
    if (inside) {
      return i;
    }
  }
  std::cout << "ça crash pour : ";
  for (auto& el : point_to_interpolate) {
    std::cout << el << " , ";
  }
  std::cout << std::endl;
  return static_cast<std::size_t>(-1);
}

/**
 * @brief
 *
 * @tparam HyperOctreeType
 * @tparam T
 * @tparam DIM
 * @param point_to_interpolate
 * @return T
 */
template <template <typename, std::size_t> class HyperOctreeType, typename T, std::size_t DIM>
std::vector<T> HyperOctreeGrid<HyperOctreeType, T, DIM>::search_point_in_tree(
    const std::array<double, DIM>& point_to_interpolate) {
  std::size_t index = this->search_octree_with_point(point_to_interpolate);
  std::vector<T> res = this->vec_hoctree[index].find_hypercube(point_to_interpolate);
  return res;
}
