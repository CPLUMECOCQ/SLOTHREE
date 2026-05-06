/**
 * @file MortonIndex.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief 
 * @version 0.1
 * @date 2025-10-15
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once
#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Utils/Maths.hpp"
#include "Utils/Utils.hpp"

template <typename T, typename V, typename I, std::size_t DIM>
class MortonIndex {
 private:
 public:
  inline static T vect_indices_to_index(const V& ind, I r_max);
  inline static V index_to_vect_indices(const T& morton, I r_max);
};

/**
 * @brief
 *
 * @tparam T
 * @tparam DIM
 * @param ind
 * @param r_max
 * @return T
 */
template <typename T, typename V, typename I, std::size_t DIM>
T MortonIndex<T, V, I, DIM>::vect_indices_to_index(const V& ind, I r_max) {
  // Catch_Time_Section("MortonIndex::vect_indices_to_index");

  T morton = 0;
  // r_max += 1;
  for (I bit = 0; bit < r_max; ++bit) {
    for (std::size_t d = 0; d < DIM; ++d) {
      morton |= ((ind[d] >> (r_max - 1 - bit)) & 1ULL) << (bit * DIM + d);
    }
  }
  // std::cout << "morton : " << morton << std::endl;
  return morton;
}

template <typename T, typename V, typename I, std::size_t DIM>
V MortonIndex<T, V, I, DIM>::index_to_vect_indices(const T& morton, I r_max) {
  V indices{};
  indices.fill(0);
  // r_max += 1;
  for (I bit = 0; bit < r_max; ++bit) {
    for (std::size_t d = 0; d < DIM; ++d) {
      T mask = 1ULL << (bit * DIM + d);
      if (morton & mask) {
        indices[d] |= 1 << (r_max - 1 - bit);
      }
    }
  }
  return indices;
}
