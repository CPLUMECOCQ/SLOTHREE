/**
 * @file LeafOffset.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief Course designed to reduce the computational cost of binary search using subviews
 * @version 0.1
 * @date 2025-10-22
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
#include <span>
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

template <typename T>
class LeafOffset {
 private:
 public:
  using iter = std::vector<T>::const_iterator;
  std::vector<T> index;
  std::vector<std::span<const T>> offset;
  void sort_data();
  void fill_containers(const std::vector<T>& index_, const std::vector<T>& vec_val_indx,
                       const std::vector<std::uint64_t>& begin_indx,
                       const std::vector<std::uint64_t>& end_indx);

  LeafOffset();
};


/**
 * @brief Sort data by ascending index without loosing offset
 * 
 * @tparam T 
 */
template <typename T>
void LeafOffset<T>::sort_data() {
  std::vector<T> indices(this->index.size());
  std::iota(indices.begin(), indices.end(), 0);  // indices[i] = i

  std::sort(indices.begin(), indices.end(), [this](std::size_t i1, std::size_t i2) {
    return (this->index)[i1] < (this->index)[i2];
  });  // sort by indx

  std::vector<std::uint64_t> index_sorted(this->index.size());  // temp vec
  std::vector<std::span<const T>> offset_sorted(this->offset.size());

  // fill the temp vec by sorted ind
  std::transform(indices.begin(), indices.end(), index_sorted.begin(),
                 [this](std::size_t i) { return (this->index)[i]; });
  std::transform(indices.begin(), indices.end(), offset_sorted.begin(),
                 [this](std::size_t i) { return (this->offset)[i]; });

  // fill real vec with sorted data
  this->index = std::move(index_sorted);
  this->offset = std::move(offset_sorted);
}


/**
 * @brief Fill structure to use offset
 * 
 * @tparam T 
 * @param index_ 
 * @param vec_val_indx 
 * @param begin_indx 
 * @param end_indx 
 */
template <typename T>
void LeafOffset<T>::fill_containers(const std::vector<T>& index_,
                                    const std::vector<T>& vec_val_indx,
                                    const std::vector<std::uint64_t>& begin_indx,
                                    const std::vector<std::uint64_t>& end_indx) {
  std::size_t vecSize = begin_indx.size();
  // std::cout << "vec_val_indx.data() = " << vec_val_indx.data() << std::endl;

  this->offset.resize(vecSize);
  this->index = index_;
  for (std::uint64_t i = 0; i < begin_indx.size(); i++) {
    std::size_t size = static_cast<std::size_t>(end_indx[i] - begin_indx[i]);
    offset[i] = std::span<const T>(&vec_val_indx[begin_indx[i]], size);  // std::transform ???
  }
}

/**
 * @brief Construct a new Leaf Offset< T>:: Leaf Offset object
 *
 * @tparam T
 * @param index_
 * @param vec_val_indx
 * @param begin_indx
 * @param end_indx
 */
template <typename T>
LeafOffset<T>::LeafOffset() {}
