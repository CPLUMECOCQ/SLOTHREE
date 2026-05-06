/**
 * @file ImplicitHyperOctree.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief
 * @version 0.1
 * @date 2025-10-20
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
#include <unordered_map>
#include <vector>

#include "HyperOctree.hpp"
#include "LeafOffset.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Utils/Maths.hpp"
#include "Utils/Utils.hpp"
#include "Utils/UtilsForData.hpp"
#include "Utils/UtilsForDebug.hpp"
#include "Utils/UtilsForTensorialAlgebra.hpp"
#include "Utils/UtilsforUnorderedMap.hpp"

template <typename T, std::size_t DIM>
class ImplicitHyperOctree {
 private:
  using VectorisationIndex =
      MortonIndex<std::uint64_t, std::array<std::uint64_t, DIM>, std::uint_fast8_t, DIM>;

  // std::unordered_map<std::string, std::shared_ptr<std::vector<double>>>
  //     values;  // transformer en  std::vector< std::shared_ptr<std::vector<double>>> values; et
  // // faire la correspondance avec this->variables_names pour optimiser
  std::vector<std::shared_ptr<std::vector<double>>> values;
  std::shared_ptr<std::vector<std::uint64_t>> index;
  std::shared_ptr<std::vector<std::vector<std::uint64_t>>> leaves_indx;
  std::shared_ptr<std::vector<LeafOffset<std::uint64_t>>> indxWithOffset;
  bool no_refinement;

 public:
  std::vector<std::string> variables_names;  // A UTILISER POUR BOUCLER LA DESSUS
  std::array<double, DIM> hypercube_center;
  std::uint_fast8_t max_depth;
  double initial_size;

  std::vector<T> find_hypercube(const std::array<double, DIM>& point);
  std::vector<T> interpolate_on_first_lvl(const std::array<double, DIM>& point);

  inline std::array<std::uint64_t, DIM> coord_to_index(const std::array<double, DIM>& coord);
  inline std::array<double, DIM> index_to_coord(const std::array<std::uint64_t, DIM>& index);

  void read_file_for_map_values(std::vector<std::vector<std::uint64_t>>& begin_offset,
                                std::vector<std::vector<std::uint64_t>>& end_offset,
                                const std::string& filename = "", const double& temperature = 0.,
                                const std::string& subOctreeName = "",
                                const std::string& family = "f(x)");
  ImplicitHyperOctree(std::vector<std::shared_ptr<std::vector<double>>> values_,
                      std::shared_ptr<std::vector<std::uint64_t>> index_,
                      std::shared_ptr<std::vector<std::vector<std::uint64_t>>> leaves_indx_,
                      std::shared_ptr<std::vector<LeafOffset<std::uint64_t>>> indxWithOffset_,
                      const std::vector<std::string>& variables_names_);
  ~ImplicitHyperOctree();
};

template <typename T, std::size_t DIM>
std::vector<T> ImplicitHyperOctree<T, DIM>::interpolate_on_first_lvl(
    const std::array<double, DIM>& point) {
  std::size_t variables_names_size = this->variables_names.size();
  std::vector<T> val_interpolated(variables_names_size, 0.);
  std::vector<T> local_value(variables_names_size);
  double alpha_i;
  std::array<double, DIM> alpha;
  double weight;
  std::array<std::uint64_t, DIM> vec_indx_center = this->coord_to_index(this->hypercube_center);
  std::uint64_t morton_center =
      VectorisationIndex::vect_indices_to_index(vec_indx_center, this->max_depth + 2);
  auto& vec_lvl = (*this->indxWithOffset)[0].index;

  std::array<double, DIM> corner;
  std::array<std::array<double, DIM>, 1 << DIM> vec_corner;
  double half_size = this->initial_size / 2.;
  for (std::size_t k = 0; k < (1ULL << DIM); ++k) {
    corner = this->hypercube_center;
    for (std::size_t j = 0; j < DIM; ++j) {
      std::size_t bit_corner = (k >> (DIM - 1 - j)) & 1ULL;
      corner[j] += (bit_corner ? half_size : -half_size);
    }
    vec_corner[k] = std::move(corner);
  }

  for (std::size_t i = 0; i < DIM; ++i) {
    double x0 = this->hypercube_center[i] - this->initial_size / 2.0;
    alpha[i] = (point[i] - x0) / (2 * half_size);
  }
  auto& vec_index = *this->index;
  for (auto& corner_coords_ : vec_corner) {
    std::array<std::uint64_t, DIM> corner_coords = this->coord_to_index(corner_coords_);
    std::uint64_t localMortonIndex =
        VectorisationIndex::vect_indices_to_index(corner_coords, this->max_depth + 2);

    // EVALUATE

    auto it_corner = std::lower_bound(this->index->begin(), this->index->end(), localMortonIndex);
    if (it_corner == this->index->end() || *it_corner != localMortonIndex) {
      throw std::out_of_range("alors la bonne chance pour débugger");
    }

    std::uint64_t indx_global = std::distance(this->index->begin(), it_corner);

    for (std::size_t v = 0; v < variables_names_size; v++) {
      local_value[v] = (*this->values[v])[indx_global];
    }

    // END EVALUATE

    weight = 1.;
    for (std::size_t i = 0; i < DIM; ++i) {
      bool condition = corner_coords[i] < vec_indx_center[i];
      alpha_i = alpha[i];
      weight *= condition ? (1 - alpha_i) : alpha_i;
    }

    for (std::size_t v = 0; v < variables_names_size; v++) {
      val_interpolated[v] += weight * local_value[v];
    }
  }
  return val_interpolated;
}
/**
 * @brief Main function: implicitly reconstructs an octree structure and performs multilinear
 * interpolation.
 *
 * @tparam T
 * @tparam DIM
 * @param point
 * @return std::unordered_map<std::string, T>
 */
template <typename T, std::size_t DIM>
std::vector<T> ImplicitHyperOctree<T, DIM>::find_hypercube(const std::array<double, DIM>& point) {
  std::size_t variables_names_size = this->variables_names.size();
  std::vector<T> val_interpolated(variables_names_size, 0.);
  if (no_refinement) {
    val_interpolated = this->interpolate_on_first_lvl(point);
    return val_interpolated;
  } else {
    std::array<double, DIM> cell_center = this->hypercube_center;
    double cell_size = this->initial_size;
    std::array<std::uint64_t, DIM> vec_indx_center;
    // std::unordered_map<std::string, T> val_interpolated;
    std::vector<T> local_value(variables_names_size);
    double alpha_i;
    std::array<double, DIM> alpha;
    double weight;

    uint_fast8_t lvl;
    for (lvl = 0; lvl <= max_depth; ++lvl) {
      double half_size = cell_size / 2.0;
      std::uint64_t child_mask = 0;

      for (size_t d = 0; d < DIM; ++d) {
        double offset = ((point[d] >= cell_center[d]) ? (half_size / 2) : -(half_size / 2.));
        cell_center[d] += offset;
      }
      cell_size = half_size;
      half_size = cell_size / 2.;
      vec_indx_center = this->coord_to_index(cell_center);

      std::uint64_t morton_center =
          VectorisationIndex::vect_indices_to_index(vec_indx_center, this->max_depth + 2);
      auto& vec_lvl = (*this->indxWithOffset)[lvl + 1].index;

      if (!vec_lvl.empty()) {
        auto it = std::lower_bound(vec_lvl.begin(), vec_lvl.end(), morton_center);

        std::array<double, DIM> corner;
        std::array<std::array<double, DIM>, 1 << DIM> vec_corner;

        for (std::size_t k = 0; k < (1ULL << DIM); ++k) {
          corner = cell_center;
          for (std::size_t j = 0; j < DIM; ++j) {
            std::size_t bit_corner = (k >> (DIM - 1 - j)) & 1ULL;
            corner[j] += (bit_corner ? half_size : -half_size);
          }
          vec_corner[k] = std::move(corner);
        }
        if (it != vec_lvl.end() && *it == morton_center) {
          std::uint64_t OffsetIndex = std::distance(vec_lvl.begin(), it);

          for (std::size_t i = 0; i < DIM; ++i) {
            double x0 = cell_center[i] - cell_size / 2.0;
            // double x1 = cell_center[i] + cell_size / 2.0;
            // alpha[i] = (point[i] - x0) / (x1 - x0);  // alpha[i] = (point[i] - x0) / (2 *
            // half_size);
            alpha[i] = (point[i] - x0) / (2 * half_size);
          }

          auto& vec_index = *this->index;
          for (auto& corner_coords_ : vec_corner) {
            std::array<std::uint64_t, DIM> corner_coords = this->coord_to_index(corner_coords_);
            std::uint64_t localMortonIndex =
                VectorisationIndex::vect_indices_to_index(corner_coords, this->max_depth + 2);

            // EVALUATE
            auto& Indexoffset = (*this->indxWithOffset)[lvl + 1].offset[OffsetIndex];
            std::int64_t offset_global_base = Indexoffset.data() - vec_index.data();

            auto it_corner =
                std::lower_bound(Indexoffset.begin(), Indexoffset.end(), localMortonIndex);
            if (it_corner == Indexoffset.end() || *it_corner != localMortonIndex) {
              throw std::out_of_range("alors la bonne chance pour débugger");
            }
            std::uint64_t indx =
                it_corner - Indexoffset.begin();  // equivalent a std::uint64_t indx =
                                                  // std::distance(Indexoffset.begin(), it_corner);

            std::uint64_t indx_global = offset_global_base + indx;

            // auto it_corner =
            //     std::lower_bound(this->index->begin(), this->index->end(), localMortonIndex);
            // if (it_corner == this->index->end() || *it_corner != localMortonIndex) {
            //   throw std::out_of_range("alors la bonne chance pour débugger");
            // }
            // std::uint64_t indx = std::distance(this->index->begin(), it_corner);

            for (std::size_t v = 0; v < variables_names_size; v++) {
              local_value[v] = (*this->values[v])[indx_global];
            }

            // END EVALUATE

            weight = 1.;
            for (std::size_t i = 0; i < DIM; ++i) {
              bool condition = corner_coords[i] < vec_indx_center[i];
              alpha_i = alpha[i];
              weight *= condition ? (1 - alpha_i) : alpha_i;
            }

            for (std::size_t v = 0; v < variables_names_size; v++) {
              val_interpolated[v] += weight * local_value[v];
            }
          }
          return val_interpolated;
        }
      }
    }
  }
  std::cout << "ça plante dans implicitHyperOctree" << std::endl;

  return val_interpolated;
}

/**
 * @brief Need to be the same as in MultiHyperOctree. Do not modify without first discussing it with
 * C. Introïni or C. Plumecocq.
 *
 * @tparam T
 * @tparam DIM
 * @param coord
 * @return std::array<std::uint64_t, DIM>
 */
template <typename T, std::size_t DIM>
std::array<std::uint64_t, DIM> ImplicitHyperOctree<T, DIM>::coord_to_index(
    const std::array<double, DIM>& coord) {
  std::array<std::uint64_t, DIM> index;

  double h = this->initial_size / (1ULL << (this->max_depth + 1));
  std::uint64_t max_index = (1ULL << (this->max_depth + 1));

  for (std::size_t i = 0; i < DIM; ++i) {
    double a = this->hypercube_center[i] - this->initial_size / 2.0;

    double pos = (coord[i] - a) / h;
    std::uint64_t idx = static_cast<std::uint64_t>(std::round(pos));

    if (idx > max_index) idx = max_index;
    index[i] = idx;
  }
  return index;
}

/**
 * @brief Need to be the same as in MultiHyperOctree. Do not modify without first discussing it with
 * C. Introïni or C. Plumecocq.
 *
 * @tparam T
 * @tparam DIM
 * @param index
 * @return std::array<double, DIM>
 */
template <typename T, std::size_t DIM>
std::array<double, DIM> ImplicitHyperOctree<T, DIM>::index_to_coord(
    const std::array<std::uint64_t, DIM>& index) {
  // Catch_Time_Section("HyperOctree::index_to_coord");

  std::array<double, DIM> coord;
  double h = this->initial_size / (1ULL << (this->max_depth + 1));

  for (std::size_t i = 0; i < DIM; ++i) {
    double a = this->hypercube_center[i] - this->initial_size / 2.0;
    coord[i] = a + index[i] * h;
  }
  return coord;
}

/**
 * @brief Recovering data from the HDF5 file could be improved but is sufficiently effective
 for the
 * moment.
 *
 * @tparam T
 * @tparam DIM
 * @param begin_offset
 * @param end_offset
 * @param temperature
 * @param family
 */
template <typename T, std::size_t DIM>
void ImplicitHyperOctree<T, DIM>::read_file_for_map_values(
    std::vector<std::vector<std::uint64_t>>& begin_offset,
    std::vector<std::vector<std::uint64_t>>& end_offset, const std::string& filename,
    const double& temperature, const std::string& subOctreeName, const std::string& family) {
  // std::string file_name = "res" + std::to_string(temperature) + ".h5";
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("Fichier HDF5 non trouvé : " + filename);
  }

  H5::H5File file(filename, H5F_ACC_RDONLY);
  std::string base_group = subOctreeName + "/T=" + std::to_string(temperature);

  H5::Group base_grp = file.openGroup(base_group);

  int num_attrs = base_grp.getNumAttrs();
  for (int i = 0; i < num_attrs; ++i) {
    H5::Attribute attr = base_grp.openAttribute(i);
    std::string name_attr = attr.getName();
    if (name_attr == "max_depth") {
      int val;
      attr.read(H5::PredType::NATIVE_INT, &val);
      this->max_depth = static_cast<std::uint_fast8_t>(val);
    }
    if (name_attr == "half_size") {
      double val;
      attr.read(H5::PredType::NATIVE_DOUBLE, &val);
      this->initial_size = static_cast<double>(2. * val);
    }
    if (name_attr == "center") {
      attr.read(H5::PredType::NATIVE_DOUBLE, hypercube_center.data());
    }
  }

  leaves_indx->resize(this->max_depth + 1);
  indxWithOffset->resize(this->max_depth + 1);

  leaves_indx->clear();
  begin_offset.clear();
  end_offset.clear();

  hsize_t num_objs;
  base_group = base_group + "/" + family;

  H5Gget_num_objs(file.openGroup(base_group).getId(), &num_objs);
  for (std::size_t l = 0; l < num_objs; ++l) {
    std::string leaf_group_name = base_group + "/leave_lvl_" + std::to_string(l);
    if (!H5Lexists(file.getId(), leaf_group_name.c_str(), H5P_DEFAULT)) continue;

    H5::DataSet dset_center = file.openDataSet(leaf_group_name + "/center_octree");
    H5::DataSpace dspace_center = dset_center.getSpace();
    hsize_t dims_center[2];
    dspace_center.getSimpleExtentDims(dims_center, nullptr);
    std::size_t num_entries = dims_center[0];
    std::size_t key_size = dims_center[1];

    std::vector<double> flat_keys(num_entries * key_size);
    dset_center.read(flat_keys.data(), H5::PredType::NATIVE_DOUBLE);

    std::vector<std::uint64_t> keys_morton(num_entries);
    for (std::size_t row = 0; row < num_entries; ++row) {
      std::array<double, DIM> key_vec;
      for (std::size_t i = 0; i < key_size; ++i) {
        key_vec[i] = flat_keys[row * key_size + i];
      }
      keys_morton[row] = VectorisationIndex::vect_indices_to_index(this->coord_to_index(key_vec),
                                                                   this->max_depth + 2);
    }

    leaves_indx->push_back(std::move(keys_morton));

    H5::DataSet dset_beg = file.openDataSet(leaf_group_name + "/begin_offset");
    std::vector<std::uint64_t> flat_beg(num_entries);
    dset_beg.read(flat_beg.data(), H5::PredType::NATIVE_UINT64);
    begin_offset.push_back(std::move(flat_beg));

    H5::DataSet dset_end = file.openDataSet(leaf_group_name + "/end_offset");
    std::vector<std::uint64_t> flat_end(num_entries);
    dset_end.read(flat_end.data(), H5::PredType::NATIVE_UINT64);
    end_offset.push_back(std::move(flat_end));
  }

  H5::Group main_group = file.openGroup(base_group);
  hsize_t nobj;
  H5Gget_num_objs(main_group.getId(), &nobj);

  index->clear();

  for (hsize_t i = 0; i < nobj; ++i) {
    char name[1024];
    ssize_t len = H5Gget_objname_by_idx(main_group.getId(), i, name, sizeof(name));
    if (len <= 0) continue;
    std::string key(name);
    if (key.rfind("leave_lvl_", 0) == 0) continue;
    std::string group_name = base_group + "/" + key;

    if (!H5Lexists(file.getId(), group_name.c_str(), H5P_DEFAULT)) continue;

    H5::DataSet dset_indx = file.openDataSet(base_group + "/indx");
    H5::DataSet dset_val = file.openDataSet(base_group + "/" + key);

    H5::DataSpace dspace_indx = dset_indx.getSpace();
    hsize_t dims_indx[2];
    dspace_indx.getSimpleExtentDims(dims_indx, nullptr);
    std::size_t num_entries = dims_indx[0];
    std::size_t key_size = dims_indx[1];

    std::vector<double> flat_keys(num_entries * key_size);
    dset_indx.read(flat_keys.data(), H5::PredType::NATIVE_DOUBLE);

    for (std::size_t v = 0; v < this->variables_names.size(); v++) {
      std::string mapKey = this->variables_names[v];
      values[v]->clear();
      H5::DataSet dset_val = file.openDataSet(base_group + "/" + mapKey);
      std::vector<T> values_(num_entries);
      dset_val.read(values_.data(), H5::PredType::NATIVE_DOUBLE);
      (*values[v]) = (std::move(values_));
    }

    std::vector<std::uint64_t> indices_(num_entries);
    for (std::size_t row = 0; row < num_entries; ++row) {
      std::array<double, DIM> key_vec;
      for (std::size_t j = 0; j < key_size; ++j) {
        key_vec[j] = flat_keys[row * key_size + j];
      }
      indices_[row] = VectorisationIndex::vect_indices_to_index(this->coord_to_index(key_vec),
                                                                this->max_depth + 2);
    }

    (*index) = std::move(indices_);
  }

  for (size_t l = 0; l <= this->max_depth; l++) {
    auto& vec_lvl = (*leaves_indx)[l];
    if (l == 0 && vec_lvl.size() != 0) {
      no_refinement = true;
    }
    auto& indxWithOffset_ = (*indxWithOffset)[l];
    indxWithOffset_.fill_containers(vec_lvl, (*index), begin_offset[l], end_offset[l]);
    begin_offset[l].clear();
    end_offset[l].clear();
    indxWithOffset_.sort_data();
  }
}

/**
 * @brief Construct a new Implicit Hyper Octree< T,  D I M>:: Implicit Hyper Octree object
 *
 * @tparam T
 * @tparam DIM
 * @param values_
 * @param index_
 * @param leaves_indx_
 * @param indxWithOffset_
 */
template <typename T, std::size_t DIM>
ImplicitHyperOctree<T, DIM>::ImplicitHyperOctree(
    std::vector<std::shared_ptr<std::vector<double>>> values_,
    std::shared_ptr<std::vector<std::uint64_t>> index_,
    std::shared_ptr<std::vector<std::vector<std::uint64_t>>> leaves_indx_,
    std::shared_ptr<std::vector<LeafOffset<std::uint64_t>>> indxWithOffset_,
    const std::vector<std::string>& variables_names_)
    : values(values_),
      index(index_),
      leaves_indx(leaves_indx_),
      indxWithOffset(indxWithOffset_),
      variables_names(variables_names_),
      no_refinement(false) {}

/**
 * @brief Destroy the Implicit Hyper Octree< T,  D I M>:: Implicit Hyper Octree object
 *
 * @tparam T
 * @tparam DIM
 */
template <typename T, std::size_t DIM>
ImplicitHyperOctree<T, DIM>::~ImplicitHyperOctree() {}