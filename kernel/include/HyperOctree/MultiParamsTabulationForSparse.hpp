/**
 * @file MultiParamsTabulationForSparse.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief 
 * @version 0.1
 * @date 2026-05-21
 * 
 * @copyright Copyright (c) 2026
 * 
 */
#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "HyperOctree.hpp"
#include "HyperOctreeGrid.hpp"
#include "ImplicitHyperOctree.hpp"
#include "Calphad/CalphadBase.hpp"
#include "Calphad/CalphadUtils.hpp"
#include "IO/HDF54Sloth.hpp"
#include "Interpolators/MultiLinearInterpolator.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Options/Options.hpp"
#include "Parameters/Parameter.hpp"
#include "Parameters/Parameters.hpp"
#include "Utils/Utils.hpp"

#pragma once

template <typename T, std::size_t DIM>
class MultiParamsTabulationForSparse : public CalphadBase<T> {
 private:
  std::string given_phase_;

  std::map<std::string, unsigned int> idx_elem_;
  std::vector<std::string> input_composition_order_;
  std::size_t INTERP_DIM;
  std::map<std::string, std::string> filename;

  std::ofstream fichier;

  std::vector<std::tuple<std::string, std::string, double>> unsuspended_phases_;

  std::map<std::string, std::vector<double>> temperature_vec;
  std::map<std::string, std::size_t> nbreOctree;

  std::map<std::string, std::vector<HyperOctreeGrid<ImplicitHyperOctree, double, DIM>>>
      GridForChemicalPot;
  std::map<std::string, std::vector<HyperOctreeGrid<ImplicitHyperOctree, double, DIM>>>
      GridForChemicalMob;
  std::map<std::string, std::vector<HyperOctreeGrid<ImplicitHyperOctree, double, DIM>>>
      GridForChemicalEnergies;

  std::vector<std::size_t> list_of_aux_gf_index_for_tabulation;

  std::vector<std::string> list_of_dataset_tabulation_parameters;

  std::vector<std::string> list_of_element;

  std::vector<std::string> families;
  std::vector<std::vector<std::string>> sub_variables;

  inline static std::function<double(double)> scalling_func_mob;
  inline static std::function<double(double)> scalling_func_energy;
  inline static std::function<double(double)> scalling_func_potentials;

  std::unique_ptr<CalphadUtils<T>> CU_;
  void compute(const int dt, const std::set<int>& list_nodes, const std::vector<T>& tp_gf,
               const std::vector<std::tuple<std::string, std::string>>& chemical_system,
               const std::string& given_phase);

  void check_variables_consistency(
      std::vector<std::tuple<std::vector<std::string>, std::reference_wrapper<T>>>& output_system);

 public:
  constexpr explicit MultiParamsTabulationForSparse(const Parameters& params);

  constexpr MultiParamsTabulationForSparse(const Parameters& params, bool is_KKS);

  void initialize(
      const std::vector<std::tuple<std::string, std::string>>& sorted_chemical_system) override;

  void execute(const int dt, const std::set<int>& list_nodes, const std::vector<T>& tp_gf,
               const std::vector<std::tuple<std::string, std::string>>& chemical_system,
               std::optional<std::vector<std::tuple<std::string, std::string, double>>>
                   status_phase = std::nullopt) override;
  void finalize() override;

  ////////////////////////////////

  void get_parameters() override;

  ////////////////////////////////
  ~MultiParamsTabulationForSparse();
};

////////////////////////////////
////////////////////////////////

/**
 * @brief Get the parameters associated with the MultiParamsTabulationForSparse object
 * @tparam T
 */
template <typename T, std::size_t DIM>
void MultiParamsTabulationForSparse<T, DIM>::get_parameters() {
  // this->description_ = this->params_.template get_param_value_or_default<std::string>(
  //     "description",
  //     "Analytical thermodynamic description for an ideal solution using tabulated data");
  this->scalling_func_mob =
      this->params_.template get_param_value_or_default<std::function<double(double)>>(
          "scalling_func_mobilities", [](double v) { return std::exp(v); });
  this->scalling_func_energy =
      this->params_.template get_param_value_or_default<std::function<double(double)>>(
          "scalling_func_energy", [](double v) { return v; });
  this->scalling_func_potentials =
      this->params_.template get_param_value_or_default<std::function<double(double)>>(
          "scalling_func_potentials", [](double v) { return v; });
  this->list_of_element =
      this->params_.template get_param_value_or_default<std::vector<std::string>>(
          "list_of_elements", {"O", "U"});

  this->list_of_dataset_tabulation_parameters =
      this->params_.template get_param_value_or_default<std::vector<std::string>>(
          "list_of_dataset_tabulation_parameters", {"T", "xO"});
  this->list_of_aux_gf_index_for_tabulation =
      this->params_.template get_param_value_or_default<std::vector<std::size_t>>(
          "list_of_aux_gf_index_for_tabulation", {0, 2});
  this->filename =
      this->params_.template get_param_value_or_default<std::map<std::string, std::string>>(
          "data_filename", {});
  this->families = this->params_.template get_param_value_or_default<std::vector<std::string>>(
      "data_families", {"MU", "M", "ENERGIES"});
  this->sub_variables =
      this->params_.template get_param_value_or_default<std::vector<std::vector<std::string>>>(
          "data_sub_variables", {{"mu_O", "mu_U"}, {"M_O", "M_U"}, {"G", "H", "GM", "HM"}});

  this->temperature_vec =
      this->params_.template get_param_value_or_default<std::map<std::string, std::vector<double>>>(
          "temperature_map", {});

  this->nbreOctree =
      this->params_.template get_param_value_or_default<std::map<std::string, std::size_t>>(
          "data_nbreOctree_by_phase", std::map<std::string, std::size_t>{});
  this->INTERP_DIM = static_cast<std::size_t>(
      this->params_.template get_param_value<int>("dimension_of_interpolation"));
  this->given_phase_ =
      this->params_.template get_param_value_or_default<std::string>("GivenPhase", "");
  //   bool check_parameters_mob_mu =
  //       (list_of_dataset_name_mob.size() == list_of_dataset_name_mu.size()) &&
  //       (list_of_dataset_name_mob.size() == list_of_element.size());
  //   bool check_parameters_tab_params = true;
  //   //   (list_of_aux_gf_index_for_tabulation.size() ==
  //   //    list_of_dataset_tabulation_parameters.size());
  //   MFEM_VERIFY(
  //       check_parameters_tab_params,
  //       " Parameters : (1) list_of_aux_gf_index_for_tabulation , (2) "
  //       "list_of_dataset_tabulation_parameters , (3) dimension_of_interpolation are not
  //       consistent");
  //   MFEM_VERIFY(check_parameters_mob_mu,
  //               " Parameters : (1) list_of_dataset_name_mob , (2) "
  //               "list_of_dataset_name_mu , (3) list_of_element are not consistent");
}

////////////////////////////////
////////////////////////////////
/**
 * @brief Construct a new MultiParamsTabulationForSparse::MultiParamsTabulationForSparse object
 *
 * @param params
 */
template <typename T, std::size_t DIM>
constexpr MultiParamsTabulationForSparse<T, DIM>::MultiParamsTabulationForSparse(
    const Parameters& params)
    : CalphadBase<T>(params, false), fichier("composition.domain", std::ios::app) {
  this->CU_ = std::make_unique<CalphadUtils<T>>();
  this->get_parameters();
}

/**
 * @brief Construct a new MultiParamsTabulationForSparse::MultiParamsTabulationForSparse object
 *
 * @param params
 */
template <typename T, std::size_t DIM>
constexpr MultiParamsTabulationForSparse<T, DIM>::MultiParamsTabulationForSparse(
    const Parameters& params, bool is_KKS)
    : CalphadBase<T>(params, is_KKS), fichier("composition.domain", std::ios::app) {
  this->CU_ = std::make_unique<CalphadUtils<T>>();
  this->get_parameters();
}

/**
 * @brief Initialization of the thermodynamic calculation
 * @tparam T
 */
template <typename T, std::size_t DIM>
void MultiParamsTabulationForSparse<T, DIM>::initialize(
    const std::vector<std::tuple<std::string, std::string>>& sorted_chemical_system) {
  Catch_Time_Section("MultiParamsTabulationForSparse::initialize");

  // TODOCP : creer les containers et appeler l'interface hdf5

  for (const auto& pair : temperature_vec) {
    const std::string& phase_name = pair.first;
    const std::vector<double>& phase_temp_vec = pair.second;
    GridForChemicalPot[phase_name].resize(phase_temp_vec.size());
    GridForChemicalMob[phase_name].resize(phase_temp_vec.size());
    GridForChemicalEnergies[phase_name].resize(phase_temp_vec.size());
    std::string filename = this->filename[phase_name];
    for (std::size_t n = 1; n <= this->nbreOctree[phase_name]; n++) {
      std::string subOctreeName = "O" + std::to_string(n) + "/";
      for (std::size_t index_T = 0; index_T < phase_temp_vec.size(); index_T++) {
        double temperature = phase_temp_vec[index_T];

        std::vector<std::string> keys = this->sub_variables[0];
        std::vector<std::vector<std::uint64_t>> vec_ref_for_implicit;
        std::vector<std::vector<std::uint64_t>> begin_offset;
        std::vector<std::vector<std::uint64_t>> end_offset;
        std::vector<std::shared_ptr<std::vector<double>>> vec_val(keys.size());
        for (std::size_t i = 0; i < keys.size(); i++) {
          vec_val[i] = std::make_shared<std::vector<double>>();
        }

        auto vec_indx = std::make_shared<std::vector<std::uint64_t>>();
        auto morton_leaves =
            std::make_shared<std::vector<std::vector<std::uint64_t>>>(vec_ref_for_implicit);

        auto indxWithOffset = std::make_shared<std::vector<LeafOffset<std::uint64_t>>>();
        ImplicitHyperOctree<double, DIM> implicitOctree(vec_val, vec_indx, morton_leaves,
                                                        indxWithOffset, keys);
        implicitOctree.read_file_for_map_values(begin_offset, end_offset, filename, temperature,
                                                subOctreeName, families[0]);
        GridForChemicalPot[phase_name][index_T].add_hyperoctree(std::move(implicitOctree));
      }

      for (std::size_t index_T = 0; index_T < phase_temp_vec.size(); index_T++) {
        double temperature = phase_temp_vec[index_T];

        std::vector<std::string> keys = this->sub_variables[1];
        std::vector<std::vector<std::uint64_t>> vec_ref_for_implicit;
        std::vector<std::vector<std::uint64_t>> begin_offset;
        std::vector<std::vector<std::uint64_t>> end_offset;
        std::vector<std::shared_ptr<std::vector<double>>> vec_val(keys.size());
        for (std::size_t i = 0; i < keys.size(); i++) {
          vec_val[i] = std::make_shared<std::vector<double>>();
        }

        auto vec_indx = std::make_shared<std::vector<std::uint64_t>>();
        auto morton_leaves =
            std::make_shared<std::vector<std::vector<std::uint64_t>>>(vec_ref_for_implicit);

        auto indxWithOffset = std::make_shared<std::vector<LeafOffset<std::uint64_t>>>();
        ImplicitHyperOctree<double, DIM> implicitOctree(vec_val, vec_indx, morton_leaves,
                                                        indxWithOffset, keys);
        implicitOctree.read_file_for_map_values(begin_offset, end_offset, filename, temperature,
                                                subOctreeName, families[1]);
        GridForChemicalMob[phase_name][index_T].add_hyperoctree(std::move(implicitOctree));
      }

      for (std::size_t index_T = 0; index_T < phase_temp_vec.size(); index_T++) {
        double temperature = phase_temp_vec[index_T];

        std::vector<std::string> keys = this->sub_variables[2];
        std::vector<std::vector<std::uint64_t>> vec_ref_for_implicit;
        std::vector<std::vector<std::uint64_t>> begin_offset;
        std::vector<std::vector<std::uint64_t>> end_offset;
        std::vector<std::shared_ptr<std::vector<double>>> vec_val(keys.size());
        for (std::size_t i = 0; i < keys.size(); i++) {
          vec_val[i] = std::make_shared<std::vector<double>>();
        }

        auto vec_indx = std::make_shared<std::vector<std::uint64_t>>();
        auto morton_leaves =
            std::make_shared<std::vector<std::vector<std::uint64_t>>>(vec_ref_for_implicit);

        auto indxWithOffset = std::make_shared<std::vector<LeafOffset<std::uint64_t>>>();
        ImplicitHyperOctree<double, DIM> implicitOctree(vec_val, vec_indx, morton_leaves,
                                                        indxWithOffset, keys);
        implicitOctree.read_file_for_map_values(begin_offset, end_offset, filename, temperature,
                                                subOctreeName, families[2]);
        GridForChemicalEnergies[phase_name][index_T].add_hyperoctree(std::move(implicitOctree));
      }
    }
  }
  // Pour n'avoir que O,U,PU en clé
  for (auto& s : this->sub_variables[0]) {
    s.erase(0, std::min<std::size_t>(3, s.size()));
  }
  for (auto& s : this->sub_variables[1]) {
    s.erase(0, std::min<std::size_t>(2, s.size()));
  }

  // Check input composition consistency (cf CalphadInformedNeuralNetwork.hpp)
  if (this->params_.template has_parameter("InputCompositionOrder")) {
    this->input_composition_order_ =
        this->params_.template get_param_value<vString>("InputCompositionOrder");
  }

  for (int i = 0; i < sorted_chemical_system.size(); i++) {
    const std::string& elem = std::get<0>(sorted_chemical_system[i]);
    this->idx_elem_.emplace(elem, i);
  }
  if (!this->input_composition_order_.empty()) {
    std::vector<std::string> v_elem;
    for (const auto& tup : sorted_chemical_system) {
      const std::string& elem = std::get<0>(tup);
      v_elem.emplace_back(elem);
      this->idx_elem_[elem] = std::distance(this->input_composition_order_.begin(),
                                            std::find(this->input_composition_order_.begin(),
                                                      this->input_composition_order_.end(), elem));
    }
    // Sort before comparison
    std::ranges::sort(v_elem);
    auto sort_input_composition_order = this->input_composition_order_;
    std::ranges::sort(sort_input_composition_order);

    MFEM_VERIFY(std::ranges::equal(v_elem, sort_input_composition_order),
                "Error: InputCompositionOrder is not consistent with composition deduced from "
                "auxiliary variables. Please check your data");
  }
}

/**
 * @brief Main method to calculate equilibrium states
 * @tparam T
 * @param dt
 * @param aux_gf
 * @param
 * chemical_system
 * @param
 * output_system
 */
template <typename T, std::size_t DIM>
void MultiParamsTabulationForSparse<T, DIM>::execute(
    const int dt, const std::set<int>& list_nodes, const std::vector<T>& tp_gf,
    const std::vector<std::tuple<std::string, std::string>>& chemical_system,
    std::optional<std::vector<std::tuple<std::string, std::string, double>>> status_phase) {
  Catch_Time_Section("MultiParamsTabulationForSparse::execute");

  std::string phase;
  if (status_phase.has_value()) {
    const auto& phase_vector = *status_phase;
    MFEM_VERIFY(
        phase_vector.size() == 1,
        "MultiParamsTabulationForSparse: status phase must contain exactly one element. \n");
    phase = std::get<0>(phase_vector[0]);
  } else {
    if (this->given_phase_.size() > 0) {
      phase = this->given_phase_;
    } else {
      throw std::runtime_error(
          "MultiParamsTabulationForSparse::execute: GivenPhase parameter must be defined.");
    }
  }

  this->compute(dt, list_nodes, tp_gf, chemical_system, phase);
}

template <typename T, std::size_t DIM>
void MultiParamsTabulationForSparse<T, DIM>::compute(
    const int dt, const std::set<int>& list_nodes, const std::vector<T>& tp_gf,
    const std::vector<std::tuple<std::string, std::string>>& chemical_system,
    const std::string& given_phase) {
  //         // TODOCP : refaire pour tenir compte des nouvelles fonctionnalités :)
  // Catch_Time_Section("MultiParamsTabulationForSparse::compute");
  const std::vector<std::string> energy_names = this->sub_variables[2];
  std::vector<double> energy_l;
  std::vector<double> energy_r;
  std::vector<double> mu_l;
  std::vector<double> mu_r;
  std::vector<double> mob_l;
  std::vector<double> mob_r;
  // Pré-filtrer les noeuds une seule fois
  std::vector<int> sorted_n_t_p = this->CU_->sort_nodes(tp_gf[0], tp_gf[1], "No", "No");
  sorted_n_t_p.erase(
      std::remove_if(sorted_n_t_p.begin(), sorted_n_t_p.end(),
                     [&list_nodes](int node) { return list_nodes.find(node) == list_nodes.end(); }),
      sorted_n_t_p.end());

  //   // Pré-allouer les structures pour éviter les reallocations
  std::vector<double> tp_gf_at_node(tp_gf.size());
  std::array<double, DIM> point_to_interpolate;

  const auto& temps = temperature_vec[given_phase];
  // std::cout << "given_phase   : " << given_phase << std::endl;
  const size_t max_idx = temps.size() - 1;

  for (const auto& id : sorted_n_t_p) {
    // fichier << given_phase << ",";
    // Remplir tp_gf_at_node une seule fois par noeud
    for (size_t i = 0; i < tp_gf.size(); ++i) {
      tp_gf_at_node[i] = tp_gf[i][id];
      // fichier << tp_gf_at_node[i] << ",";
    }
    // fichier << "\n";

    this->heat_capacity_[id] = 60.;
    // Pré-remplir point_to_interpolate
    for (size_t id_elem = 0; id_elem < list_of_aux_gf_index_for_tabulation.size(); ++id_elem) {
      point_to_interpolate[id_elem] = tp_gf_at_node[list_of_aux_gf_index_for_tabulation[id_elem]];
      // std::cout << point_to_interpolate[id_elem] << std::endl;
    }

    // Trouver les indices de température une seule fois
    auto lower_it = std::lower_bound(temps.begin(), temps.end(), tp_gf_at_node[0]);
    size_t index_l = (lower_it == temps.begin()) ? 0 : (std::distance(temps.begin(), lower_it) - 1);
    index_l = std::clamp(index_l, size_t(0), max_idx);
    size_t index_r = std::clamp(index_l + 1, size_t(0), max_idx);
    double u = (tp_gf_at_node[0] - temps[index_l]) / (temps[index_r] - temps[index_l]);
    // std::cout << index_l << "  " << temps[index_l] << "   " << index_r << std::endl;
    // Calculer les énergies

    energy_l =
        GridForChemicalEnergies[given_phase][index_l].search_point_in_tree(point_to_interpolate);
    energy_r =
        GridForChemicalEnergies[given_phase][index_r].search_point_in_tree(point_to_interpolate);
    for (std::size_t v = 0; v < this->sub_variables[2].size(); v++) {
      this->energies_of_phases_[{id, given_phase, this->sub_variables[2][v]}] =
          std::lerp(energy_l[v], energy_r[v], u);
    }

    // Calculer les potentiels chimiques, fractions molaires et mobilités

    // Potentiels chimiques
    mu_l = GridForChemicalPot[given_phase][index_l].search_point_in_tree(point_to_interpolate);
    mu_r = GridForChemicalPot[given_phase][index_r].search_point_in_tree(point_to_interpolate);
    // this->chemical_potentials_[{id, elem}] = std::lerp(mu_l, mu_r, u);
    for (std::size_t v = 0; v < this->sub_variables[0].size(); v++) {
      this->chemical_potentials_[{id, this->sub_variables[0][v]}] = std::lerp(mu_l[v], mu_r[v], u);
    }
    // Fractions molaires
    for (size_t id_elem = 0; id_elem < chemical_system.size(); ++id_elem) {
      const auto& elem = std::get<0>(chemical_system[id_elem]);
      this->elem_mole_fraction_by_phase_[{id, given_phase, elem}] =
          tp_gf_at_node[this->idx_elem_[elem] + 2];
    }

    // Mobilités
    if (given_phase != "LIQUID") {
      // mob_l = GridForChemicalMob[given_phase][index_l].search_point_in_tree(point_to_interpolate);
      // mob_r = GridForChemicalMob[given_phase][index_r].search_point_in_tree(point_to_interpolate);
      // for (std::size_t v = 0; v < this->sub_variables[1].size(); v++) {
      //   this->mobilities_[{id, given_phase, this->sub_variables[1][v]}] =
      //       std::exp(std::lerp(mob_l[v], mob_r[v], u));
      // }
      this->mobilities_[{id, given_phase, "O"}] = 2.e-9;
      this->mobilities_[{id, given_phase, "U"}] = 5.e-11;
      this->mobilities_[{id, given_phase, "PU"}] = 4.e-16;

    } else {
      for (std::size_t v = 0; v < this->sub_variables[1].size(); v++) {
        this->mobilities_[{id, given_phase, this->sub_variables[1][v]}] = 0.;
      }
    }

    // Mise à jour de la fraction molaire pour l'élément retiré
    auto& last_fraction =
        this->elem_mole_fraction_by_phase_[{id, given_phase, this->element_removed_from_ic_}];
    last_fraction = 1.;
    for (size_t id_elem = 0; id_elem < chemical_system.size(); ++id_elem) {
      const auto& elem = std::get<0>(chemical_system[id_elem]);
      if (elem != this->element_removed_from_ic_) {
        last_fraction -= this->elem_mole_fraction_by_phase_[{id, given_phase, elem}];

        // this->diffusion_chemical_potentials_[{id, elem}] =
        //     this->chemical_potentials_[{id, elem}] -
        //     this->chemical_potentials_[{id, this->element_removed_from_ic_}];
      }
    }
    for (std::size_t v = 0; v < this->sub_variables[0].size(); v++) {
      if (this->sub_variables[0][v] != this->element_removed_from_ic_) {
        this->diffusion_chemical_potentials_[{id, this->sub_variables[0][v]}] =
            this->chemical_potentials_[{id, this->sub_variables[0][v]}] -
            this->chemical_potentials_[{id, this->element_removed_from_ic_}];
      }
    }
  }

  // fichier.flush();
}

/**
 * @brief Check the consistency of outputs required for the current Calphad problem
 * @tparam T
 * @param output_system
 */
template <typename T, std::size_t DIM>
void MultiParamsTabulationForSparse<T, DIM>::check_variables_consistency(
    std::vector<std::tuple<std::vector<std::string>, std::reference_wrapper<T>>>& output_system) {
  for (auto& [output_infos, output_value] : output_system) {
    const std::string& output_type = output_infos.back();

    switch (calphad_outputs::from(output_type)) {
      case calphad_outputs::h:
      case calphad_outputs::hm: {
        MFEM_VERIFY(false, "MultiParamsTabulationForSparse is only built for mu, x and g.\n");

        SlothInfo::debug("Output not available for this Calphad problem: ", output_type);
        break;
      }
    }
  }
}

/**
 * @brief Finalization actions (free memory)
 *
 * @tparam T
 */
template <typename T, std::size_t DIM>
void MultiParamsTabulationForSparse<T, DIM>::finalize() {
  fichier.close();
}

/**
 * @brief Destroy the Binary Melting< T>::Binary Melting object
 * @tparam T
 */
template <typename T, std::size_t DIM>
MultiParamsTabulationForSparse<T, DIM>::~MultiParamsTabulationForSparse() {}
