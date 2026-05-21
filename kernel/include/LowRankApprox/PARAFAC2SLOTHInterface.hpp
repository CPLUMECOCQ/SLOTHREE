/**
 * @file PARAFAC2SLOTHInterface.hpp
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

#include "Calphad/CalphadBase.hpp"
#include "Calphad/CalphadUtils.hpp"
#include "IO/HDF54Sloth.hpp"
#include "Interpolators/MultiLinearInterpolator.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Options/Options.hpp"
#include "Parameters/Parameter.hpp"
#include "Parameters/Parameters.hpp"
#include "PARAFACReconstruction.hpp"
#include "Utils/Utils.hpp"

#pragma once

template <typename T, std::size_t DIM>
class PARAFAC2SLOTHInterface : public CalphadBase<T> {
 private:
  std::string given_phase_;

  std::map<std::string, unsigned int> idx_elem_;
  std::vector<std::string> input_composition_order_;
  std::map<std::string, std::string> filename;

  std::ofstream fichier;

  std::vector<std::tuple<std::string, std::string, double>> unsuspended_phases_;

  std::map<std::string, std::vector<double>> temperature_vec;
  std::map<std::string, std::size_t> nbreOctree;

  std::map<std::string, std::map<std::string, PARAFACReconstruction<DIM>>> mapForChemicalPot;
  std::map<std::string, std::map<std::string, PARAFACReconstruction<DIM>>> mapForChemicalMob;
  std::map<std::string, std::map<std::string, PARAFACReconstruction<DIM>>> mapForChemicalEnergies;
  std::map<std::string, std::map<std::string, std::map<std::string, PARAFACReconstruction<DIM>>>>
      mapForChemicalPotDerivative;

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
  constexpr explicit PARAFAC2SLOTHInterface(const Parameters& params);

  constexpr PARAFAC2SLOTHInterface(const Parameters& params, bool is_KKS);

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
  ~PARAFAC2SLOTHInterface();
};

////////////////////////////////
////////////////////////////////

/**
 * @brief Get the parameters associated with the PARAFAC2SLOTHInterface object
 * @tparam T
 */
template <typename T, std::size_t DIM>
void PARAFAC2SLOTHInterface<T, DIM>::get_parameters() {
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
  //   this->mu_names = this->params_.template get_param_value_or_default<std::vector<std::string>>(
  //       "dataset_name_for_chemical_pot", {""});
  //   this->nrj_names = this->params_.template
  //   get_param_value_or_default<std::vector<std::string>>(
  //       "dataset_name_for_chemical_pot", {""});
  this->sub_variables =
      this->params_.template get_param_value_or_default<std::vector<std::vector<std::string>>>(
          "data_sub_variables", {{"mu_O", "mu_U"}, {"M_O", "M_U"}, {"G", "H", "GM", "HM"}});

  this->temperature_vec =
      this->params_.template get_param_value_or_default<std::map<std::string, std::vector<double>>>(
          "temperature_map", {});

  this->filename =
      this->params_.template get_param_value_or_default<std::map<std::string, std::string>>(
          "data_filename", {});

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
 * @brief Construct a new PARAFAC2SLOTHInterface::PARAFAC2SLOTHInterface object
 *
 * @param params
 */
template <typename T, std::size_t DIM>
constexpr PARAFAC2SLOTHInterface<T, DIM>::PARAFAC2SLOTHInterface(const Parameters& params)
    : CalphadBase<T>(params, false), fichier("composition.domain", std::ios::app) {
  this->CU_ = std::make_unique<CalphadUtils<T>>();
  this->get_parameters();
}

/**
 * @brief Construct a new TT_Interpolation::TT_Interpolation object
 *
 * @param params
 */
template <typename T, std::size_t DIM>
constexpr PARAFAC2SLOTHInterface<T, DIM>::PARAFAC2SLOTHInterface(const Parameters& params, bool is_KKS)
    : CalphadBase<T>(params, is_KKS), fichier("composition.domain", std::ios::app) {
  this->CU_ = std::make_unique<CalphadUtils<T>>();
  this->get_parameters();
}

/**
 * @brief Initialization of the thermodynamic calculation
 * @tparam T
 */
template <typename T, std::size_t DIM>
void PARAFAC2SLOTHInterface<T, DIM>::initialize(
    const std::vector<std::tuple<std::string, std::string>>& sorted_chemical_system) {
  Catch_Time_Section("PARAFAC2SLOTHInterface::initialize");

  std::string files = this->filename["C1_MO2"];  //"sol_tensor_train_rank_20.h5";
  std::string filel = this->filename["LIQUID"];  // "liq_tensor_train_rank_20.h5";

  std::vector<std::string> coord_grid = {"T", "O", "U"};

  mapForChemicalPot["C1_MO2"].emplace("O", PARAFACReconstruction<DIM>());
  mapForChemicalPot["C1_MO2"].emplace("U", PARAFACReconstruction<DIM>());
  mapForChemicalPot["C1_MO2"].emplace("PU", PARAFACReconstruction<DIM>());

  mapForChemicalPot["C1_MO2"]["O"].import_cores_from_hdf5(files, "mu_O", coord_grid);
  mapForChemicalPot["C1_MO2"]["U"].import_cores_from_hdf5(files, "mu_U", coord_grid);
  mapForChemicalPot["C1_MO2"]["PU"].import_cores_from_hdf5(files, "mu_PU", coord_grid);

  mapForChemicalPot["LIQUID"].emplace("O", PARAFACReconstruction<DIM>());
  mapForChemicalPot["LIQUID"].emplace("U", PARAFACReconstruction<DIM>());
  mapForChemicalPot["LIQUID"].emplace("PU", PARAFACReconstruction<DIM>());

  mapForChemicalPot["LIQUID"]["O"].import_cores_from_hdf5(filel, "mu_O", coord_grid);
  mapForChemicalPot["LIQUID"]["U"].import_cores_from_hdf5(filel, "mu_U", coord_grid);
  mapForChemicalPot["LIQUID"]["PU"].import_cores_from_hdf5(filel, "mu_PU", coord_grid);

  std::vector<std::string> chem_phases = {"C1_MO2", "LIQUID"};
  std::vector<std::string> elements = {"O", "U", "PU"};
  std::vector<std::string> files_n = {files, filel};

  for (size_t i = 0; i < chem_phases.size(); ++i) {
    const auto& phase = chem_phases[i];
    auto& file = files_n[i];
    for (const auto& elem : elements) {
      mapForChemicalPot[phase].emplace(elem, PARAFACReconstruction<DIM>());

      mapForChemicalPot[phase][elem].import_cores_from_hdf5(file, "mu_" + elem, coord_grid);
    }
  }
  // mapForChemicalMob["C1_MO2"].emplace("O", PARAFACReconstruction<DIM>());
  // mapForChemicalMob["C1_MO2"].emplace("U", PARAFACReconstruction<DIM>());
  // mapForChemicalMob["C1_MO2"].emplace("PU", PARAFACReconstruction<DIM>());

  // mapForChemicalMob["C1_MO2"]["O"].import_cores_from_hdf5(files, "M_O", coord_grid);
  // mapForChemicalMob["C1_MO2"]["U"].import_cores_from_hdf5(files, "M_U", coord_grid);
  // mapForChemicalMob["C1_MO2"]["PU"].import_cores_from_hdf5(files, "M_PU", coord_grid);

  mapForChemicalEnergies["C1_MO2"].emplace("G", PARAFACReconstruction<DIM>());
  mapForChemicalEnergies["C1_MO2"]["G"].import_cores_from_hdf5(files, "G", coord_grid);

  mapForChemicalEnergies["LIQUID"].emplace("G", PARAFACReconstruction<DIM>());
  mapForChemicalEnergies["LIQUID"]["G"].import_cores_from_hdf5(filel, "G", coord_grid);

  mapForChemicalEnergies["C1_MO2"].emplace("GM", PARAFACReconstruction<DIM>());
  mapForChemicalEnergies["C1_MO2"]["GM"].import_cores_from_hdf5(files, "GM", coord_grid);

  mapForChemicalEnergies["LIQUID"].emplace("GM", PARAFACReconstruction<DIM>());
  mapForChemicalEnergies["LIQUID"]["GM"].import_cores_from_hdf5(filel, "GM", coord_grid);

  // Pour n'avoir que O,U,PU en clé
  //   for (auto& s : this->sub_variables[0]) {
  //     s.erase(0, std::min<std::size_t>(3, s.size()));
  //   }
  //   for (auto& s : this->sub_variables[1]) {
  //     s.erase(0, std::min<std::size_t>(2, s.size()));
  //   }

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
void PARAFAC2SLOTHInterface<T, DIM>::execute(
    const int dt, const std::set<int>& list_nodes, const std::vector<T>& tp_gf,
    const std::vector<std::tuple<std::string, std::string>>& chemical_system,
    std::optional<std::vector<std::tuple<std::string, std::string, double>>> status_phase) {
  Catch_Time_Section("PARAFAC2SLOTHInterface::execute");

  std::string phase;
  if (status_phase.has_value()) {
    const auto& phase_vector = *status_phase;
    MFEM_VERIFY(phase_vector.size() == 1,
                "PARAFAC2SLOTHInterface: status phase must contain exactly one element. \n");
    phase = std::get<0>(phase_vector[0]);
  } else {
    if (this->given_phase_.size() > 0) {
      phase = this->given_phase_;
    } else {
      throw std::runtime_error("PARAFAC2SLOTHInterface::execute: GivenPhase parameter must be defined.");
    }
  }

  this->compute(dt, list_nodes, tp_gf, chemical_system, phase);
}

template <typename T, std::size_t DIM>
void PARAFAC2SLOTHInterface<T, DIM>::compute(
    const int dt, const std::set<int>& list_nodes, const std::vector<T>& tp_gf,
    const std::vector<std::tuple<std::string, std::string>>& chemical_system,
    const std::string& given_phase) {
  //         // TODOCP : refaire pour tenir compte des nouvelles fonctionnalités :)
  // Catch_Time_Section("PARAFAC2SLOTHInterface::compute");
  const std::vector<std::string> energy_names = this->sub_variables[2];

  // Pré-filtrer les noeuds une seule fois
  std::vector<int> sorted_n_t_p = this->CU_->sort_nodes(tp_gf[0], tp_gf[1], "No", "No");
  sorted_n_t_p.erase(
      std::remove_if(sorted_n_t_p.begin(), sorted_n_t_p.end(),
                     [&list_nodes](int node) { return list_nodes.find(node) == list_nodes.end(); }),
      sorted_n_t_p.end());

  //   // Pré-allouer les structures pour éviter les reallocations
  std::vector<double> tp_gf_at_node(tp_gf.size());
  std::array<double, DIM> point_to_interpolate;

  // std::cout << "given_phase   : " << given_phase << std::endl;

  for (const auto& id : sorted_n_t_p) {
    // fichier << given_phase << ",";
    // Remplir tp_gf_at_node une seule fois par noeud
    for (size_t i = 0; i < tp_gf.size(); ++i) {
      tp_gf_at_node[i] = tp_gf[i][id];
      // fichier << tp_gf_at_node[i] << ",";
    }
    // fichier << "\n";

    this->heat_capacity_[id] = 60.;
    point_to_interpolate[0] = tp_gf_at_node[0];

    // Pré-remplir point_to_interpolate
    for (size_t id_elem = 0; id_elem < list_of_aux_gf_index_for_tabulation.size(); ++id_elem) {
      point_to_interpolate[id_elem + 1] =
          tp_gf_at_node[list_of_aux_gf_index_for_tabulation[id_elem]];
    }

    // Trouver les indices de température une seule fois
    // std::cout << index_l << "  " << temps[index_l] << "   " << index_r << std::endl;
    // Calculer les énergies
    this->energies_of_phases_[{id, given_phase, "G"}] =
        mapForChemicalEnergies[given_phase]["G"].get_value_on_grid(point_to_interpolate);
    this->energies_of_phases_[{id, given_phase, "GM"}] =
        mapForChemicalEnergies[given_phase]["GM"].get_value_on_grid(point_to_interpolate);

    this->chemical_potentials_[{id, "O"}] =
        mapForChemicalPot[given_phase]["O"].get_value_on_grid(point_to_interpolate);
    this->chemical_potentials_[{id, "U"}] =
        mapForChemicalPot[given_phase]["U"].get_value_on_grid(point_to_interpolate);
    this->chemical_potentials_[{id, "PU"}] =
        mapForChemicalPot[given_phase]["PU"].get_value_on_grid(point_to_interpolate);

    // Potentiels chimiques

    // Fractions molaires
    for (size_t id_elem = 0; id_elem < chemical_system.size(); ++id_elem) {
      const auto& elem = std::get<0>(chemical_system[id_elem]);
      this->elem_mole_fraction_by_phase_[{id, given_phase, elem}] =
          tp_gf_at_node[this->idx_elem_[elem] + 2];
    }

    // Mobilités
    // std::cout << this->sub_variables[1][0] << "   " << this->sub_variables[1][1] << std::endl;
    if (given_phase != "LIQUID") {
      // for (std::size_t v = 0; v < this->sub_variables[1].size(); v++) {
      //   this->mobilities_[{id, given_phase, this->sub_variables[1][v]}] =
      //       std::exp(mapForChemicalMob[given_phase][this->sub_variables[1][v]].get_value_on_grid(
      //           point_to_interpolate));
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
        this->diffusion_chemical_potentials_[{id, elem}] =
            this->chemical_potentials_[{id, elem}] -
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
void PARAFAC2SLOTHInterface<T, DIM>::check_variables_consistency(
    std::vector<std::tuple<std::vector<std::string>, std::reference_wrapper<T>>>& output_system) {
  for (auto& [output_infos, output_value] : output_system) {
    const std::string& output_type = output_infos.back();

    switch (calphad_outputs::from(output_type)) {
      case calphad_outputs::h:
      case calphad_outputs::hm: {
        MFEM_VERIFY(false, "PARAFAC2SLOTHInterface is only built for mu, x and g.\n");

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
void PARAFAC2SLOTHInterface<T, DIM>::finalize() {
  fichier.close();
}

/**
 * @brief Destroy the Binary Melting< T>::Binary Melting object
 * @tparam T
 */
template <typename T, std::size_t DIM>
PARAFAC2SLOTHInterface<T, DIM>::~PARAFAC2SLOTHInterface() {}
