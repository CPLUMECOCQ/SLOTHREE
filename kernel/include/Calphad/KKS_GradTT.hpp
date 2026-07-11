/**
 * @file KKS_GradTT.hpp
 * @author cp273896 (clement.plumecocq@cea.fr)
 * @brief
 * @version 0.1
 * @date 2026-05-27
 *
 * @copyright Copyright (c) 2026
 *
 */
#pragma once
#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "Calphad/CalphadUtils.hpp"
#include "Coefficients/Coefficient.hpp"
#include "Coefficients/CommonCoefficients.hpp"
#include "Glossary/Glossary.hpp"
#include "HyperOctree/HyperOctree.hpp"
#include "HyperOctree/HyperOctreeGrid.hpp"
#include "HyperOctree/ImplicitHyperOctree.hpp"
#include "LowRankApprox//TTCores.hpp"
#include "MAToolsProfiling/MATimersAPI.hxx"
#include "Options/Options.hpp"
#include "Parameters/Parameter.hpp"
#include "Parameters/Parameters.hpp"

/**
 * @brief Linearized KKS problem for two-phase multi-component system
 *
 * @tparam T
 */
template <typename T>
class KKS_GradTT {
 private:
  // Tolerance to detect the presence of a phase at the previous time-step
  const double xmin_tol_ = 1.e-12;
  // Parameters for GMRES solver used to solve system resulting from the linearization
  const double kks_abs_tol_solver_ = 1.e-16;
  const double kks_rel_tol_solver_ = 1.e-16;
  const int kks_max_iter_solver_ = 100;
  const int kks_print_level_solver_ = 0;

  // Flag to detect if specialized values must be saved
  bool KKS_enable_save_specialized_{false};

  // Nucleation strategy for nucleation (LiquidFraction by default)
  std::string KKS_nucleation_strategy_;
  // Melting temperature used if the nucleation strategy is set to GivenMeltingTemperature
  double given_melting_temperature_{std::numeric_limits<double>::max()};

  std::set<int> check_nucleation(CalphadBase<T>& CALPHAD, const std::set<int>& indices_ph_1,
                                 const T& tp_gf_ph_1);

  std::map<std::tuple<std::string, std::string, std::string>, TTCores<3>> dmu_tab_;
  std::map<std::tuple<std::string, std::string, std::string, int>, double> dmu_at_note_;

 protected:
  // Common methods for CALPHAD studies
  std::shared_ptr<CalphadUtils<T>> CU_;

  // Symbol of the chemical element removed from the system when initializing equilibrium
  // calculations (performed with molar fractions).
  std::string element_removed_from_ic_;

  // Name of the phase expected to form during phase transition
  std::string KKS_secondary_phase_;

  // The mobility of the AC equation
  double KKS_mobility_for_seed_;

  // The value of the spherical seed of secondary phase for starting nucleation
  double KKS_seed_;

  // The radius of the spherical seed of secondary phase for starting nucleation
  double KKS_seed_radius_;

  // The increment of temperature used to calculated derivatives by finite difference scheme
  double KKS_temperature_increment_;

  // The increment of composition used to calculated derivatives by finite difference scheme.
  // The same increment is used for all components.
  double KKS_composition_increment_;

  // The value of of the threshold to identify the interface
  double KKS_threshold_;

  // Only used with nucleation strategy defined by LiquidFraction (and GEM)
  // Used to detect the range of temperature where one or two phase are considered.
  // Usefull with GEM to avoid convergence difficulties
  double KKS_temperature_threshold_;

  // Numerical scheme for temperature. Explicit and Implicit (default) are available.
  // Implicit scheme simplifies the linearization but requires to solve heat transfer equation
  // first.
  bool KKS_temperature_scheme_;

  // Flag to identify is the nucleation started
  bool KKS_nucleation_started_{false};

  // Flag to freeze the nucleation. By default, when nucleation starts, it is freezed in order to
  // avoid to check again the presence other nucleii.
  bool KKS_freeze_nucleation_{true};

  // Interpolation function. It must be consistent with the choice made in AC equation.
  std::shared_ptr<Coefficient> interpolation_func_;

  // Method used to build the blocks of the linear system resulting from linearization of KKS
  // problem
  std::unique_ptr<mfem::SparseMatrix> get_A4linearKKS(
      const std::vector<std::tuple<std::string, std::string>>& chemicalsystem,
      const std::string& phase, const int node);

  mfem::Vector get_h4linearKKS(
      const std::vector<std::tuple<std::string, std::string>>& chemicalsystem,
      const std::string& phase, const int node);

  mfem::Vector get_m4linearKKS(
      const std::vector<std::tuple<std::string, std::string>>& chemicalsystem,
      const std::string& phase, const int node);

  // Specific containers used to fill blocks of the linear system resulting from linearization of
  // KKS problem
  std::map<std::tuple<int, std::string, std::string>, double> chemical_potentials_by_phase_;
  std::map<std::tuple<int, std::string, std::string>, double> chemical_potentials_left_T_;
  std::map<std::tuple<int, int, std::string, std::string>, double> chemical_potentials_left_x_;
  std::map<std::tuple<int, std::string, std::string>, double> chemical_potentials_right_T_;
  std::map<std::tuple<int, int, std::string, std::string>, double> chemical_potentials_right_x_;
  std::map<std::string, std::string> filename;

 public:
  KKS_GradTT();
  void initialize();
  void get_parameters(const CalphadBase<T>& CALPHAD);
  void execute_linearization(
      CalphadBase<T>& CALPHAD, const int dt, const double time_step, const std::vector<T>& tp_gf,
      const std::vector<T>& tp_gf_old, const std::tuple<std::string, T, T>& phasefields_gf,
      const std::vector<std::tuple<std::string, std::string>>& chemicalsystem,
      const std::vector<std::tuple<std::string, std::string, T, T>>& x_gf,
      const std::vector<std::tuple<std::string, T>>& coord_gf);

  void clear_containers();

  virtual ~KKS_GradTT();
};

#include "KKS_GradTT.tpp"
