/**
 * @file main.cpp
 * @author ci230846 (clement.introini@cea.fr)
 * @brief 2D Inter-diffusion test for a ternary system in a two-phase system
 * @version 0.1
 * @date 2025-03-27
 *
 * @copyright Copyright (c) 2025
 *
 */

//---------------------------------------
// Headers
//---------------------------------------
#include <boost/math/special_functions/bessel.hpp>
#include <string>
#include <vector>

#include "Sloth/sloth.hpp"
#include "Sloth/tests.hpp"
#include "mfem.hpp"  // NOLINT [no include the directory when naming mfem include file]

struct TestParameters {
  // CALPHAD
  double control_kks_seed = 0.5;
  double control_kks_radius = 1.e-4;
  double control_kks_threshold = 5.0e-3;
  double control_temperature_threshold = 2500.0;
  double control_KKS_given_melting_temperature = 3000.0;
  bool control_KKS_enable_save_specialized = false;

  // PDEs
  bool control_latent_heat = false;

  // Time
  double control_final_time = 30.;
  double control_time_step = 5.e-3;

  std::string liquidh5file = "data_liq.h5";
  std::string solidh5file = "data_sol.h5";
  std::string savefold = "Saves";
};

void common_parameters(mfem::OptionsParser& args, TestParameters& p) {
  args.AddOption(&p.control_kks_seed, "-s", "--kks_seed", "Seed of liquid to initiate melting.");
  args.AddOption(&p.control_kks_radius, "-r", "--kks_radius", "Radius of the initial seed.");
  args.AddOption(&p.control_kks_threshold, "-e", "--kks_threshold",
                 "Threshold for KKS calculations.");
  args.AddOption(&p.control_temperature_threshold, "-te", "--kks_temp_threshold",
                 "Minimal temperature for searching liquid during KKS calculations.");
  args.AddOption(&p.control_KKS_given_melting_temperature, "-tm", "--kks_melting_temp",
                 "Given melting temperature for KKS calculations.");
  args.AddOption(&p.control_KKS_enable_save_specialized, "-p", "--save-specialized", "-np",
                 "--no-save-specialized", "Enables the save of specialized value for KKS.");
  args.AddOption(&p.control_latent_heat, "-l", "--latent_heat", "-nl", "--no-latent-heat",
                 "Enables the use of LatentHeat integrator in heat transfer equation.");
  args.AddOption(&p.control_final_time, "-t", "--final_time", "Final time of the simulation.");
  args.AddOption(&p.control_time_step, "-dt", "--time_step",
                 "Constant time-step of the simulation.");
  args.AddOption(&p.liquidh5file, "-h5liq", "--liquid_h5_file", "file for liquid phase");
  args.AddOption(&p.solidh5file, "-h5sol", "--solid_h5_file", "file for solid phase");
  args.AddOption(&p.savefold, "-savefold", "--savefold", "folder to save");

  args.Parse();

  if (!args.Good()) {
    if (mfem::Mpi::WorldRank() == 0) {
      args.PrintUsage(mfem::out);
      std::exit(EXIT_FAILURE);
    }
  }
  if (mfem::Mpi::WorldRank() == 0) args.PrintOptions(mfem::out);
}

///---------------
/// Main program
///---------------
int main(int argc, char* argv[]) {
  setVerbosity(Verbosity::Verbose);
  //---------------------------------------
  // Initialize MPI and HYPRE
  //---------------------------------------
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();
  //---------------------------------------
  // Profiling
  Profiling::getInstance().enable();

  // ################ //
  // ################ //
  //   Read options   //
  // ################ //
  // ################ //
  TestParameters p;
  mfem::OptionsParser args(argc, argv);
  common_parameters(args, p);

  //---------------------------------------
  // Common aliases
  //---------------------------------------
  const int DIM = 2;
  using FECollection = Test<DIM>::FECollection;
  using VARS = Test<DIM>::VARS;
  using VAR = Test<DIM>::VAR;
  using PST = Test<DIM>::PST;
  using SPA = Test<DIM>::SPA;
  using BCS = Test<DIM>::BCS;

  using OPE = TransientOperator<FECollection, DIM>;
  using PB = Problem<OPE, VARS, PST>;
  //---------------------------------------
  // Meshing & Boundary Conditions
  //---------------------------------------
  const int refinement_level = 0;
  const int fe_order = 1;

  const double pellet_radius = 6.07e-3;

  SPA spatial("GMSH", fe_order, refinement_level, "camembertMelting.msh", false);
  //   SPA spatial("GMSH", fe_order, refinement_level, "camembertMelting3D.msh", false);
  //   SPA spatial("GMSH", fe_order, refinement_level, "pellet.msh", false);

  // ##############################
  //     Boundary conditions     //
  // ##############################

  auto interdiffu_bcs = BCS(&spatial, Boundary("lower", 0, "Neumann"),
                            Boundary("external", 2, "Neumann"), Boundary("upper", 1, "Neumann"));
  auto thermal_bcs =
      BCS(&spatial, Boundary("lower", 0, "Neumann"), Boundary("external", 2, "Dirichlet", 700.),
          Boundary("upper", 1, "Neumann"));
  auto calphad_bcs = BCS(&spatial, Boundary("lower", 0, "Neumann"),
                         Boundary("external", 2, "Neumann"), Boundary("upper", 1, "Neumann"));
  auto pressure_bcs =
      BCS(&spatial, Boundary("lower", 0, "Dirichlet", 5.e6),
          Boundary("external", 2, "Dirichlet", 5.e6), Boundary("upper", 1, "Dirichlet", 5.e6));

  //   // 3D
  //   auto interdiffu_bcs =
  //       BCS(&spatial, Boundary("InterPelletPlane", 0, "Neumann"),
  //           Boundary("MidPelletPlane", 1, "Neumann"), Boundary("FrontSurface", 2, "Neumann"),
  //           Boundary("BehindSurface", 3, "Neumann"), Boundary("ExternalSurface", 4, "Neumann"),
  //           Boundary("BehindSurface", 5, "Neumann"), Boundary("BehindSurface", 6, "Neumann"));
  //   auto thermal_bcs =
  //       BCS(&spatial, Boundary("ExternalSurface", 0, "Dirichlet", 700.),
  //           Boundary("InterPelletPlane", 1, "Neumann"), Boundary("MidPelletPlane", 2, "Neumann"),
  //           Boundary("FrontSurface", 3, "Neumann"), Boundary("BehindSurface", 4, "Neumann"),
  //           Boundary("BehindSurface", 5, "Neumann"), Boundary("BehindSurface", 6, "Neumann"));
  //   auto calphad_bcs =
  //       BCS(&spatial, Boundary("InterPelletPlane", 0, "Neumann"),
  //           Boundary("MidPelletPlane", 1, "Neumann"), Boundary("FrontSurface", 2, "Neumann"),
  //           Boundary("BehindSurface", 3, "Neumann"), Boundary("ExternalSurface", 4, "Neumann"),
  //           Boundary("BehindSurface", 5, "Neumann"), Boundary("BehindSurface", 6, "Neumann"));
  //   auto pressure_bcs = BCS(&spatial, Boundary("InterPelletPlane", 0, "Dirichlet", 1.e6),
  //                           Boundary("MidPelletPlane", 1, "Dirichlet", 1.e6),
  //                           Boundary("FrontSurface", 2, "Dirichlet", 1.e6),
  //                           Boundary("BehindSurface", 3, "Dirichlet", 1.e6),
  //                           Boundary("ExternalSurface", 4, "Dirichlet", 1.e6),
  //                           Boundary("ExternalSurface", 5, "Dirichlet", 1.e6),
  //                           Boundary("ExternalSurface", 6, "Dirichlet", 1.e6));

  //---------------------------------------
  // Multiphysics coupling scheme
  //---------------------------------------
  const int level_of_storage = 2;

  std::vector<SPA*> spatials{&spatial};
  //==========================================
  //======      HEAT TRANSFER           ======
  //==========================================

  auto temp = VAR(&spatial, thermal_bcs, "T", Glossary::Temperature, level_of_storage, 700.);
  temp.set_additional_information("K", "T");
  auto heat_vars = VARS(temp);

  const double rho(32.e3);  // mol/m3
  const double cp(60.);     // J/mol/K
  const double cond(2.7);   //  W/m/K

  auto src_func = std::function<double(const mfem::Vector&, double)>(
      [pellet_radius](const mfem::Vector& vcoord, double time) {
        const double pl = 9.5e4;

        const double radius = std::sqrt(vcoord[0] * vcoord[0] + vcoord[1] * vcoord[1]);
        auto chi = 90.;  // inverse neutron diffusion length (0.9cm−1 ->90m-1).
        auto chia = chi * pellet_radius;
        auto I1_chia = boost::math::cyl_bessel_i(1, chia);
        auto chir = chi * radius;  //  (pellet_radius - radius);
        auto I0_chir = boost::math::cyl_bessel_i(0, chir);
        const auto bess = 2. * chia * I0_chir / (2. * I1_chia);
        const auto func = pl * bess / (M_PI * 2. * pellet_radius * pellet_radius);

        return func;
      });

  std::vector<AnalyticalFunctions<DIM>> source_term;
  source_term.emplace_back(AnalyticalFunctions<DIM>(src_func));
  std::vector<std::string> th_rhs_integrators = {"Fourier"};
  if (p.control_latent_heat) th_rhs_integrators.push_back("LatentHeat");
  OPE th_operator(spatials, th_rhs_integrators, TimeScheme::EulerImplicit, "HeatTimeDerivative",
                  source_term);

  th_operator.overload_nl_solver(
      NLSolverType::NEWTON,
      Parameters(Parameter("description", "Newton solver "), Parameter("print_level", -1),
                 Parameter("rel_tol", 1.e-6), Parameter("abs_tol", 1.e-6)));
  th_operator.overload_solver(HypreSolverType::HYPRE_GMRES);
  th_operator.overload_preconditioner(HyprePreconditionerType::HYPRE_ILU);

  // Interface thickness
  const auto& epsilon(5.e-4);
  // Interfacial energy
  const auto& sigma(6.e-2);
  // Two-phase mobility
  const auto& mob(1.e-4);
  const auto& lambda = 3. * sigma * epsilon / 2.;
  const auto& omega = 12. * sigma / epsilon;
  //==========================================
  //======      CALPHAD from TDB        ======
  //==========================================
  //--- Variables
  auto xcoord = std::function<double(const mfem::Vector&, double)>(
      [](const mfem::Vector& vcoord, double time) { return vcoord[0]; });
  auto ycoord = std::function<double(const mfem::Vector&, double)>(
      [](const mfem::Vector& vcoord, double time) { return vcoord[1]; });
  auto XC = VAR(&spatial, calphad_bcs, "XCOORD", Glossary::Coordinate, level_of_storage,
                AnalyticalFunctions<DIM>(xcoord));
  XC.set_additional_information("XCOORD");
  auto YC = VAR(&spatial, calphad_bcs, "YCOORD", Glossary::Coordinate, level_of_storage,
                AnalyticalFunctions<DIM>(ycoord));
  YC.set_additional_information("YCOORD");
  auto coord = VARS(XC, YC);
  // Pressure
  auto pres = VAR(&spatial, pressure_bcs, "pressure", Glossary::Pressure, level_of_storage, 50.e5);
  pres.set_additional_information("Pa", "P");
  auto p_vars = VARS(pres);

  // Initial condition for composition
  const double Nmol = 3.005;
  const double initial_compo_o = 2.005 / Nmol;
  const double initial_compo_u = 0.8 / Nmol;
  double initial_compo_pu = 1. - initial_compo_o - initial_compo_u;

  auto xo =
      VAR(&spatial, interdiffu_bcs, "O", Glossary::MoleFraction, level_of_storage, initial_compo_o);
  xo.set_additional_information("O", "x");
  auto xo_vars = VARS(xo);

  auto xu =
      VAR(&spatial, interdiffu_bcs, "U", Glossary::MoleFraction, level_of_storage, initial_compo_u);
  xu.set_additional_information("U", "x");
  auto xu_vars = VARS(xu);

  initial_compo_pu = Nmol;
  GlossaryQuantity pu_type = Glossary::MoleNumber;
  std::string pu_var = "N";

  auto xpu = VAR(&spatial, interdiffu_bcs, "PU", pu_type, level_of_storage, initial_compo_pu);
  xpu.set_additional_information("PU", pu_var);
  auto xpu_vars = VARS(xpu);

  // Chemical potential
  auto muo = VAR(&spatial, calphad_bcs, "muO", Glossary::ChemicalPotential, level_of_storage, 0.);
  muo.set_additional_information("O", "mu");
  auto mu_var = VARS(muo);

  auto muu = VAR(&spatial, calphad_bcs, "muU", Glossary::ChemicalPotential, level_of_storage, 0.);
  muu.set_additional_information("U", "mu");
  auto muu_var = VARS(muu);

  auto mupu = VAR(&spatial, calphad_bcs, "muPU", Glossary::ChemicalPotential, level_of_storage, 0.);
  mupu.set_additional_information("PU", "mu");
  auto mupu_var = VARS(mupu);

  // Mobilities
  auto mobO =
      VAR(&spatial, calphad_bcs, "Mo", Glossary::InterDiffusionMobility, level_of_storage, 0.);
  mobO.set_additional_information("C1_MO2", "O", "mob");

  auto mobU =
      VAR(&spatial, calphad_bcs, "Mu", Glossary::InterDiffusionMobility, level_of_storage, 0.);
  mobU.set_additional_information("C1_MO2", "U", "mob");

  auto mobPU =
      VAR(&spatial, calphad_bcs, "Mpu", Glossary::InterDiffusionMobility, level_of_storage, 0.);
  mobPU.set_additional_information("C1_MO2", "PU", "mob");

  // MOB LIQUID

  auto lmobO =
      VAR(&spatial, calphad_bcs, "Mo", Glossary::InterDiffusionMobility, level_of_storage, 1.e-8);
  lmobO.set_additional_information("LIQUID", "O", "mob");

  auto lmobU =
      VAR(&spatial, calphad_bcs, "Mu", Glossary::InterDiffusionMobility, level_of_storage, 1.e-9);
  lmobU.set_additional_information("LIQUID", "U", "mob");

  auto lmobPU =
      VAR(&spatial, calphad_bcs, "Mpu", Glossary::InterDiffusionMobility, level_of_storage, 1.e-15);
  lmobPU.set_additional_information("LIQUID", "PU", "mob");

  auto mob_liquid = VARS(lmobO, lmobU, lmobPU);

  // Driving forces
  auto dgm_s = VAR(&spatial, calphad_bcs, "DGM_s", Glossary::DrivingForce, level_of_storage, 0.);
  dgm_s.set_additional_information("C1_MO2", "dgm");
  auto dgm_l = VAR(&spatial, calphad_bcs, "DGM_l", Glossary::DrivingForce, level_of_storage, 0.);
  dgm_l.set_additional_information("LIQUID", "dgm");

  auto nuc_l = VAR(&spatial, calphad_bcs, "NUC_l", Glossary::Nucleus, level_of_storage, 0.);
  nuc_l.set_additional_information("LIQUID", "nucleus");

  // Diffusion chemical potential
  auto dmu_opu =
      VAR(&spatial, calphad_bcs, "dmu_opu", Glossary::ChemicalPotential, level_of_storage, 0.);
  dmu_opu.set_additional_information("O", "dmu");
  auto dmu_upu =
      VAR(&spatial, calphad_bcs, "dmu_upu", Glossary::ChemicalPotential, level_of_storage, 0.);
  dmu_upu.set_additional_information("U", "dmu");

  // Mole fraction of phases
  auto xph_l = VAR(&spatial, calphad_bcs, "xph_l", Glossary::MoleFraction, level_of_storage, 0.);
  xph_l.set_additional_information("LIQUID", "xph");

  // Element molar fraction by phase

  auto xo_s = VAR(&spatial, interdiffu_bcs, "xsO", Glossary::MoleFraction, level_of_storage,
                  initial_compo_o);
  xo_s.set_additional_information("O", "C1_MO2", "xp");
  auto xu_s = VAR(&spatial, interdiffu_bcs, "xsU", Glossary::MoleFraction, level_of_storage,
                  initial_compo_u);
  xu_s.set_additional_information("U", "C1_MO2", "xp");
  auto xpu_s = VAR(&spatial, interdiffu_bcs, "xsPU", Glossary::MoleFraction, level_of_storage,
                   initial_compo_pu);
  xpu_s.set_additional_information("PU", "C1_MO2", "xp");

  auto xo_l = VAR(&spatial, interdiffu_bcs, "xlO", Glossary::MoleFraction, level_of_storage,
                  initial_compo_o);
  xo_l.set_additional_information("O", "LIQUID", "xp");
  auto xu_l = VAR(&spatial, interdiffu_bcs, "xlU", Glossary::MoleFraction, level_of_storage,
                  initial_compo_u);
  xu_l.set_additional_information("U", "LIQUID", "xp");
  auto xpu_l = VAR(&spatial, interdiffu_bcs, "xlPU", Glossary::MoleFraction, level_of_storage,
                   initial_compo_pu);
  xpu_l.set_additional_information("PU", "LIQUID", "xp");

  // Gibbs energy
  auto gl = VAR(&spatial, calphad_bcs, "g_l", Glossary::GibbsEnergy, level_of_storage, 0.);
  gl.set_additional_information("LIQUID", "g");
  auto gs = VAR(&spatial, calphad_bcs, "g_s", Glossary::GibbsEnergy, level_of_storage, 0.);
  gs.set_additional_information("C1_MO2", "g");

  auto calphad_outputs = VARS(muo, muu, mupu, mobO, mobU, mobPU, dgm_s, dgm_l, dmu_opu, dmu_upu,
                              xph_l, xo_s, xu_s, xpu_s, xo_l, xu_l, xpu_l, nuc_l, gs, gl);

  auto phi = VAR(&spatial, calphad_bcs, "phi", Glossary::PhaseField, level_of_storage, 1.);
  phi.set_additional_information("C1_MO2", "phi");
  auto var_phi = VARS(phi);
  // TDB file
  auto tdbfile = Parameter("tdb_file", "tafid.tdb");
  auto description_calphad =
      Parameter("description", "Calphad description for a U-O-Pu ternary system");
  auto element_removed_from_ic = Parameter("element_removed_from_ic", "PU");
  vTuple2StringDouble map_unsuspended_phases = {{"C1_MO2", "entered", -1}};
  auto unsuspended_phases = Parameter("unsuspended_phases", map_unsuspended_phases);

  // PARAMETERS SPARSE

  int interpol_dim = 3;
  auto input_interpol_dim = Parameter("dimension_of_interpolation", interpol_dim);
  std::vector<std::string> composition_order{"O", "U", "PU"};
  auto input_composition_order = Parameter("InputCompositionOrder", composition_order);
  std::vector<std::string> vec;
  vec = {"O", "U", "PU"};
  auto list_of_elements = Parameter("list_of_elements", vec);

  vec = {"T", "xO", "xU"};
  auto list_of_dataset_tabulation_parameters =
      Parameter("list_of_dataset_tabulation_parameters", vec);
  std::vector<std::size_t> indexgf = {2, 4};
  auto list_of_aux_gf_index_for_tabulation =
      Parameter("list_of_aux_gf_index_for_tabulation", indexgf);
  std::map<std::string, std::string> hffilename = {{"C1_MO2",  p.solidh5file},
                                                   {"LIQUID",  p.liquidh5file}};
  std::map<std::string, std::size_t> nbreOctree = {{"C1_MO2", 12}, {"LIQUID", 12}};
  auto paramh5file = Parameter("data_filename", hffilename);
  auto paramnbreOctree = Parameter("data_nbreOctree_by_phase", nbreOctree);

  std::vector<std::string> families = {"MU", "M", "ENERGIES"};
  std::vector<std::vector<std::string>> sub_variables = {
      {"mu_O", "mu_U", "mu_PU"}, {"M_O", "M_U", "M_PU"}, {"G", "H", "GM", "HM"}};
  auto paramfamilies = Parameter("data_families", families);
  auto paramsubvar = Parameter("data_sub_variables", sub_variables);
  std::map<std::string, std::vector<double>> temperature_vec;
  for (double T = 700.; T <= 3500.; T += 10.) temperature_vec["C1_MO2"].push_back(T);
  for (double T = 2500.; T <= 4000.; T += 10.) temperature_vec["LIQUID"].push_back(T);
  auto paramtemperature = Parameter("temperature_map", temperature_vec);

  auto calphad_parameters = Parameters(
      description_calphad, paramh5file, list_of_aux_gf_index_for_tabulation,
      list_of_dataset_tabulation_parameters, list_of_elements, input_interpol_dim,
      element_removed_from_ic, paramfamilies, paramsubvar, paramtemperature, paramnbreOctree);
  /// END PARAMETERS SPARSE

  auto KKS_secondary_phase = Parameter("KKS_secondary_phase", "LIQUID");
  auto KKS_temperature_increment = Parameter("KKS_temperature_increment", 1.);
  auto KKS_composition_increment = Parameter("KKS_composition_increment", 1.e-7);
  auto KKS_seed = Parameter("KKS_seed", p.control_kks_seed);
  auto KKS_seed_radius = Parameter("KKS_seed_radius", p.control_kks_radius);
  auto KKS_threshold = Parameter("KKS_threshold", p.control_kks_threshold);
  auto KKS_temperature_threshold =
      Parameter("KKS_temperature_threshold", p.control_temperature_threshold);
  auto KKS_freeze_nucleation = Parameter("KKS_freeze_nucleation", true);
  auto KKS_nucleation_started = Parameter("KKS_nucleation_started", false);
  auto KKS_enable_specialized =
      Parameter("KKS_enable_save_specialized", p.control_KKS_enable_save_specialized);
  auto KKS_nucleation_strategy = Parameter("KKS_nucleation_strategy", "GivenMeltingTemperature");
  auto KKS_given_melting_temperature =
      Parameter("KKS_given_melting_temperature", p.control_KKS_given_melting_temperature);
  auto KKS_mobility = Parameter("KKS_mobility", mob);

  auto enable_KKS = Parameter("enable_KKS", true);
  auto KKS_parameters =
      Parameters(KKS_enable_specialized, KKS_secondary_phase, KKS_temperature_increment,
                 KKS_composition_increment, KKS_seed, KKS_seed_radius, KKS_threshold,
                 KKS_temperature_threshold, KKS_freeze_nucleation, KKS_nucleation_started,
                 KKS_mobility, enable_KKS, KKS_nucleation_strategy, KKS_given_melting_temperature);

  auto octree_calphad_parameters = KKS_parameters + calphad_parameters;

  auto M11 =
      VAR(&spatial, calphad_bcs, "M11", Glossary::InterDiffusionMobility, level_of_storage, 0.);
  M11.set_additional_information("O", "inter_mob");
  auto M12 =
      VAR(&spatial, calphad_bcs, "M12", Glossary::InterDiffusionMobility, level_of_storage, 0.);
  M12.set_additional_information("U", "inter_mob");

  auto MO = VARS(M11, M12);

  auto M21 =
      VAR(&spatial, calphad_bcs, "M21", Glossary::InterDiffusionMobility, level_of_storage, 0.);
  M21.set_additional_information("U", "inter_mob");
  auto M22 =
      VAR(&spatial, calphad_bcs, "M22", Glossary::InterDiffusionMobility, level_of_storage, 0.);
  M22.set_additional_information("O", "inter_mob");

  auto MU = VARS(M21, M22);
  //==========================================
  //======      Nucleation              ======
  //==========================================
  auto nuc_parameters =
      Parameters(Parameter("primary_phase", "C1_MO2"), Parameter("secondary_phase", "LIQUID"),
                 Parameter("melting_factor", 1.));

  //==========================================
  //======      Melting                 ======
  //==========================================
  auto ac_params = nuc_parameters;

  OPE ac_oper(spatials, {"AllenCahn", "MeltingCalphad"}, ac_params, TimeScheme::EulerImplicit,
              "TimeDerivative");
  ac_oper.overload_nl_solver(
      NLSolverType::NEWTON,
      Parameters(Parameter("description", "Newton solver "), Parameter("print_level", -1),
                 Parameter("rel_tol", 1.e-12), Parameter("abs_tol", 1.e-16)));

  //==========================================
  //======      Inter-diffusion         ======
  //==========================================
  //--- Variables
  const double& stabCoeff(1.e-4);

  auto td_parameters = Parameters(Parameter("ScaleCoefficientsByTemperature", true),
                                  Parameter("EnableDiffusionChemicalPotentials", true));

  //--- Operator definition
  // Operator for InterDiffusion equation on O
  OPE interdiffu_oper_o(spatials, {"MassFlux"}, td_parameters, TimeScheme::EulerImplicit,
                        "TimeDerivative");
  interdiffu_oper_o.overload_nl_solver(
      NLSolverType::NEWTON,
      Parameters(Parameter("description", "Newton solver "), Parameter("print_level", -1),
                 Parameter("rel_tol", 1.e-12), Parameter("abs_tol", 1.e-16)));
  // Operator for InterDiffusion equation on U
  OPE interdiffu_oper_u(spatials, {"MassFlux"}, td_parameters, TimeScheme::EulerImplicit,
                        "TimeDerivative");
  interdiffu_oper_u.overload_nl_solver(
      NLSolverType::NEWTON,
      Parameters(Parameter("description", "Newton solver "), Parameter("print_level", -1),
                 Parameter("rel_tol", 1.e-12), Parameter("abs_tol", 1.e-16)));

  //==========================================
  //==========================================
  //--- Post-Processing
  const std::string& main_folder_path = p.savefold;
  std::string calculation_path = "MobilitiesO";
  bool enable_save_specialized_at_iter = true;
  const auto& frequency = 200;
  std::vector<int> iterations_list = {
      0,    200,  400,  600,  800,  1000, 1200, 1400, 1600, 1740, 1741, 1742, 1743,
      1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756,
      1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769,
      1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782,
      1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795,
      1796, 1797, 1798, 1799, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400,
      3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000};
  bool saveGF = true;
  auto pst_parameters_mob =
      Parameters(Parameter("main_folder_path", main_folder_path),
                 Parameter("calculation_path", calculation_path),
                 Parameter("iterations_list", iterations_list),
                 Parameter("enable_save_specialized_at_iter", enable_save_specialized_at_iter),
                 Parameter("saveGF", saveGF));
  auto mob_pst_o = PST(&spatial, pst_parameters_mob);

  calculation_path = "HeatTransfer";
  auto pst_parameters_heat =
      Parameters(Parameter("main_folder_path", main_folder_path),
                 Parameter("calculation_path", calculation_path),
                 Parameter("iterations_list", iterations_list),
                 Parameter("enable_save_specialized_at_iter", enable_save_specialized_at_iter),
                 Parameter("enable_compute_energies", false), Parameter("saveGF", saveGF));
  auto heat_pst = PST(&spatial, pst_parameters_heat);

  calculation_path = "Melting";
  auto pst_parameters_ac =
      Parameters(Parameter("main_folder_path", main_folder_path),
                 Parameter("calculation_path", calculation_path),
                 Parameter("iterations_list", iterations_list),
                 Parameter("enable_save_specialized_at_iter", enable_save_specialized_at_iter),
                 Parameter("enable_compute_energies", false), Parameter("saveGF", saveGF));
  auto ac_pst = PST(&spatial, pst_parameters_ac);

  calculation_path = "MobilitiesU";
  auto pst_parameters_mob_u =
      Parameters(Parameter("main_folder_path", main_folder_path),
                 Parameter("calculation_path", calculation_path),
                 Parameter("iterations_list", iterations_list),
                 Parameter("enable_save_specialized_at_iter", enable_save_specialized_at_iter),
                 Parameter("saveGF", saveGF));
  auto mob_pst_u = PST(&spatial, pst_parameters_mob_u);

  calculation_path = "InterDiffusion_o";
  auto pst_parameters =
      Parameters(Parameter("main_folder_path", main_folder_path),
                 Parameter("calculation_path", calculation_path),
                 Parameter("iterations_list", iterations_list),
                 Parameter("enable_save_specialized_at_iter", enable_save_specialized_at_iter),
                 Parameter("enable_compute_energies", false), Parameter("saveGF", saveGF));
  auto interdiffu_pst = PST(&spatial, pst_parameters);
  calculation_path = "Calphad";
  auto cc_pst_parameters =
      Parameters(Parameter("main_folder_path", main_folder_path),
                 Parameter("calculation_path", calculation_path),
                 Parameter("iterations_list", iterations_list),
                 Parameter("enable_save_specialized_at_iter", enable_save_specialized_at_iter),
                 Parameter("saveGF", saveGF));
  auto cc_pst = PST(&spatial, cc_pst_parameters);

  calculation_path = "InterDiffusion_u";
  auto diffu_pst_parameters =
      Parameters(Parameter("main_folder_path", main_folder_path),
                 Parameter("calculation_path", calculation_path),
                 Parameter("iterations_list", iterations_list),
                 Parameter("enable_save_specialized_at_iter", enable_save_specialized_at_iter),
                 Parameter("enable_compute_energies", false), Parameter("saveGF", saveGF));
  auto interdiffu_pst_u = PST(&spatial, diffu_pst_parameters);
  //   auto pst_parameters_mob =
  //       Parameters(Parameter("main_folder_path", main_folder_path),
  //                  Parameter("calculation_path", calculation_path), Parameter("frequency",
  //                  frequency), Parameter("enable_save_specialized_at_iter",
  //                  enable_save_specialized_at_iter), Parameter("saveGF", saveGF));
  //   auto mob_pst_o = PST(&spatial, pst_parameters_mob);

  //   calculation_path = "HeatTransfer";
  //   auto pst_parameters_heat =
  //       Parameters(Parameter("main_folder_path", main_folder_path),
  //                  Parameter("calculation_path", calculation_path), Parameter("frequency",
  //                  frequency), Parameter("enable_save_specialized_at_iter",
  //                  enable_save_specialized_at_iter), Parameter("enable_compute_energies", false),
  //                  Parameter("saveGF", saveGF));
  //   auto heat_pst = PST(&spatial, pst_parameters_heat);

  //   calculation_path = "Melting";
  //   auto pst_parameters_ac =
  //       Parameters(Parameter("main_folder_path", main_folder_path),
  //                  Parameter("calculation_path", calculation_path), Parameter("frequency",
  //                  frequency), Parameter("enable_save_specialized_at_iter",
  //                  enable_save_specialized_at_iter), Parameter("enable_compute_energies", false),
  //                  Parameter("saveGF", saveGF));
  //   auto ac_pst = PST(&spatial, pst_parameters_ac);

  //   calculation_path = "MobilitiesU";
  //   auto pst_parameters_mob_u =
  //       Parameters(Parameter("main_folder_path", main_folder_path),
  //                  Parameter("calculation_path", calculation_path), Parameter("frequency",
  //                  frequency), Parameter("enable_save_specialized_at_iter",
  //                  enable_save_specialized_at_iter), Parameter("saveGF", saveGF));
  //   auto mob_pst_u = PST(&spatial, pst_parameters_mob_u);

  //   calculation_path = "InterDiffusion_o";
  //   auto pst_parameters =
  //       Parameters(Parameter("main_folder_path", main_folder_path),
  //                  Parameter("calculation_path", calculation_path), Parameter("frequency",
  //                  frequency), Parameter("enable_save_specialized_at_iter",
  //                  enable_save_specialized_at_iter), Parameter("enable_compute_energies", false),
  //                  Parameter("saveGF", saveGF));
  //   auto interdiffu_pst = PST(&spatial, pst_parameters);
  //   calculation_path = "Calphad";
  //   auto cc_pst_parameters =
  //       Parameters(Parameter("main_folder_path", main_folder_path),
  //                  Parameter("calculation_path", calculation_path), Parameter("frequency",
  //                  frequency), Parameter("enable_save_specialized_at_iter",
  //                  enable_save_specialized_at_iter), Parameter("saveGF", saveGF));
  //   auto cc_pst = PST(&spatial, cc_pst_parameters);

  //   calculation_path = "InterDiffusion_u";
  //   auto diffu_pst_parameters =
  //       Parameters(Parameter("main_folder_path", main_folder_path),
  //                  Parameter("calculation_path", calculation_path), Parameter("frequency",
  //                  frequency), Parameter("enable_save_specialized_at_iter",
  //                  enable_save_specialized_at_iter), Parameter("enable_compute_energies", false),
  //                  Parameter("saveGF", saveGF));
  //   auto interdiffu_pst_u = PST(&spatial, diffu_pst_parameters);

  //-----------------------
  // Problems
  //-----------------------
  //==========================================
  //======      HEAT TRANSFER           ======
  //==========================================
  Coefficient grad_energy(Glossary::GradEnergy, Scheme::Implicit, GradientEnergy(lambda));
  Coefficient double_well(Glossary::FreeEnergy, Scheme::Implicit, W(omega));
  Coefficient capillary(Glossary::Capillary, lambda);
  Coefficient mobility(Glossary::Mobility, mob);
  Coefficient density(Glossary::Concentration, rho);
  Coefficient heat_capacity(Glossary::Cp, cp);
  Coefficient conductivity(Glossary::Conductivity, cond);
  Coefficients coef_heat(density, heat_capacity, conductivity);

  if (p.control_latent_heat) coef_heat.add(mobility);
  PB th_problem("Heat tranfer", th_operator, heat_vars, {coef_heat}, heat_pst, var_phi);

  //==========================================
  //======      CALPHAD                 ======
  //==========================================

  Calphad_Problem<TT_Interpolation<mfem::Vector, 3>, VARS, PST> ia_problem(
      octree_calphad_parameters, calphad_outputs, cc_pst, heat_vars, p_vars, xo_vars, xu_vars,
      xpu_vars, var_phi, coord);

  //==========================================
  //======      Melting                 ======
  //==========================================
  Coefficient interpolation(Glossary::InterpolationFunction, Scheme::Implicit, H());

  Coefficients coef_ac(double_well, capillary, mobility, interpolation, grad_energy);

  PB ac_problem("AllenCahn", ac_oper, var_phi, {coef_ac}, ac_pst, calphad_outputs);
  //======================
  // Oxygen
  //======================
  Coefficient Dstab(Glossary::Diffusivity, stabCoeff);
  Coefficients coef_inter(Dstab);
  auto ppo_parameters =
      Parameters(Parameter("Description", "Oxygen Mobilities"), Parameter("first_component", "O"),
                 Parameter("last_component", "Pu"), Parameter("primary_phase", "C1_MO2"),
                 Parameter("secondary_phase", "LIQUID"));

  Property_problem<InterDiffusionCoefficient, VARS, PST> oxygen_interdiffusion_mobilities(
      "Oxygen inter-diffusion mobilities", ppo_parameters, MO, mob_pst_o, xo_vars, xu_vars,
      heat_vars, calphad_outputs, var_phi, mob_liquid);

  PB interdiffu_problem_o("Interdiffusion O", interdiffu_oper_o, xo_vars, {coef_inter},
                          interdiffu_pst, calphad_outputs, MO, heat_vars);

  //======================
  // Uranium
  //======================
  auto ppu_parameters =
      Parameters(Parameter("Description", "Oxygen Mobilities"), Parameter("first_component", "U"),
                 Parameter("last_component", "PU"), Parameter("primary_phase", "C1_MO2"),
                 Parameter("secondary_phase", "LIQUID"));

  Property_problem<InterDiffusionCoefficient, VARS, PST> uranium_interdiffusion_mobilities(
      "Uranium inter-diffusion mobilities", ppu_parameters, MU, mob_pst_u, xo_vars, xu_vars,
      heat_vars, calphad_outputs, var_phi, mob_liquid);

  PB interdiffu_problem_u("Interdiffusion U", interdiffu_oper_u, xu_vars, {coef_inter},
                          interdiffu_pst_u, calphad_outputs, MU, heat_vars);

  //-----------------------
  // Coupling
  //-----------------------
  auto th_coupling = Coupling("Thermal coupling", th_problem);
  auto ia_coupling = Coupling("IA/Calphad coupling", ia_problem);
  auto ac_coupling = Coupling("Melting coupling", ac_problem);
  auto diffusion_coupling =
      Coupling("Diffusion coupling", oxygen_interdiffusion_mobilities,
               uranium_interdiffusion_mobilities, interdiffu_problem_o, interdiffu_problem_u);

  //---------------------------------------
  // Time discretization
  //---------------------------------------
  const double t_initial = 0.0;
  auto time_parameters = Parameters(Parameter("initial_time", t_initial),
                                    Parameter("final_time", p.control_final_time),
                                    Parameter("time_step", p.control_time_step));

  auto time = TimeDiscretization(time_parameters, th_coupling, ia_coupling, ac_coupling,
                                 diffusion_coupling);

  time.solve();

  //---------------------------------------
  // Profiling stop
  //---------------------------------------
  Profiling::getInstance().print();

  //---------------------------------------
  // Finalize MPI
  //---------------------------------------
  mfem::Mpi::Finalize();
}