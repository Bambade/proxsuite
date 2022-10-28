//
// Copyright (c) 2022 INRIA
//

#include <proxsuite/proxqp/dense/solve_primal_dual_splitting_full_sides_reduced.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>


namespace proxsuite {
namespace proxqp {

namespace dense {

namespace python {

template<typename T>
void
exposeSocpObjectDense(pybind11::module_ m)
{

  ::pybind11::class_<proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>>(m, "PrimalDualSplittingSettings", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.") // constructor
    .def_readwrite("tau", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::tau)
    .def_readwrite("alpha_over_relaxed", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::alpha_over_relaxed)
    .def_readwrite("eps_update", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::eps_update)
    .def_readwrite("mu_update_factor", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::mu_update_factor)
    .def_readwrite("max_iter", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::max_iter)
    .def_readwrite("max_iter_inner_loop", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::max_iter_inner_loop)
    .def_readwrite("check_termination", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::check_termination)
    .def_readwrite("mu_update_fact_bound", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::mu_update_fact_bound)
    .def_readwrite("mu_update_fact_bound_inv", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::mu_update_fact_bound_inv)
    .def_readwrite("eps_abs", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::eps_abs)
    .def_readwrite("eps_rel", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::eps_rel)
    .def_readwrite("eps_primal_inf", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::eps_primal_inf)
    .def_readwrite("eps_dual_inf", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::eps_dual_inf)
    .def_readwrite("nb_iterative_refinement",
                   &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::nb_iterative_refinement)
    .def_readwrite("initial_guess", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::initial_guess)
    .def_readwrite("preconditioner_accuracy",
                   &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::preconditioner_accuracy)
    .def_readwrite("preconditioner_max_iter",
                   &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::preconditioner_max_iter)
    .def_readwrite("power_iteration_accuracy",
                   &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::power_iteration_accuracy)
    .def_readwrite("power_iteration_max_iter",
                   &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::power_iteration_max_iter)
    .def_readwrite("compute_timings", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::compute_timings)
    .def_readwrite("compute_preconditioner",
                   &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::compute_preconditioner)
    .def_readwrite("update_preconditioner", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::update_preconditioner)
    .def_readwrite("verbose", &proxsuite::proxqp::dense::PrimalDualSplittingSettings<T>::verbose);

  ::pybind11::class_<proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>>(m, "PrimalDualSplittingInfo", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.")
    .def_readwrite("mu_in", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::mu_in)
    .def_readwrite("max_eig", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::max_eig)
    .def_readwrite("eps_current", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::eps_current)
    .def_readwrite("rho", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::rho)
    .def_readwrite("admm_step_size", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::admm_step_size)
    .def_readwrite("iter", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::iter)
    .def_readwrite("run_time", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::run_time)
    .def_readwrite("setup_time", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::setup_time)
    .def_readwrite("solve_time", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::solve_time)
    .def_readwrite("pri_res", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::pri_res)
    .def_readwrite("dua_res", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::dua_res)
    .def_readwrite("objValue", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::objValue)
    .def_readwrite("status", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::status)
    .def_readwrite("mu_updates", &proxsuite::proxqp::dense::PrimalDualSplittingInfo<T>::mu_updates);

  ::pybind11::class_<proxsuite::proxqp::dense::PrimalDualSplittingResults<T>>(m, "PrimalDualSplittingResults", pybind11::module_local())
    .def(::pybind11::init<proxsuite::linalg::veg::i64, proxsuite::linalg::veg::i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("m", 0, "total number of constraints."),
         "Constructor from QP model dimensions.") // constructor
    .PROXSUITE_PYTHON_EIGEN_READWRITE(proxsuite::proxqp::dense::PrimalDualSplittingResults<T>, x, "The primal solution.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      proxsuite::proxqp::dense::PrimalDualSplittingResults<T>,
      y,
      "The dual solution associated to the constraints (in the following order: equality, inequality and then soc constraints).")
    .def_readwrite("info", &proxsuite::proxqp::dense::PrimalDualSplittingResults<T>::info);

  ::pybind11::class_<proxsuite::proxqp::dense::SOCP<T>>(m, "SOCP") //,pybind11::module_local()
    .def(::pybind11::init<proxsuite::linalg::veg::i64, proxsuite::linalg::veg::i64, proxsuite::linalg::veg::i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         pybind11::arg_v("n_in", 0, "number of inequality constraints."),
         "Constructor using SOCP model dimensions.") // constructor
    .def_readwrite(
      "model", &proxsuite::proxqp::dense::SOCP<T>::model, "class containing the SOCP model")
    .def_readwrite(
      "results",
      &proxsuite::proxqp::dense::SOCP<T>::results,
      "class containing the solution or certificate of infeasibility, "
      "and "
      "information statistics in an info subclass.")
    .def_readwrite("settings",
                   &proxsuite::proxqp::dense::SOCP<T>::settings,
                   "class with settings option of the solver.")
    .def(
      "init",
      &proxsuite::proxqp::dense::SOCP<T>::init,
      "function for initializing the model when passing sparse matrices in "
      "entry.",
      pybind11::arg_v("H", std::nullopt, "quadratic cost"),
      pybind11::arg_v("g", std::nullopt, "linear cost"),
      pybind11::arg_v("A", std::nullopt, "equality constraint matrix"),
      pybind11::arg_v("b", std::nullopt, "equality constraint vector"),
      pybind11::arg_v("C", std::nullopt, "inequality constraint matrix"),
      pybind11::arg_v("u", std::nullopt, "inequality constraint vector"),
      pybind11::arg_v("l", std::nullopt, "inequality constraint vector"),
      pybind11::arg_v("compute_preconditioner",
                      true,
                      "execute the preconditioner for reducing "
                      "ill-conditioning and speeding up solver execution."),
      pybind11::arg_v("rho", std::nullopt, "primal proximal parameter"),
      pybind11::arg_v(
        "mu_in", std::nullopt, "dual inequality constraint proximal parameter")
    )

    .def("solve",
         static_cast<void (proxsuite::proxqp::dense::SOCP<T>::*)()>(&proxsuite::proxqp::dense::SOCP<T>::solve),
         "function used for solving the QP problem, using default parameters.")
    .def("solve",
         static_cast<void (proxsuite::proxqp::dense::SOCP<T>::*)(
           std::optional<proxsuite::proxqp::dense::VecRef<T>> x,
           std::optional<proxsuite::proxqp::dense::VecRef<T>> y)>(&proxsuite::proxqp::dense::SOCP<T>::solve),
         "function used for solving the QP problem, when passing a warm start")
    .def("cleanup",
         &proxsuite::proxqp::dense::SOCP<T>::cleanup,
         "function used for cleaning the result "
         "class.");
}

} // namespace python
} // namespace dense


} // namespace proxqp
} // namespace proxsuite
