//
// Copyright (c) 2022 INRIA
//

#include <proxsuite/proxqp/sparse/prox_socp_sparse.hpp>
#include <proxsuite/proxqp/dense/prox_socp_dense.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace proxsuite {
namespace proxqp {
using proxsuite::linalg::veg::isize;

namespace sparse {

namespace python {

template<typename T, typename I>
void
exposeSocpObjectSparse(pybind11::module_ m)
{

  ::pybind11::class_<sparse::ProxSocpSettings<T>>(m, "SocpSettings", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.") // constructor
    .def_readwrite("mu_min_eq", &sparse::ProxSocpSettings<T>::mu_min_eq)
    .def_readwrite("mu_min_in", &sparse::ProxSocpSettings<T>::mu_min_in)
    .def_readwrite("mu_min_soc", &sparse::ProxSocpSettings<T>::mu_min_soc)
    .def_readwrite("tau", &sparse::ProxSocpSettings<T>::tau)
    .def_readwrite("mu_update_factor", &sparse::ProxSocpSettings<T>::mu_update_factor)
    .def_readwrite("max_iter", &sparse::ProxSocpSettings<T>::max_iter)
    .def_readwrite("check_termination", &sparse::ProxSocpSettings<T>::check_termination)
    .def_readwrite("mu_update_fact_bound", &sparse::ProxSocpSettings<T>::mu_update_fact_bound)
    .def_readwrite("eps_abs", &sparse::ProxSocpSettings<T>::eps_abs)
    .def_readwrite("eps_rel", &sparse::ProxSocpSettings<T>::eps_rel)
    .def_readwrite("eps_primal_inf", &sparse::ProxSocpSettings<T>::eps_primal_inf)
    .def_readwrite("eps_dual_inf", &sparse::ProxSocpSettings<T>::eps_dual_inf)
    .def_readwrite("nb_iterative_refinement",
                   &sparse::ProxSocpSettings<T>::nb_iterative_refinement)
    .def_readwrite("initial_guess", &sparse::ProxSocpSettings<T>::initial_guess)
    .def_readwrite("preconditioner_accuracy",
                   &sparse::ProxSocpSettings<T>::preconditioner_accuracy)
    .def_readwrite("preconditioner_max_iter",
                   &sparse::ProxSocpSettings<T>::preconditioner_max_iter)
    .def_readwrite("compute_timings", &sparse::ProxSocpSettings<T>::compute_timings)
    .def_readwrite("compute_preconditioner",
                   &sparse::ProxSocpSettings<T>::compute_preconditioner)
    .def_readwrite("update_preconditioner", &sparse::ProxSocpSettings<T>::update_preconditioner)
    .def_readwrite("verbose", &sparse::ProxSocpSettings<T>::verbose);

  ::pybind11::class_<sparse::ProxSocpInfo<T>>(m, "SocpInfo", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.")
    .def_readwrite("mu_eq", &sparse::ProxSocpInfo<T>::mu_eq)
    .def_readwrite("mu_in", &sparse::ProxSocpInfo<T>::mu_in)
    .def_readwrite("mu_soc", &sparse::ProxSocpInfo<T>::mu_soc)
    .def_readwrite("rho", &sparse::ProxSocpInfo<T>::rho)
    .def_readwrite("iter", &sparse::ProxSocpInfo<T>::iter)
    .def_readwrite("run_time", &sparse::ProxSocpInfo<T>::run_time)
    .def_readwrite("setup_time", &sparse::ProxSocpInfo<T>::setup_time)
    .def_readwrite("solve_time", &sparse::ProxSocpInfo<T>::solve_time)
    .def_readwrite("pri_res", &sparse::ProxSocpInfo<T>::pri_res)
    .def_readwrite("dua_res", &sparse::ProxSocpInfo<T>::dua_res)
    .def_readwrite("objValue", &sparse::ProxSocpInfo<T>::objValue)
    .def_readwrite("status", &sparse::ProxSocpInfo<T>::status)
    .def_readwrite("mu_updates", &sparse::ProxSocpInfo<T>::mu_updates);

  ::pybind11::class_<sparse::ProxSocpResults<T>>(m, "SocpResults", pybind11::module_local())
    .def(::pybind11::init<i64, i64, i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("m", 0, "total number of constraints."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         "Constructor from QP model dimensions.") // constructor
    .PROXSUITE_PYTHON_EIGEN_READWRITE(sparse::ProxSocpResults<T>, x, "The primal solution.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      sparse::ProxSocpResults<T>,
      y,
      "The dual solution associated to the constraints (in the following order: equality, inequality and then soc constraints).")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      sparse::ProxSocpResults<T>,
      z,
      "The slack variable associated to the inequality and soc constraints.")
    .def_readwrite("info", &sparse::ProxSocpResults<T>::info);



  ::pybind11::class_<sparse::SOCP<T, I>>(m, "SOCP") //,pybind11::module_local()
    .def(::pybind11::init<i64, i64, i64,i64,Eigen::Matrix<i64, Eigen::Dynamic, 1>>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("m", 0, "total number of constraints."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         pybind11::arg_v("n_in", 0, "number of inequality constraints."),
         pybind11::arg_v("dims", std::nullopt, "vector of second order cone dimensions."),
         "Constructor using SOCP model dimensions.") // constructor
    .def_readwrite(
      "model", &sparse::SOCP<T, I>::model, "class containing the SOCP model")
    .def_readwrite(
      "results",
      &sparse::SOCP<T, I>::results,
      "class containing the solution or certificate of infeasibility, "
      "and "
      "information statistics in an info subclass.")
    .def_readwrite("settings",
                   &sparse::SOCP<T, I>::settings,
                   "class with settings option of the solver.")
    .def(
      "init",
      &sparse::SOCP<T, I>::init,
      "function for initializing the model when passing sparse matrices in "
      "entry.",
      pybind11::arg_v("H", std::nullopt, "quadratic cost"),
      pybind11::arg_v("g", std::nullopt, "linear cost"),
      pybind11::arg_v("A", std::nullopt, "constraint matrix"),
      pybind11::arg_v("u", std::nullopt, "constraint vector, containing the equality constraint part, the upper inequality part and the soc constraint affine parts"),
      pybind11::arg_v("l", std::nullopt, "lower inequality constraint vector"),
      pybind11::arg_v("compute_preconditioner",
                      true,
                      "execute the preconditioner for reducing "
                      "ill-conditioning and speeding up solver execution."),
      pybind11::arg_v("rho", std::nullopt, "primal proximal parameter"),
      pybind11::arg_v(
        "mu_eq", std::nullopt, "dual equality constraint proximal parameter"),
      pybind11::arg_v(
        "mu_in", std::nullopt, "dual inequality constraint proximal parameter"),
      pybind11::arg_v(
        "mu_soc", std::nullopt, "dual soc constraint proximal parameter")
        )

    .def("solve",
         static_cast<void (sparse::SOCP<T, I>::*)()>(&sparse::SOCP<T, I>::solve),
         "function used for solving the QP problem, using default parameters.")
    .def("solve",
         static_cast<void (sparse::SOCP<T, I>::*)(
           std::optional<sparse::VecRef<T>> x,
           std::optional<sparse::VecRef<T>> y)>(&sparse::SOCP<T, I>::solve),
         "function used for solving the QP problem, when passing a warm start")
    .def("cleanup",
         &sparse::SOCP<T, I>::cleanup,
         "function used for cleaning the result "
         "class.");
}




} // namespace python
} // namespace sparse


namespace dense {

namespace python {

template<typename T>
void
exposeSocpObjectDense(pybind11::module_ m)
{

  ::pybind11::class_<ProxSocpSettings<T>>(m, "SocpSettings", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.") // constructor
    .def_readwrite("mu_min_eq", &ProxSocpSettings<T>::mu_min_eq)
    .def_readwrite("mu_min_in", &ProxSocpSettings<T>::mu_min_in)
    .def_readwrite("mu_min_soc", &ProxSocpSettings<T>::mu_min_soc)
    .def_readwrite("tau", &ProxSocpSettings<T>::tau)
    .def_readwrite("mu_update_factor", &ProxSocpSettings<T>::mu_update_factor)
    .def_readwrite("max_iter", &ProxSocpSettings<T>::max_iter)
    .def_readwrite("check_termination", &ProxSocpSettings<T>::check_termination)
    .def_readwrite("mu_update_fact_bound", &ProxSocpSettings<T>::mu_update_fact_bound)
    .def_readwrite("eps_abs", &ProxSocpSettings<T>::eps_abs)
    .def_readwrite("eps_rel", &ProxSocpSettings<T>::eps_rel)
    .def_readwrite("eps_primal_inf", &ProxSocpSettings<T>::eps_primal_inf)
    .def_readwrite("eps_dual_inf", &ProxSocpSettings<T>::eps_dual_inf)
    .def_readwrite("nb_iterative_refinement",
                   &ProxSocpSettings<T>::nb_iterative_refinement)
    .def_readwrite("initial_guess", &ProxSocpSettings<T>::initial_guess)
    .def_readwrite("preconditioner_accuracy",
                   &ProxSocpSettings<T>::preconditioner_accuracy)
    .def_readwrite("preconditioner_max_iter",
                   &ProxSocpSettings<T>::preconditioner_max_iter)
    .def_readwrite("compute_timings", &ProxSocpSettings<T>::compute_timings)
    .def_readwrite("compute_preconditioner",
                   &ProxSocpSettings<T>::compute_preconditioner)
    .def_readwrite("update_preconditioner", &ProxSocpSettings<T>::update_preconditioner)
    .def_readwrite("verbose", &ProxSocpSettings<T>::verbose);

  ::pybind11::class_<dense::ProxSocpInfo<T>>(m, "SocpInfo", pybind11::module_local())
    .def(::pybind11::init(), "Default constructor.")
    .def_readwrite("mu_eq", &dense::ProxSocpInfo<T>::mu_eq)
    .def_readwrite("mu_in", &dense::ProxSocpInfo<T>::mu_in)
    .def_readwrite("mu_soc", &dense::ProxSocpInfo<T>::mu_soc)
    .def_readwrite("rho", &dense::ProxSocpInfo<T>::rho)
    .def_readwrite("iter", &dense::ProxSocpInfo<T>::iter)
    .def_readwrite("run_time", &dense::ProxSocpInfo<T>::run_time)
    .def_readwrite("setup_time", &dense::ProxSocpInfo<T>::setup_time)
    .def_readwrite("solve_time", &dense::ProxSocpInfo<T>::solve_time)
    .def_readwrite("pri_res", &dense::ProxSocpInfo<T>::pri_res)
    .def_readwrite("dua_res", &dense::ProxSocpInfo<T>::dua_res)
    .def_readwrite("objValue", &dense::ProxSocpInfo<T>::objValue)
    .def_readwrite("status", &dense::ProxSocpInfo<T>::status)
    .def_readwrite("mu_updates", &dense::ProxSocpInfo<T>::mu_updates);

  ::pybind11::class_<ProxSocpResults<T>>(m, "SocpResults", pybind11::module_local())
    .def(::pybind11::init<i64, i64, i64>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("m", 0, "total number of constraints."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         "Constructor from QP model dimensions.") // constructor
    .PROXSUITE_PYTHON_EIGEN_READWRITE(ProxSocpResults<T>, x, "The primal solution.")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      ProxSocpResults<T>,
      y,
      "The dual solution associated to the constraints (in the following order: equality, inequality and then soc constraints).")
    .PROXSUITE_PYTHON_EIGEN_READWRITE(
      ProxSocpResults<T>,
      z,
      "The slack variable associated to the inequality and soc constraints.")
    .def_readwrite("info", &ProxSocpResults<T>::info);



  ::pybind11::class_<dense::SOCP<T>>(m, "SOCP") //,pybind11::module_local()
    .def(::pybind11::init<i64, i64, i64,i64,Eigen::Matrix<i64, Eigen::Dynamic, 1>>(),
         pybind11::arg_v("n", 0, "primal dimension."),
         pybind11::arg_v("m", 0, "total number of constraints."),
         pybind11::arg_v("n_eq", 0, "number of equality constraints."),
         pybind11::arg_v("n_in", 0, "number of inequality constraints."),
         pybind11::arg_v("dims", std::nullopt, "vector of second order cone dimensions."),
         "Constructor using SOCP model dimensions.") // constructor
    .def_readwrite(
      "model", &dense::SOCP<T>::model, "class containing the SOCP model")
    .def_readwrite(
      "results",
      &dense::SOCP<T>::results,
      "class containing the solution or certificate of infeasibility, "
      "and "
      "information statistics in an info subclass.")
    .def_readwrite("settings",
                   &dense::SOCP<T>::settings,
                   "class with settings option of the solver.")
    .def(
      "init",
      &dense::SOCP<T>::init,
      "function for initializing the model when passing sparse matrices in "
      "entry.",
      pybind11::arg_v("H", std::nullopt, "quadratic cost"),
      pybind11::arg_v("g", std::nullopt, "linear cost"),
      pybind11::arg_v("A", std::nullopt, "constraint matrix"),
      pybind11::arg_v("u", std::nullopt, "constraint vector, containing the equality constraint part, the upper inequality part and the soc constraint affine parts"),
      pybind11::arg_v("l", std::nullopt, "lower inequality constraint vector"),
      pybind11::arg_v("compute_preconditioner",
                      true,
                      "execute the preconditioner for reducing "
                      "ill-conditioning and speeding up solver execution."),
      pybind11::arg_v("rho", std::nullopt, "primal proximal parameter"),
      pybind11::arg_v(
        "mu_eq", std::nullopt, "dual equality constraint proximal parameter"),
      pybind11::arg_v(
        "mu_in", std::nullopt, "dual inequality constraint proximal parameter"),
      pybind11::arg_v(
        "mu_soc", std::nullopt, "dual soc constraint proximal parameter")
        )

    .def("solve",
         static_cast<void (dense::SOCP<T>::*)()>(&dense::SOCP<T>::solve),
         "function used for solving the QP problem, using default parameters.")
    .def("solve",
         static_cast<void (dense::SOCP<T>::*)(
           std::optional<dense::VecRef<T>> x,
           std::optional<dense::VecRef<T>> y)>(&dense::SOCP<T>::solve),
         "function used for solving the QP problem, when passing a warm start")
    .def("cleanup",
         &dense::SOCP<T>::cleanup,
         "function used for cleaning the result "
         "class.");
}

} // namespace python
} // namespace dense


} // namespace proxqp
} // namespace proxsuite
