//
// Copyright (c) 2022 INRIA
//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "algorithms.hpp"
#include <proxsuite/proxqp/dense/utils.hpp>
#include <proxsuite/helpers/version.hpp>

namespace proxsuite {
//namespace proxqp {
namespace python {

template<typename T>
void
exposeCommon(pybind11::module_ m)
{
  proxsuite::proxqp::python::exposeResults<T>(m);
  proxsuite::proxqp::python::exposeSettings<T>(m);
}

template<typename T, typename I>
void
exposeSparseAlgorithms(pybind11::module_ m)
{
  proxsuite::proxqp::sparse::python::exposeSparseModel<T, I>(m);
  proxsuite::proxqp::sparse::python::exposeQpObjectSparse<T, I>(m);
  proxsuite::proxqp::sparse::python::solveSparseQp<T, I>(m);
}

template<typename T>
void
exposeDenseAlgorithms(pybind11::module_ m)
{
  proxsuite::proxqp::dense::python::exposeDenseModel<T>(m);
  proxsuite::proxqp::dense::python::exposeQpObjectDense<T>(m);
  proxsuite::proxqp::dense::python::solveDenseQp<T>(m);
}

template<typename T>
void
exposeDenseLinalg(pybind11::module_ m)
{
  proxsuite::linalg::dense::python::DenseIterativeSolve<T>(m);
}


template<typename T,typename I>
void
exposeSparseLinalg(pybind11::module_ m)
{
  proxsuite::linalg::sparse::python::SparseFactorization<T,I>(m);
  proxsuite::linalg::sparse::python::SparseEigenFactorization<T>(m);
  
}


template<typename T,typename I>
void
exposeSparseSocp(pybind11::module_ m)
{
  proxsuite::proxqp::sparse::python::exposeSocpObjectSparse<T,I>(m);
}

template<typename T>
void
exposeDenseSocp(pybind11::module_ m)
{
  proxsuite::proxqp::dense::python::exposeSocpObjectDense<T>(m);
}


PYBIND11_MODULE(PYTHON_MODULE_NAME, m)
{
  m.doc() = R"pbdoc(
        The proxSuite library
    ------------------------

    .. currentmodule:: proxsuite
    .. autosummary::
        :toctree: _generate

        proxsuite
    )pbdoc";

  //pybind11::module_ linalg_module = m.def_submodule("linalg","The linear system solvers of the proxSuite library");
  //exposeDenseLinalg<proxsuite::proxqp::f64>(linalg_module);
  //exposeSparseLinalg<proxsuite::proxqp::f64,int32_t>(linalg_module);
  pybind11::module_ socp_module = m.def_submodule("proxsocp","The proxSocp solvers of the proxSuite library");
  

  pybind11::module_ dense_socp_module =
    socp_module.def_submodule("dense", "Dense solver of proxSOCP");
  pybind11::module_ sparse_socp_module =
    socp_module.def_submodule("sparse", "Sparse solver of proxSOCP");
  exposeSparseSocp<c_float,c_int>(sparse_socp_module);
  exposeDenseSocp<proxsuite::proxqp::f64>(dense_socp_module);

  pybind11::module_ proxqp_module =
    m.def_submodule("proxqp", "The proxQP solvers of the proxSuite library");
  exposeCommon<proxsuite::proxqp::f64>(proxqp_module);
  pybind11::module_ dense_module =
    proxqp_module.def_submodule("dense", "Dense solver of proxQP");
  exposeDenseAlgorithms<proxsuite::proxqp::f64>(dense_module);
  pybind11::module_ sparse_module =
    proxqp_module.def_submodule("sparse", "Sparse solver of proxQP");
  exposeSparseAlgorithms<proxsuite::proxqp::f64, int32_t>(sparse_module);

  // Add version
  m.attr("__version__") = helpers::printVersion();

  // Add helpers
  pybind11::module_ helpers_module =
    m.def_submodule("helpers", "Helper module");
  helpers_module.def("printVersion",
                     helpers::printVersion,
                     pybind11::arg("delimiter") = ".",
                     "Print the current version of the package.");
  helpers_module.def("checkVersionAtLeast",
                     helpers::checkVersionAtLeast,
                     pybind11::arg("major_version"),
                     pybind11::arg("minor_version"),
                     pybind11::arg("patch_version"),
                     "Check version of the package is at least greater than "
                     "the one provided as input.");
}

} // namespace python

//} // namespace proxqp
} // namespace proxsuite
