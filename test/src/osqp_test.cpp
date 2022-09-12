//
// Copyright (c) 2022 INRIA
//
#include <doctest.hpp>
#include <proxsuite/linalg/qdldl/lin_sys.h>
#include <proxsuite/linalg/qdldl/lin_alg.h>

#include <Eigen/Sparse>
#include <proxsuite/linalg/sparse/core.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>

#include <proxsuite/proxqp/utils/random_qp_problems.hpp>

#include <proxsuite/linalg/veg/util/dynstack_alloc.hpp>

#include <utils.hpp>
#include "data_osqp.h"

DOCTEST_TEST_CASE("qp: start from solution using the wrapper framework")
{
  c_int m, exitflag = 0;
  c_float *rho_vec;
  LinSysSolver *s;  // Private structure to form KKT factorization
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings)); // Settings
  solve_linsys_sols_data *data = generate_problem_solve_linsys_sols_data();

  // Settings
  settings->rho   = data->test_solve_KKT_rho;
  settings->sigma = data->test_solve_KKT_sigma;

  // Set rho_vec
  m       = data->test_solve_KKT_A->m;
  rho_vec = (c_float*) c_calloc(m, sizeof(c_float));
  vec_add_scalar(rho_vec, settings->rho, m);

  // Form and factorize KKT matrix

  exitflag = init_linsys_solver(&s, data->test_solve_KKT_Pu, data->test_solve_KKT_A,
                                settings->sigma, rho_vec, LINSYS_SOLVER, 0);
  
  std::cout << "exitflag " << exitflag << std::endl;  

  // Solve  KKT x = b via LDL given factorization
  s->solve(s, data->test_solve_KKT_rhs);
  //mu_assert(
  //  "Linear systems solve tests: error in forming and solving KKT system!",
  //  vec_norm_inf_diff(data->test_solve_KKT_rhs, data->test_solve_KKT_x,
  //                    data->test_solve_KKT_m + data->test_solve_KKT_n) < TESTS_TOL);
  c_float err = vec_norm_inf_diff(data->test_solve_KKT_rhs, data->test_solve_KKT_x,
                    data->test_solve_KKT_m + data->test_solve_KKT_n);
  std::cout << "err " << err << std::endl;  
  // Cleanup
  s->free(s);
  c_free(settings);
  c_free(rho_vec);
  clean_problem_solve_linsys_sols_data(data);
}


using T = c_float;
//using I =  proxsuite::proxqp::utils::c_int;
using I = c_int;
template<typename T, typename I>
using SparseMat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;

static constexpr auto DYN = Eigen::Dynamic;
template<typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;


template<typename T, typename I>
struct SocpView
{
  proxsuite::linalg::sparse::MatRef<T, I> H;
  proxsuite::linalg::sparse::DenseVecRef<T> g;
  proxsuite::linalg::sparse::MatRef<T, I> AT;
  proxsuite::linalg::sparse::DenseVecRef<T> l;
  proxsuite::linalg::sparse::DenseVecRef<T> u;
};

template<typename T, typename I>
struct SocpViewMut
{
  proxsuite::linalg::sparse::MatMut<T, I> H;
  proxsuite::linalg::sparse::DenseVecMut<T> g;
  proxsuite::linalg::sparse::MatMut<T, I> AT;
  proxsuite::linalg::sparse::DenseVecMut<T> l;
  proxsuite::linalg::sparse::DenseVecMut<T> u;

  auto as_const() noexcept -> SocpView<T, I>
  {
    return {
      H.as_const(),  g.as_const(), AT.as_const(),
      l.as_const(), u.as_const(),
    };
  }
};
using namespace proxsuite::proxqp;
using namespace proxsuite::proxqp::utils;

//c_float vec_norm_inf_diff(const c_float *a, const c_float *b, c_int l) {
//    c_float nmDiff = 0.0, tmp;
//    c_int   i;
//
//    for (i = 0; i < l; i++) {
//    tmp = c_absval(a[i] - b[i]);
//
//    if (tmp > nmDiff) nmDiff = tmp;
//    }
//    return nmDiff;
//}

DOCTEST_TEST_CASE("another test")
{
  c_int m, exitflag = 0;
  c_float *rho_vec;
  LinSysSolver *s;  // Private structure to form KKT factorization
  OSQPSettings *settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings)); // Settings
  solve_linsys_sols_data *data = generate_problem_solve_linsys_sols_data();

  // Settings
  settings->rho   = data->test_solve_KKT_rho;
  settings->sigma = data->test_solve_KKT_sigma;

  // Set rho_vec
  m       = data->test_solve_KKT_A->m;
  //rho_vec = (c_float*) c_calloc(m, sizeof(c_float));
  //vec_add_scalar(rho_vec, settings->rho, m);

 //   SparseMat<T, I>  A_scaled(data->test_solve_KKT_m,data->test_solve_KKT_n) ;
 //   A_scaled.setZero();
 //   auto H_scaled = static_cast<SparseMat<T, I>>(Eigen::Matrix<T,DYN, DYN>::Identity(data->test_solve_KKT_n,data->test_solve_KKT_n));
 //   //H_scaled.setZero();
 
 //   Vec<T>  g_scaled(data->test_solve_KKT_n) ;
 //   g_scaled.setZero();
 //   Vec<T>  l_scaled(data->test_solve_KKT_m) ;
 //   l_scaled.setZero();
 //   Vec<T>  u_scaled(data->test_solve_KKT_m) ;
 //   u_scaled.setZero();

  c_int n = data->test_solve_KKT_n;
  T p = 1;
  auto H_scaled = proxqp::utils::rand::sparse_positive_definite_rand(n, T(10.0), p);
  auto g_scaled = proxqp::utils::rand::vector_rand<T>(n);
  auto A_scaled = proxqp::utils::rand::sparse_matrix_rand<T>(m, n, p);
  auto l_scaled = proxqp::utils::rand::vector_rand<T>(m);
  auto u_scaled = (l_scaled.array() + 1).matrix().eval();

  SocpViewMut<T, I> socp_view= {
          {proxsuite::linalg::sparse::from_eigen, H_scaled},
          {proxsuite::linalg::sparse::from_eigen, g_scaled},
          {proxsuite::linalg::sparse::from_eigen, A_scaled},
          {proxsuite::linalg::sparse::from_eigen, l_scaled},
          {proxsuite::linalg::sparse::from_eigen, u_scaled},
  };

  LinSysSolver *linsys_solver;
  //const c_float *rho_vec;
  rho_vec     = static_cast<c_float*>(c_malloc(m * sizeof(c_float)));
  for (isize i = 0; i < m; i++) {
      rho_vec[i]     = 1;
  }
  const csc H = socp_view.H.to_csc();
  const csc A = socp_view.AT.to_csc();
  // Form and factorize KKT matrix
  exitflag = init_linsys_solver(&(linsys_solver),&H,&A,
                                c_float(1), rho_vec,
                                QDLDL_SOLVER, c_int(0));
  std::cout << "exitflag 2 " << exitflag << std::endl; 
   // Solve  KKT x = b via LDL given factorization
   linsys_solver->solve(linsys_solver, data->test_solve_KKT_rhs);

   
  // c_float err = vec_norm_inf_diff(data->test_solve_KKT_rhs, data->test_solve_KKT_x,
  //                  data->test_solve_KKT_m + data->test_solve_KKT_n);
  //  std::cout << "err " << err << std::endl;
//   //mu_assert(
//   //  "Linear systems solve tests: error in forming and solving KKT system!",
//   //  vec_norm_inf_diff(data->test_solve_KKT_rhs, data->test_solve_KKT_x,
//   //                    data->test_solve_KKT_m + data->test_solve_KKT_n) < TESTS_TOL);


    // Cleanup
    linsys_solver->free(linsys_solver);
    c_free(settings);
    c_free(rho_vec);
    clean_problem_solve_linsys_sols_data(data);
}
