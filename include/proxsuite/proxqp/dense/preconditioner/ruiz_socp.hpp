//
// Copyright (c) 2022 INRIA
//
/** \file */

#ifndef PROXSUITE_QP_DENSE_PRECOND_RUIZ_SOCP_HPP
#define PROXSUITE_QP_DENSE_PRECOND_RUIZ_SOCP_HPP

#include "proxsuite/proxqp/dense/fwd.hpp"

namespace proxsuite {
namespace proxqp {
namespace dense {

namespace preconditioner {

namespace detail {

template<typename T>
auto
ruiz_scale_socp_in_place( //
  VectorViewMut<T> delta_,
  VectorViewMut<T> tmp_delta_preallocated,
  SocpViewMut<T> socp,
  isize n_eq,
  T epsilon,
  isize max_iter,
  proxsuite::proxqp::Symmetry sym) -> T
{

  T c = 1;
  auto S = delta_.to_eigen();

  isize n = socp.H.rows;
  isize m = socp.A.rows;
  isize n_in = socp.l.to_eigen().rows();

  T gamma = 1;
  i64 iter = 1;

  auto H = socp.H.to_eigen();
  auto g = socp.g.to_eigen();
  auto A = socp.A.to_eigen();
  auto u = socp.u.to_eigen();
  auto l = socp.l.to_eigen();

  auto delta = tmp_delta_preallocated.to_eigen();

  T machine_eps = std::numeric_limits<T>::epsilon();

  while (infty_norm((1 - delta.array()).matrix()) > epsilon) {
    //std::cout << "iter " << iter << " err " << infty_norm((1 - delta.array()).matrix())<< std::endl;
    if (iter == max_iter) {
      break;
    } else {
      ++iter;
    }
    // norm_infty of each column of A, i.e.,
    // each row of AT
    // normalization vector
    {
      for (isize k = 0; k < n; ++k) {
        switch (sym) {
          case proxsuite::proxqp::Symmetry::upper: { // upper triangular part
            T col = std::max({
                                 infty_norm(H.col(k).head(k)),
                                 infty_norm(H.row(k).tail(n - k)),
                                 infty_norm(A.col(k)),
                               });
            if (col==0){
              col = T(1);
            }
            delta(k) = T(1) / (sqrt(col) +
                               machine_eps);
            break;
          }
          case proxsuite::proxqp::Symmetry::lower: { // lower triangular part
            T col = std::max({
                                 infty_norm(H.row(k).head(k)),
                                 infty_norm(H.col(k).tail(n - k)),
                                 infty_norm(A.col(k)),
                               });
            if (col==0){
              col = T(1);
            }    
            delta(k) = T(1) / (sqrt(col) +
                               machine_eps);
            break;
          }
          case proxsuite::proxqp::Symmetry::general: {
            T col = std::max({
                                 infty_norm(H.col(k)),
                                 infty_norm(A.col(k)),
                               });
            if (col==0){
              col = T(1);
            }     
            delta(k) = T(1) / (sqrt(col) +
                               machine_eps);

            break;
          }
        }
      }

      for (isize k = 0; k < m; ++k) {
        T aux = sqrt(infty_norm(A.row(k)));
        if (aux==0){
          aux = T(1);
        }  
        delta(n + k) = T(1) / (aux + machine_eps);
      }
    }
    {

      // normalize A and C
      A = delta.segment(n, m).asDiagonal() * A * delta.head(n).asDiagonal();
      // normalize vectors
      g.array() *= delta.head(n).array();
      u.array() *= delta.tail(m).array();
      l.array() *= delta.segment(n+n_eq,n_in).array();

      // normalize H
      switch (sym) {
        case proxsuite::proxqp::Symmetry::upper: {
          // upper triangular part
          for (isize j = 0; j < n; ++j) {
            H.col(j).head(j + 1) *= delta(j);
          }
          // normalisation des lignes
          for (isize i = 0; i < n; ++i) {
            H.row(i).tail(n - i) *= delta(i);
          }
          break;
        }
        case proxsuite::proxqp::Symmetry::lower: {
          // lower triangular part
          for (isize j = 0; j < n; ++j) {
            H.col(j).tail(n - j) *= delta(j);
          }
          // normalisation des lignes
          for (isize i = 0; i < n; ++i) {
            H.row(i).head(i + 1) *= delta(i);
          }
          break;
        }
        case proxsuite::proxqp::Symmetry::general: {
          // all matrix
          H = delta.head(n).asDiagonal() * H * delta.head(n).asDiagonal();
          break;
        }
        default:
          break;
      }

      // additional normalization for the cost function
      switch (sym) {
        case Symmetry::upper: {
          // upper triangular part
          T tmp = T(0);
          for (isize j = 0; j < n; ++j) {
            tmp += proxqp::dense::infty_norm(H.row(j).tail(n - j));
          }
          gamma = 1 / std::max(tmp / T(n), infty_norm(socp.g.to_eigen()));
          break;
        }
        case Symmetry::lower: {
          // lower triangular part
          T tmp = T(0);
          for (isize j = 0; j < n; ++j) {
            tmp += proxqp::dense::infty_norm(H.col(j).tail(n - j));
          }
          gamma = 1 / std::max(tmp / T(n), infty_norm(socp.g.to_eigen()));
          break;
        }
        case Symmetry::general: {
          // all matrix
          gamma =
            1 /
            std::max(infty_norm(socp.g.to_eigen()),
                     (H.colwise().template lpNorm<Eigen::Infinity>()).mean());
          break;
        }
        default:
          break;
      }

      g *= gamma;
      H *= gamma;

      S.array() *= delta.array(); // coefficientwise product
      c *= gamma;
    }
  }
  return c;
}
} // namespace detail

template<typename T>
struct RuizSocpEquilibration
{
  Vec<T> delta;
  isize n;
  isize n_in;
  isize n_eq;
  isize m;
  T c;
  T epsilon;
  i64 max_iter;
  proxsuite::proxqp::Symmetry sym;

  std::ostream* logger_ptr = nullptr;

  RuizSocpEquilibration(isize n_,
                    isize m_,
                    isize n_eq_,
                    isize n_in_,
                    T epsilon_ = T(1e-3),
                    i64 max_iter_ = 10,
                    proxsuite::proxqp::Symmetry sym_ = proxsuite::proxqp::Symmetry::general,
                    std::ostream* logger = nullptr)
    : delta(Eigen::Matrix<T, -1, 1>::Ones(n_ + m_))
    , n(n_)
    , n_in(n_in_)
    , n_eq(n_eq_)
    , m(m_)
    , c(1)
    , epsilon(epsilon_)
    , max_iter(max_iter_)
    , sym(sym_)
    , logger_ptr(logger)
  {
    delta.setOnes();
  }

  static auto scale_socp_in_place_req(proxsuite::linalg::veg::Tag<T> tag,
                                    isize n,
                                    isize m)
    -> proxsuite::linalg::veg::dynstack::StackReq
  {
    return proxsuite::linalg::dense::temp_vec_req(tag, n + m);
  }

  void scale_socp_in_place(SocpViewMut<T> socp,
                         bool execute_preconditioner,
                         const isize max_iter,
                         const T epsilon,
                         proxsuite::linalg::veg::dynstack::DynStackMut stack)
  {
    if (execute_preconditioner) {
      delta.setOnes();
      LDLT_TEMP_VEC(T, tmp_delta, socp.H.rows + socp.A.rows , stack);
      tmp_delta.setZero();
      c = detail::ruiz_scale_socp_in_place( //
        VectorViewMut<T>{ proxqp::from_eigen, delta },
        VectorViewMut<T>{ proxqp::from_eigen, tmp_delta },
        socp,
        n_eq,
        epsilon,
        max_iter,
        sym);
    } else {

      auto H = socp.H.to_eigen();
      auto g = socp.g.to_eigen();
      auto A = socp.A.to_eigen();
      auto u = socp.u.to_eigen();
      auto l = socp.l.to_eigen();
      isize n = socp.H.rows;
      isize m = socp.A.rows;

      // normalize A and C
      A = delta.segment(n, m).asDiagonal() * A * delta.head(n).asDiagonal();

      // normalize H
      switch (sym) {
        case proxsuite::proxqp::Symmetry::upper: {
          // upper triangular part
          for (isize j = 0; j < n; ++j) {
            H.col(j).head(j + 1) *= delta(j);
          }
          // normalisation des lignes
          for (isize i = 0; i < n; ++i) {
            H.row(i).tail(n - i) *= delta(i);
          }
          break;
        }
        case proxsuite::proxqp::Symmetry::lower: {
          // lower triangular part
          for (isize j = 0; j < n; ++j) {
            H.col(j).tail(n - j) *= delta(j);
          }
          // normalisation des lignes
          for (isize i = 0; i < n; ++i) {
            H.row(i).head(i + 1) *= delta(i);
          }
          break;
        }
        case proxsuite::proxqp::Symmetry::general: {
          // all matrix
          H = delta.head(n).asDiagonal() * H * delta.head(n).asDiagonal();
          break;
        }
        default:
          break;
      }

      // normalize vectors
      g.array() *= delta.head(n).array();
      l.array() *= delta.segment(n+n_eq,n_in).array();
      u.array() *= delta.tail(m).array();

      g *= c;
      H *= c;
    }
  }

  // modifies variables in place
  void scale_primal_in_place(VectorViewMut<T> primal)
  {
    primal.to_eigen().array() /= delta.array().head(n);
  }
  void scale_dual_in_place(VectorViewMut<T> dual)
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() /
                              delta.tail(m).array() * c;
  }

  void unscale_primal_in_place(VectorViewMut<T> primal)
  {
    primal.to_eigen().array() *= delta.array().head(n);
  }
  void unscale_dual_in_place(VectorViewMut<T> dual)
  {
    dual.to_eigen().array() = dual.as_const().to_eigen().array() *
                              delta.tail(m).array() / c;
  }
  // modifies residuals in place
  void scale_primal_residual_in_place(VectorViewMut<T> primal)
  {
    primal.to_eigen().array() *= delta.tail(m).array();
  }

  void scale_dual_residual_in_place(VectorViewMut<T> dual)
  {
    dual.to_eigen().array() *= delta.head(n).array() * c;
  }
  void unscale_primal_residual_in_place(VectorViewMut<T> primal)
  {
    primal.to_eigen().array() /= delta.tail(m).array();
  }
  void unscale_dual_residual_in_place(VectorViewMut<T> dual)
  {
    dual.to_eigen().array() /= delta.head(n).array() * c;
  }
};

} // namespace preconditioner

} // namespace dense
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_PRECOND_RUIZ_SOCP_HPP */
