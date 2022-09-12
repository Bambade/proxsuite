//
// Copyright (c) 2022 INRIA
//
/** \file */

#ifndef PROXSUITE_QP_SPARSE_PRECOND_RUIZ_SOCP_HPP
#define PROXSUITE_QP_SPARSE_PRECOND_RUIZ_SOCP_HPP

#include "proxsuite/proxqp/sparse/fwd.hpp"

namespace proxsuite {
namespace proxqp {
namespace sparse {

namespace preconditioner {
//enum struct Symmetry
//{
//  LOWER,
//  UPPER,
//};

namespace detail {
//template<typename T, typename I>
//void
//rowwise_infty_norm(T* row_norm, proxsuite::linalg::sparse::MatRef<T, I> m)
//{
//  using namespace proxsuite::linalg::sparse::util;
//
//  I const* mi = m.row_indices();
//  T const* mx = m.values();
//
//  for (usize j = 0; j < usize(m.ncols()); ++j) {
//    auto col_start = m.col_start(j);
//    auto col_end = m.col_end(j);
//
//    for (usize p = col_start; p < col_end; ++p) {
//      usize i = zero_extend(mi[p]);
//      T mij = fabs(mx[p]);
//      row_norm[i] = std::max(row_norm[i], mij);
//    }
//  }
//}
//
//template<typename T, typename I>
//void
//colwise_infty_norm_symhi(T* col_norm, proxsuite::linalg::sparse::MatRef<T, I> h)
//{
//  using namespace proxsuite::linalg::sparse::util;
//
//  I const* hi = h.row_indices();
//  T const* hx = h.values();
//
//  for (usize j = 0; j < usize(h.ncols()); ++j) {
//    auto col_start = h.col_start(j);
//    auto col_end = h.col_end(j);
//
//    T norm_j = 0;
//
//    for (usize p = col_start; p < col_end; ++p) {
//      usize i = zero_extend(hi[p]);
//      if (i > j) {
//        break;
//      }
//
//      T hij = fabs(hx[p]);
//      norm_j = std::max(norm_j, hij);
//      col_norm[i] = std::max(col_norm[i], hij);
//    }
//
//    col_norm[j] = norm_j;
//  }
//}
//
//template<typename T, typename I>
//void
//colwise_infty_norm_symlo(T* col_norm, proxsuite::linalg::sparse::MatRef<T, I> h)
//{
//  using namespace proxsuite::linalg::sparse::util;
//
//  I const* hi = h.row_indices();
//  T const* hx = h.values();
//
//  for (usize j = 0; j < usize(h.ncols()); ++j) {
//    auto col_start = h.col_start(j);
//    auto col_end = h.col_end(j);
//
//    T norm_j = 0;
//
//    if (col_end > col_start) {
//      usize p = col_end;
//      while (true) {
//        --p;
//        usize i = zero_extend(hi[p]);
//        if (i < j) {
//          break;
//        }
//
//        T hij = fabs(hx[p]);
//        norm_j = std::max(norm_j, hij);
//        col_norm[i] = std::max(col_norm[i], hij);
//
//        if (p <= col_start) {
//          break;
//        }
//      }
//    }
//    col_norm[j] = std::max(col_norm[j], norm_j);
//  }
//}

//template<typename T, typename I>
//void
//colwise_infty_norm(T* col_norm, proxsuite::linalg::sparse::MatRef<T, I> h)
//{
//  using namespace proxsuite::linalg::sparse::util;
//
//  I const* hi = h.row_indices();
//  T const* hx = h.values();
//
//  for (usize j = 0; j < usize(h.ncols()); ++j) {
//    auto col_start = h.col_start(j);
//    auto col_end = h.col_end(j);
//
//    T norm_j = 0;
//
//    for (usize p = col_start; p < col_end; ++p) {
//      usize i = zero_extend(hi[p]);
//      //if (i > j) {
//      //  break;
//      //}
//
//      T hij = fabs(hx[p]);
//      norm_j = std::max(norm_j, hij);
//      col_norm[i] = std::max(col_norm[i], hij);
//    }
//
//    col_norm[j] = norm_j;
//  }
//}

template<typename T, typename I>
void
colwise_infty_norm(T* row_norm, proxsuite::linalg::sparse::MatRef<T, I> m)
{
  using namespace proxsuite::linalg::sparse::util;

  I const* mi = m.row_indices();
  T const* mx = m.values();

  for (usize j = 0; j < usize(m.ncols()); ++j) {
    auto col_start = m.col_start(j);
    auto col_end = m.col_end(j);

    for (usize p = col_start; p < col_end; ++p) {
      usize i = zero_extend(mi[p]);
      T mij = fabs(mx[p]);
      row_norm[j] = std::max(row_norm[j], mij);
    }
  }
}


template<typename T, typename I>
auto
ruiz_scale_socp_in_place( //
  VectorViewMut<T> delta_,
  SocpViewMut<T, I> socp,
  isize n,
  isize n_eq,
  isize n_in,
  isize n_soc,
  T epsilon,
  isize max_iter,
  proxsuite::proxqp::sparse::preconditioner::Symmetry sym,
  proxsuite::linalg::veg::dynstack::DynStackMut stack) -> T
{

  T c = 1;
  auto S = delta_.to_eigen();

  isize m = n_eq+n_in+n_soc;

  T gamma = 1;
  i64 iter = 1;

  LDLT_TEMP_VEC(T, delta, n + m, stack);

  I* Hi = socp.H.row_indices_mut();
  T* Hx = socp.H.values_mut();

  I* ATi = socp.AT.row_indices_mut();
  T* ATx = socp.AT.values_mut();

  T const machine_eps = std::numeric_limits<T>::epsilon();

  while (infty_norm((1 - delta.array()).matrix()) > epsilon) {
    //std::cout <<"err " << infty_norm((1 - delta.array()).matrix()) << std::endl;
    if (iter == max_iter) {
      break;
    } else {
      ++iter;
    }
    
    // norm_infty of each column of A, i.e.,
    // each row of AT
    
    {
      /////////// using A and not AT
      auto _a_infty_norm = stack.make_new(proxsuite::linalg::veg::Tag<T>{}, n); // n col with m rows
      
      auto _h_infty_norm = stack.make_new(proxsuite::linalg::veg::Tag<T>{}, n);
      T* a_infty_norm = _a_infty_norm.ptr_mut();
      
      T* h_infty_norm = _h_infty_norm.ptr_mut();

      proxsuite::proxqp::sparse::preconditioner::detail::colwise_infty_norm(a_infty_norm, socp.AT.as_const());
      switch (sym) {
        case proxsuite::proxqp::sparse::preconditioner::Symmetry::LOWER: {
          proxsuite::proxqp::sparse::preconditioner::detail::colwise_infty_norm_symlo(h_infty_norm, socp.H.as_const());
          break;
        }
        case proxsuite::proxqp::sparse::preconditioner::Symmetry::UPPER: {
          proxsuite::proxqp::sparse::preconditioner::detail::colwise_infty_norm_symhi(h_infty_norm, socp.H.as_const());
          break;
        }
      }

      for (isize j = 0; j < n; ++j) {
        //std::cout << "j " << j << " col norm " << std::max({h_infty_norm[j],a_infty_norm[j]}) << std::endl;
        T col = std::max({
                                           h_infty_norm[j],
                                           a_infty_norm[j],
                                         });
        if (col == 0){
          col = T(1);
        }
        delta(j) = T(1) / (machine_eps + sqrt(col));
      }
    }
    using namespace proxsuite::linalg::sparse::util;
    {
    auto _a_infty_row_norm = stack.make_new(proxsuite::linalg::veg::Tag<T>{}, m); // m rows with n cols
    T* a_infty_row_norm = _a_infty_row_norm.ptr_mut();
    proxsuite::proxqp::sparse::preconditioner::detail::rowwise_infty_norm(a_infty_row_norm, socp.AT.as_const());
    // idem for the rows
    for (usize j = 0; j < usize(m); ++j) {
      //T a_row_norm = 0;
      //socp.AT.to_eigen();
      //usize col_start = socp.AT.col_start(j);
      //usize col_end = socp.AT.col_end(j);
      //for (usize p = col_start; p < col_end; ++p) {
      //  T aji = fabs(ATx[p]);
      //  a_row_norm = std::max(a_row_norm, aji);
      //}
      T col = a_infty_row_norm[j];
        if (col == 0){
          col = T(1);
        }
      delta(n + isize(j)) = T(1) / (machine_eps + sqrt(col));
      //std::cout << "j " << j+n << " a_row_norm " << a_row_norm << std::endl;
    }
    }
    // normalize A
    for (usize j = 0; j < usize(n); ++j) {
      usize col_start = socp.AT.col_start(j);
      usize col_end = socp.AT.col_end(j);

      //T delta_j = delta(n + isize(j));

      T delta_i = delta(isize(j));

      for (usize p = col_start; p < col_end; ++p) {
        usize i = zero_extend(ATi[p]);
        
        T& aji = ATx[p];

        //T delta_i = delta(isize(i));
        T delta_j = delta(isize(i+n));
        aji = delta_i * (aji * delta_j);
      }
    }
    /*
    ///////////////////////////// using A.T //////////////////
      auto _a_infty_norm = stack.make_new(proxsuite::linalg::veg::Tag<T>{}, n); // n col with m rows
      
      auto _h_infty_norm = stack.make_new(proxsuite::linalg::veg::Tag<T>{}, n);
      T* a_infty_norm = _a_infty_norm.ptr_mut();
      T* h_infty_norm = _h_infty_norm.ptr_mut();
      proxsuite::proxqp::sparse::preconditioner::detail::rowwise_infty_norm(a_infty_norm, socp.AT.as_const());
      switch (sym) {
        case proxsuite::proxqp::sparse::preconditioner::Symmetry::LOWER: {
          proxsuite::proxqp::sparse::preconditioner::detail::colwise_infty_norm_symlo(h_infty_norm, socp.H.as_const());
          break;
        }
        case proxsuite::proxqp::sparse::preconditioner::Symmetry::UPPER: {
          proxsuite::proxqp::sparse::preconditioner::detail::colwise_infty_norm_symhi(h_infty_norm, socp.H.as_const());
          break;
        }
      }

      for (isize j = 0; j < n; ++j) {
        //std::cout << "j " << j << " col norm " << std::max({h_infty_norm[j],a_infty_norm[j]}) << std::endl;
        T col = std::max({
                                           h_infty_norm[j],
                                           a_infty_norm[j],
                                         });
        if (col == 0){
          col = T(1);
        }
        delta(j) = T(1) / (machine_eps + sqrt(col));
      }
    }
    using namespace proxsuite::linalg::sparse::util;
    {
    // idem for the rows
    for (usize j = 0; j < usize(m); ++j) {
      T a_row_norm = 0;
      socp.AT.to_eigen();
      usize col_start = socp.AT.col_start(j);
      usize col_end = socp.AT.col_end(j);
      for (usize p = col_start; p < col_end; ++p) {
        T aji = fabs(ATx[p]);
        a_row_norm = std::max(a_row_norm, aji);
      }
      if (a_row_norm == 0){
        a_row_norm = T(1);
      }
      delta(n + isize(j)) = T(1) / (machine_eps + sqrt(a_row_norm));
    }
    }
    // normalize A
    for (usize j = 0; j < usize(m); ++j) {
      usize col_start = socp.AT.col_start(j);
      usize col_end = socp.AT.col_end(j);

      T delta_j = delta(n + isize(j));

      for (usize p = col_start; p < col_end; ++p) {
        usize i = zero_extend(ATi[p]);
        
        T& aji = ATx[p];

        T delta_i = delta(isize(i));
        aji = delta_i * (aji * delta_j);
      }
    }
    ///////////////////////////// using A.T //////////////////
    */
    // normalize H
    switch (sym) {
      case proxsuite::proxqp::sparse::preconditioner::Symmetry::LOWER: {
        for (usize j = 0; j < usize(n); ++j) {
          usize col_start = socp.H.col_start(j);
          usize col_end = socp.H.col_end(j);
          T delta_j = delta(isize(j));

          if (col_end > col_start) {
            usize p = col_end;
            while (true) {
              --p;
              usize i = zero_extend(Hi[p]);
              if (i < j) {
                break;
              }
              Hx[p] = delta_j * Hx[p] * delta(isize(i));

              if (p <= col_start) {
                break;
              }
            }
          }
        }
        break;
      }
      case proxsuite::proxqp::sparse::preconditioner::Symmetry::UPPER: {
        for (usize j = 0; j < usize(n); ++j) {
          usize col_start = socp.H.col_start(j);
          usize col_end = socp.H.col_end(j);
          T delta_j = delta(isize(j));

          for (usize p = col_start; p < col_end; ++p) {
            usize i = zero_extend(Hi[p]);
            if (i > j) {
              break;
            }
            Hx[p] = delta_j * Hx[p] * delta(isize(i));
          }
        }
        break;
      }
    }

    // normalize vectors
    socp.g.to_eigen().array() *= delta.head(n).array();
    socp.l.to_eigen().array() *= delta.segment(n+n_eq,n_in).array();
    socp.u.to_eigen().array() *= delta.tail(m).array();

    // additional normalization
    auto _h_infty_norm = stack.make_new(proxsuite::linalg::veg::Tag<T>{}, n);
    T* h_infty_norm = _h_infty_norm.ptr_mut();

    switch (sym) {
      case proxsuite::proxqp::sparse::preconditioner::Symmetry::LOWER: {
        proxsuite::proxqp::sparse::preconditioner::detail::colwise_infty_norm_symlo(h_infty_norm, socp.H.as_const());
        break;
      }
      case proxsuite::proxqp::sparse::preconditioner::Symmetry::UPPER: {
        proxsuite::proxqp::sparse::preconditioner::detail::colwise_infty_norm_symhi(h_infty_norm, socp.H.as_const());
        break;
      }
    }

    T avg = 0;
    for (isize i = 0; i < n; ++i) {
      avg += h_infty_norm[i];
    }
    avg /= T(n);

    gamma = 1 / std::max(avg, infty_norm(socp.g.to_eigen()));

    socp.g.to_eigen() *= gamma;
    socp.H.to_eigen() *= gamma;

    S.array() *= delta.array();
    c *= gamma;
  }
  return c;
}
} // namespace detail

template<typename T, typename I>
struct RuizSocpEquilibration
{
  Eigen::Matrix<T, -1, 1> delta;
  isize n;
  isize n_in;
  isize n_eq;
  isize m;
  T c;
  T epsilon;
  i64 max_iter;
  proxsuite::proxqp::sparse::preconditioner::Symmetry sym;

  std::ostream* logger_ptr = nullptr;

  RuizSocpEquilibration(isize n_,
                    isize m_,
                    isize n_eq_,
                    isize n_in_,
                    T epsilon_ = T(1e-3),
                    i64 max_iter_ = 10,
                    proxsuite::proxqp::sparse::preconditioner::Symmetry sym_ = proxsuite::proxqp::sparse::preconditioner::Symmetry::UPPER,
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
    return proxsuite::linalg::dense::temp_vec_req(tag, n + m) &
           proxsuite::linalg::veg::dynstack::StackReq::with_len(tag, 2 * n);
  }

  void scale_socp_in_place(SocpViewMut<T, I> socp,
                         bool execute_preconditioner,
                         const isize max_iter,
                         const T epsilon,
                         proxsuite::linalg::veg::dynstack::DynStackMut stack)
  {
    if (execute_preconditioner) {
      delta.setOnes();
      isize n_soc = m-n_eq-n_in;
      c = detail::ruiz_scale_socp_in_place( //
        { proxqp::from_eigen, delta },
        socp,
        n,
        n_eq,
        n_in,
        n_soc,
        epsilon,
        max_iter,
        sym,
        stack);
    } else { // TO CHANGE USING ONLY A AND NOT AT
      using proxsuite::linalg::sparse::util::zero_extend;

      I* Hi = socp.H.row_indices_mut();
      T* Hx = socp.H.values_mut();

      I* ATi = socp.AT.row_indices_mut();
      T* ATx = socp.AT.values_mut();

      // normalize AT
      /*
      for (usize j = 0; j < usize(m); ++j) {
        usize col_start = socp.AT.col_start(j);
        usize col_end = socp.AT.col_end(j);

        T delta_j = delta(n + isize(j));

        for (usize p = col_start; p < col_end; ++p) {
          usize i = zero_extend(ATi[p]);
          T& aji = ATx[p];
          T delta_i = delta(isize(i));
          aji = delta_i * (aji * delta_j);
        }
      }
      */
      // normalize A
      for (usize j = 0; j < usize(n); ++j) {
        usize col_start = socp.AT.col_start(j);
        usize col_end = socp.AT.col_end(j);

        //T delta_j = delta(n + isize(j));

        T delta_i = delta(isize(j));

        for (usize p = col_start; p < col_end; ++p) {
          usize i = zero_extend(ATi[p]);
          
          T& aji = ATx[p];

          //T delta_i = delta(isize(i));
          T delta_j = delta(isize(i+n));
          aji = delta_i * (aji * delta_j);
        }
      }

      // normalize H
      switch (sym) {
        case proxsuite::proxqp::sparse::preconditioner::Symmetry::LOWER: {
          for (usize j = 0; j < usize(n); ++j) {
            usize col_start = socp.H.col_start(j);
            usize col_end = socp.H.col_end(j);
            T delta_j = delta(isize(j));

            if (col_end > col_start) {
              usize p = col_end;
              while (true) {
                --p;
                usize i = zero_extend(Hi[p]);
                if (i < j) {
                  break;
                }
                Hx[p] = delta_j * Hx[p] * delta(isize(i));

                if (p <= col_start) {
                  break;
                }
              }
            }
          }
          break;
        }
        case proxsuite::proxqp::sparse::preconditioner::Symmetry::UPPER: {
          for (usize j = 0; j < usize(n); ++j) {
            usize col_start = socp.H.col_start(j);
            usize col_end = socp.H.col_end(j);
            T delta_j = delta(isize(j));

            for (usize p = col_start; p < col_end; ++p) {
              usize i = zero_extend(Hi[p]);
              if (i > j) {
                break;
              }
              Hx[p] = delta_j * Hx[p] * delta(isize(i));
            }
          }
          break;
        }
      }

      // normalize vectors
      socp.g.to_eigen().array() *= delta.head(n).array();
      socp.l.to_eigen().array() *= delta.segment(n+n_eq,n_in).array();
      socp.u.to_eigen().array() *= delta.tail(m).array();

      socp.g.to_eigen() *= c;
      socp.H.to_eigen() *= c;
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

} // namespace sparse
} // namespace proxqp
} // namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_PRECOND_RUIZ_SOCP_HPP */
