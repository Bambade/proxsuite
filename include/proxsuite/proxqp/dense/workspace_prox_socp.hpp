//
// Copyright (c) 2022, INRIA
//
/** \file */

#ifndef PROXSUITE_QP_DENSE_ProxSocpWorkspace_HPP
#define PROXSUITE_QP_DENSE_ProxSocpWorkspace_HPP

#include <proxsuite/linalg/dense/core.hpp>
#include <proxsuite/linalg/veg/vec.hpp>

#include "proxsuite/proxqp/dense/fwd.hpp"

#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/proxqp/timings.hpp>
#include <proxsuite/proxqp/dense/views.hpp>
#include <proxsuite/proxqp/sparse/views.hpp>
#include <proxsuite/proxqp/sparse/utils.hpp>

#include <iostream>
#include <memory>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace proxqp {
namespace dense {

template<typename T>
struct SocpView
{
  static constexpr Layout layout = rowmajor;

  MatrixViewMut<T, layout> H;
  VectorViewMut<T> g;

  MatrixViewMut<T, layout> A;

  VectorViewMut<T> l;
  VectorViewMut<T> u;

  VEG_INLINE constexpr auto as_const() const noexcept -> SocpView<T>
  {
    return {
      H.as_const(), g.as_const(), A.as_const(),
      l.as_const(), u.as_const()
    };
  }
};

template<typename Scalar>
struct SocpViewMut
{
  static constexpr Layout layout = rowmajor;

  MatrixViewMut<Scalar, layout> H;
  VectorViewMut<Scalar> g;

  MatrixViewMut<Scalar, layout> A;

  VectorViewMut<Scalar> l;
  VectorViewMut<Scalar> u;

  VEG_INLINE constexpr auto as_const() const noexcept -> SocpView<Scalar>
  {
    return {
      H.as_const(), g.as_const(), A.as_const(),
      l.as_const(), u.as_const(),
    };
  }
};

///
/// @brief This class stores the SocpModel of the QP problem.
///
/*!
 * SocpModel class of the sparse solver storing the QP problem structure.
 */
template<typename T>
struct SocpModel
{
  isize dim;
  isize n_eq;
  isize n_in;
  isize n_soc;

  Mat<T> H;
  Mat<T> A;
  Vec<T> g;
  Vec<T> l;
  Vec<T> u;
  Vec<isize> dims;

  /*!
   * Default constructor.
   * @param _dim primal variable dimension.
   * @param _m number of constraints.
   */
  SocpModel(isize _dim, isize _n_eq, isize _n_in, Vec<isize> dims_)
    : dim(_dim),
    n_eq(_n_eq),
	  n_in(_n_in),
	  dims(dims_),
    H(_dim, _dim),
    g(_dim),
    A(_n_eq+_n_in+dims_.sum(), _dim),
    u(_n_eq+_n_in+dims_.sum()),
    l(_n_in)
  {
    PROXSUITE_THROW_PRETTY(_dim == 0,
                           std::invalid_argument,
                           "wrong argument size: the dimension wrt primal "
                           "variable x should be strictly positive.");
    H.setZero();
    A.setZero();
    g.setZero();
    u.setZero();
    l.setZero();
	  n_soc = dims_.sum();
    //std::cout << "n_soc " << n_soc << std::endl;
    // TODO check elements inside dims_
  }
};



template <typename T>
struct ProxSocpSettings {

	T mu_min_eq;
	T mu_min_in;
	T mu_min_soc;
	
	T tau;

    // mu update
    isize check_termination;
    T mu_update_fact_bound;

	T mu_update_factor;

	isize max_iter;
	T eps_abs;
	T eps_rel;
	T eps_refact;
	isize nb_iterative_refinement;

	bool verbose;
	bool update_preconditioner;
	bool compute_preconditioner;
	bool compute_timings;
	InitialGuessStatus initial_guess;

	T eps_primal_inf;
	T eps_dual_inf;
	isize preconditioner_max_iter;
	T preconditioner_accuracy;
	/*!
	 * Default constructor.
	 * @param mu_min_eq_ minimal authorized value for mu_eq.
	 * @param mu_min_in_ minimal authorized value for mu_in.
	 * @param mu_min_soc_ minimal authorized value for mu_soc.
	 * @param tau parameter update for slack variables.
	 * @param mu_update_factor_ update factor used for updating mu_eq and mu_in.
     * @param mu_update_fact_bound bound above which mu_in and mu_eq are updated (if acceleration strategy is used).
     * @param check_termination parameter used for calibrating when a mu_eq or mu_in update is made.
	 * @param eps_abs_ asbolute stopping criterion of the solver.
	 * @param eps_rel_ relative stopping criterion of the solver.
	 * @param max_iter_ maximal number of authorized iteration.
	 * @param nb_iterative_refinement_ number of iterative refinements.
	 * @param eps_refact_ threshold value for refactorizing the ldlt factorization in the iterative refinement loop.
	 * @param safe_guard safeguard parameter ensuring global convergence of ProxQP scheme.
	 * @param VERBOSE if set to true, the solver prints information at each loop. 
	 * @param initial_guess_ sets the initial guess option for initilizing x, y and z.
	 * @param update_preconditioner_ If set to true, the preconditioner will be re-derived with the update method.
	 * @param compute_preconditioner_ If set to true, the preconditioner will be derived with the init method.
	 * @param compute_timings_ If set to true, timings will be computed by the solver (setup time, solving time, and run time = setup time + solving time).
	 * @param preconditioner_max_iter_ maximal number of authorized iterations for the preconditioner.
	 * @param preconditioner_accuracy_ accuracy level of the preconditioner.
	 * @param eps_primal_inf_ threshold under which primal infeasibility is detected.
	 * @param eps_dual_inf_ threshold under which dual infeasibility is detected.
	 * @param constant_update if set to true, constant update strategy is used for calibrating mu_eq and mu_in. If set to false, acceleration strategy is used.
	 */
	ProxSocpSettings(
			T mu_min_eq_ = 1e-9,
			T mu_min_in_ = 1e-6,
			T mu_min_soc_ = 1.E-6,
			T tau_ = 0.1,
			T mu_update_factor_ = 0.1,
			T eps_abs_ = 1.e-3,
			T eps_rel_ = 0,
			isize max_iter_ = 10000,
			isize nb_iterative_refinement_ = 10,
			T eps_refact_ = 1.e-6,
			bool VERBOSE = false,
			InitialGuessStatus initial_guess_ = InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT,
			bool update_preconditioner_ = true,
			bool compute_preconditioner_ = true,
			bool compute_timings_ = true,
			isize preconditioner_max_iter_ = 10,
			T preconditioner_accuracy_ = 1.e-3,
			T eps_primal_inf_ = 1.E-4,
			T eps_dual_inf_ = 1.E-4,
            T mu_update_fact_bound_ = 0.95,
            isize check_termination_ = 25
            )
			:
				mu_min_eq(mu_min_eq_),
				mu_min_in(mu_min_in_),
				mu_min_soc(mu_min_soc_),
				tau(tau_),
				mu_update_factor(mu_update_factor_),
				max_iter(max_iter_),
				eps_abs(eps_abs_),
				eps_rel(eps_rel_),
				eps_refact(eps_refact_),
				nb_iterative_refinement(nb_iterative_refinement_),
				verbose(VERBOSE),
				update_preconditioner(update_preconditioner_),
				compute_preconditioner(compute_preconditioner_),
				compute_timings(compute_timings_),
				initial_guess(initial_guess_),
				eps_primal_inf(eps_primal_inf_),
				eps_dual_inf(eps_dual_inf_),
				preconditioner_max_iter(preconditioner_max_iter_),
				preconditioner_accuracy(preconditioner_accuracy_),
                mu_update_fact_bound(mu_update_fact_bound_),
                check_termination(check_termination_)
                 {}
};


template <typename T>
struct ProxSocpInfo {
    ///// final proximal regularization parameters
    T mu_eq;
    T mu_in;
	T mu_soc;
    T mu_eq_inv;
    T mu_in_inv;
	T mu_soc_inv;
    T rho;			

    ///// iteration count
    isize iter;
    isize mu_updates;

	QPSolverOutput status;

    //// timings
	T setup_time;
	T solve_time;
	T run_time;
	T objValue;
	T pri_res;
	T dua_res;
};

template <typename T>
struct ProxSocpResults {
public:

    ///// SOLUTION STORAGE

    Vec<T> x;
    Vec<T> y;
    Vec<T> z;
	proxsuite::linalg::veg::Vec<bool> active_constraints;

	ProxSocpInfo<T> info;

	ProxSocpResults( isize dim=0, isize m = 0,isize n_eq=0)
			: //
                x(dim),
                y(m),
                z(m-n_eq)
                {
        
                x.setZero();
                y.setZero();
                z.setZero();
				proxsuite::linalg::veg::Vec<bool> active_constraints;

                info.rho = T(1e-6);
	            info.mu_eq = T(1e-10);
	            info.mu_in = T(1e-6);
				info.mu_soc = T(1.e-2);

	            info.mu_eq_inv = T(1e10);
	            info.mu_in_inv = T(1e6);
				info.mu_soc_inv = T(1.e2);

                info.iter = 0;
                info.mu_updates = 0;
                info.solve_time = 0.;
				info.setup_time = 0.;
                info.objValue =0.;
                
                }
    
    void cleanup(){
        x.setZero();
        y.setZero();
        z.setZero();


		info.rho = T(1e-6);
		info.mu_eq = T(1e-10);
		info.mu_in = T(1e-6);
		info.mu_soc = T(1.e-2);

		info.mu_eq_inv = T(1e10);
		info.mu_in_inv = T(1e6);
		info.mu_soc_inv = T(1.e2);

		info.iter = 0;
		info.mu_updates = 0;
		info.solve_time = 0.;
		info.setup_time = 0.;
		info.objValue =0.;
                
    }
};
///
/// @brief This class defines the workspace of the sparse solver.
///
/*!
 * ProxSocpWorkspace class of the sparse solver.
*/
template <typename T>
struct ProxSocpWorkspace {

  ///// Cholesky Factorization
  proxsuite::linalg::dense::Ldlt<T> ldl{};
  proxsuite::linalg::veg::Vec<unsigned char> ldl_stack;
  Timer<T> timer;

  ///// QP STORAGE
  Mat<T> H_scaled;
  Vec<T> g_scaled;
  Mat<T> A_scaled;
  Vec<T> u_scaled;
  Vec<T> l_scaled;

  ///// Initial variable loading

  Vec<T> x_prev;
  Vec<T> y_prev;
  Vec<T> z_hat;

  ///// KKT system storage
  Mat<T> kkt;

  ///// Newton variables
  Vec<T> dw;
  Vec<T> rhs;
  Vec<T> err;

  //// Relative residuals constants

  T primal_feasibility_rhs_1_in_u;
  T primal_feasibility_rhs_1_in_l;
  T dual_feasibility_rhs_2;
  T correction_guess_rhs_g;
  T correction_guess_rhs_b;

  Vec<T> dual_residual_scaled;
  Vec<T> primal_residual;

  bool constraints_changed;
  bool dirty;
  bool refactorize;
  bool proximal_parameter_update;

  /*!
   * Default constructor.
   * @param dim primal variable dimension.
   * @param n_eq number of equality constraints.
   * @param n_in number of inequality constraints.
   */
  ProxSocpWorkspace(isize dim, isize n_eq, isize n_in, Vec<isize> dims)
    : //
      // ruiz(preconditioner::RuizEquilibration<T>{dim, n_eq + n_in}),
    ldl{}
    , // old version with alloc
    H_scaled(dim, dim)
    , g_scaled(dim)
    , A_scaled(n_eq+n_in+dims.sum(), dim)
    , u_scaled(n_eq+n_in+dims.sum())
    , l_scaled(n_in)
    , x_prev(dim)
    , y_prev(n_eq+n_in+dims.sum())
    , z_hat(n_in+dims.sum())
    , kkt(dim + n_eq+n_in+dims.sum(), dim + n_eq+n_in+dims.sum())
    , dw(dim + n_eq+n_in+dims.sum())
    , rhs(dim + n_eq+n_in+dims.sum())
    , err(dim + n_eq+n_in+dims.sum())
    , dual_residual_scaled(dim)
    , primal_residual(n_eq+n_in+dims.sum())
    , constraints_changed(false)
    , dirty(false)
    , refactorize(false)
    , proximal_parameter_update(false)

  {
    ldl.reserve_uninit(dim + n_eq+n_in+dims.sum());
    ldl_stack.resize_for_overwrite(
      proxsuite::linalg::veg::dynstack::StackReq(

        proxsuite::linalg::dense::Ldlt<T>::factorize_req(dim + n_eq+n_in+dims.sum()) |

        (proxsuite::linalg::dense::temp_vec_req(
           proxsuite::linalg::veg::Tag<T>{}, n_eq+n_in+dims.sum()) & 
         proxsuite::linalg::veg::dynstack::StackReq{
           isize{ sizeof(isize) } * (n_eq+n_in+dims.sum()), alignof(isize) } &
         proxsuite::linalg::dense::Ldlt<T>::diagonal_update_req(
           dim + n_eq + n_in+dims.sum(), n_eq+n_in+dims.sum())) |

        //(proxsuite::linalg::dense::temp_mat_req(
        //   proxsuite::linalg::veg::Tag<T>{}, dim + n_eq + n_in+dims.sum(), n_in+dims.sum()) & //potentiellement inutile ?? faire des checks sur des tests qu'il n'y a plus d'insertions de lignes
        // proxsuite::linalg::dense::Ldlt<T>::insert_block_at_req(
        //   dim + n_eq + n_in+dims.sum(), n_in+dims.sum())) |

        proxsuite::linalg::dense::Ldlt<T>::solve_in_place_req(dim + n_eq +
                                                              n_in+dims.sum()))

        .alloc_req());

    H_scaled.setZero();
    g_scaled.setZero();
    A_scaled.setZero();
    u_scaled.setZero();
    l_scaled.setZero();
    x_prev.setZero();
    y_prev.setZero();
    z_hat.setZero();
    kkt.setZero();
    dw.setZero();
    rhs.setZero();
    err.setZero();

    primal_feasibility_rhs_1_in_u = 0;
    primal_feasibility_rhs_1_in_l = 0;
    dual_feasibility_rhs_2 = 0;

    dual_residual_scaled.setZero();
    primal_residual.setZero();
  }
  /*!
   * Clean-ups solver's workspace.
   */
  void cleanup()
  {
    H_scaled.setZero();
    g_scaled.setZero();
    A_scaled.setZero();
    u_scaled.setZero();
    l_scaled.setZero();
    x_prev.setZero();
    y_prev.setZero();
    z_hat.setZero();
    kkt.setZero();
    dw.setZero();
    rhs.setZero();
    err.setZero();
    primal_feasibility_rhs_1_in_u = 0;
    primal_feasibility_rhs_1_in_l = 0;
    dual_feasibility_rhs_2 = 0;
    correction_guess_rhs_g = 0;
    correction_guess_rhs_b = 0;

    dual_residual_scaled.setZero();
    primal_residual.setZero();
  }
};

} //namespace dense
} //namespace proxqp
} //namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_DENSE_ProxSocpWorkspace_HPP */
