//
// Copyright (c) 2022, INRIA
//
/** \file */

#ifndef PROXSUITE_QP_SPARSE_ProxSocpWorkspace_HPP
#define PROXSUITE_QP_SPARSE_ProxSocpWorkspace_HPP

#include <proxsuite/linalg/dense/core.hpp>
#include <proxsuite/linalg/sparse/core.hpp>
#include <proxsuite/linalg/sparse/factorize.hpp>
#include <proxsuite/linalg/sparse/update.hpp>
#include <proxsuite/linalg/sparse/rowmod.hpp>
#include <proxsuite/linalg/veg/vec.hpp>

#include <proxsuite/proxqp/status.hpp>
#include <proxsuite/proxqp/timings.hpp>
#include <proxsuite/proxqp/dense/views.hpp>
#include <proxsuite/proxqp/sparse/views.hpp>
#include <proxsuite/proxqp/sparse/utils.hpp>

#include <iostream>
#include <memory>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#include <proxsuite/proxqp/utils/random_qp_problems.hpp>
#include <proxsuite/linalg/qdldl/qdldl_interface.h>
#include <proxsuite/linalg/qdldl/cs.h>

namespace proxsuite {
namespace proxqp {
namespace sparse {

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


///
/// @brief This class stores the SocpModel of the QP problem.
///
/*!
 * SocpModel class of the sparse solver storing the QP problem structure.
 */
template<typename T, typename I>
struct SocpModel
{
  isize dim;
  isize n_eq;
  isize n_in;
  isize n_soc;

  isize H_nnz;
  isize A_nnz;
  /*
  proxsuite::linalg::veg::Vec<I> kkt_col_ptrs;
  proxsuite::linalg::veg::Vec<I> kkt_row_indices;
  proxsuite::linalg::veg::Vec<T> kkt_values;

  proxsuite::linalg::veg::Vec<I> kkt_col_ptrs_unscaled;
  proxsuite::linalg::veg::Vec<I> kkt_row_indices_unscaled;
  proxsuite::linalg::veg::Vec<T> kkt_values_unscaled;
  */

  proxsuite::linalg::veg::Vec<I> A_col_ptrs;
  proxsuite::linalg::veg::Vec<I> A_row_indices;
  proxsuite::linalg::veg::Vec<T> A_values;
  proxsuite::linalg::veg::Vec<I> A_col_ptrs_unscaled;
  proxsuite::linalg::veg::Vec<I> A_row_indices_unscaled;
  proxsuite::linalg::veg::Vec<T> A_values_unscaled;

  proxsuite::linalg::veg::Vec<I> H_col_ptrs;
  proxsuite::linalg::veg::Vec<I> H_row_indices;
  proxsuite::linalg::veg::Vec<T> H_values;
  proxsuite::linalg::veg::Vec<I> H_col_ptrs_unscaled;
  proxsuite::linalg::veg::Vec<I> H_row_indices_unscaled;
  proxsuite::linalg::veg::Vec<T> H_values_unscaled;

  Eigen::Matrix<T, Eigen::Dynamic, 1> g;
  Eigen::Matrix<T, Eigen::Dynamic, 1> l;
  Eigen::Matrix<T, Eigen::Dynamic, 1> u;
  Eigen::Matrix<isize, Eigen::Dynamic, 1> dims;

  /*!
   * Default constructor.
   * @param _dim primal variable dimension.
   * @param _m number of constraints.
   */
  SocpModel(isize _dim, isize _n_eq, isize _n_in, Eigen::Matrix<isize, Eigen::Dynamic, 1> dims_)
    : dim(_dim)
    , n_eq(_n_eq),
	  n_in(_n_in),
	  dims(dims_)
  {
    //PROXSUITE_THROW_PRETTY(_dim == 0,
    //                       std::invalid_argument,SparseMat<T, I>  A_scaled
    //                       "wrong argument size: the dimension wrt primal "
    //                       "variable x should be strictly positive.");
    g.setZero();
    u.setZero();
    l.setZero();
	n_soc = dims_.sum();
	//std::cout << "n_soc " << n_soc << std::endl;
	// TODO check elements inside dims_
  }
  /*!
   * Returns the current (scaled) KKT matrix of the problem.
   */
  //auto kkt() const -> proxsuite::linalg::sparse::MatRef<T, I>
  //{
  //  auto n_tot = kkt_col_ptrs.len() - 1;
  //  auto nnz =
  //    isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
  //  return {
  //    proxsuite::linalg::sparse::from_raw_parts,
  //    n_tot,
  //    n_tot,
  //    nnz,
  //    kkt_col_ptrs.ptr(),
  //    nullptr,
  //    kkt_row_indices.ptr(),
  //    kkt_values.ptr(),
  //  };
  //}
  ///*!
  // * Returns the current (scaled) KKT matrix of the problem (mutable form).
  // */
  //auto kkt_mut() -> proxsuite::linalg::sparse::MatMut<T, I>
  //{
  //  auto n_tot = kkt_col_ptrs.len() - 1;
  //  auto nnz =
  //    isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
  //  return {
  //    proxsuite::linalg::sparse::from_raw_parts,
  //    n_tot,
  //    n_tot,
  //    nnz,
  //    kkt_col_ptrs.ptr_mut(),
  //    nullptr,
  //    kkt_row_indices.ptr_mut(),
  //    kkt_values.ptr_mut(),
  //  };
  //}
  ///*!
  // * Returns the original (unscaled) KKT matrix of the problem.
  // */
  //auto kkt_unscaled() const -> proxsuite::linalg::sparse::MatRef<T, I>
  //{
  //  auto n_tot = kkt_col_ptrs_unscaled.len() - 1;
  //  auto nnz = isize(proxsuite::linalg::sparse::util::zero_extend(
  //    kkt_col_ptrs_unscaled[n_tot]));
  //  return {
  //    proxsuite::linalg::sparse::from_raw_parts,
  //    n_tot,
  //    n_tot,
  //    nnz,
  //    kkt_col_ptrs_unscaled.ptr(),
  //    nullptr,
  //    kkt_row_indices_unscaled.ptr(),
  //    kkt_values_unscaled.ptr(),
  //  };
  //}
  ///*!
  // * Returns the original (unscaled) KKT matrix of the problem (mutable form).
  // */
  //auto kkt_mut_unscaled() -> proxsuite::linalg::sparse::MatMut<T, I>
  //{
  //  auto n_tot = kkt_col_ptrs_unscaled.len() - 1;
  //  auto nnz = isize(proxsuite::linalg::sparse::util::zero_extend(
  //    kkt_col_ptrs_unscaled[n_tot]));
  //  return {
  //    proxsuite::linalg::sparse::from_raw_parts,
  //    n_tot,
  //    n_tot,
  //    nnz,
  //    kkt_col_ptrs_unscaled.ptr_mut(),
  //    nullptr,
  //    kkt_row_indices_unscaled.ptr_mut(),
  //    kkt_values_unscaled.ptr_mut(),
  //  };
  //}
  /*!
   * Returns the current (scaled) A matrix of the problem.
   */
  auto A() const -> proxsuite::linalg::sparse::MatRef<T, I>
  {
    auto n_tot = A_col_ptrs.len() - 1;
    auto nnz =
      isize(proxsuite::linalg::sparse::util::zero_extend(A_col_ptrs[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_eq+n_in+n_soc,
      n_tot,
      nnz,
      A_col_ptrs.ptr(),
      nullptr,
      A_row_indices.ptr(),
      A_values.ptr(),
    };
  }
  /*!
   * Returns the current (scaled) A matrix of the problem (mutable form).
   */
  auto A_mut() -> proxsuite::linalg::sparse::MatMut<T, I>
  {
    auto n_tot = A_col_ptrs.len() - 1;
    auto nnz =
      isize(proxsuite::linalg::sparse::util::zero_extend(A_col_ptrs[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_eq+n_in+n_soc,
      n_tot,
      nnz,
      A_col_ptrs.ptr_mut(),
      nullptr,
      A_row_indices.ptr_mut(),
      A_values.ptr_mut(),
    };
  }
  /*!
   * Returns the original (unscaled) A matrix of the problem.
   */
  auto A_unscaled() const -> proxsuite::linalg::sparse::MatRef<T, I>
  {
    auto n_tot = A_col_ptrs_unscaled.len() - 1;
    auto nnz = isize(proxsuite::linalg::sparse::util::zero_extend(
      A_col_ptrs_unscaled[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_eq+n_in+n_soc,
      n_tot,
      nnz,
      A_col_ptrs_unscaled.ptr(),
      nullptr,
      A_row_indices_unscaled.ptr(),
      A_values_unscaled.ptr(),
    };
  }
  /*!
   * Returns the original (unscaled) A matrix of the problem (mutable form).
   */
  auto A_mut_unscaled() -> proxsuite::linalg::sparse::MatMut<T, I>
  {
    auto n_tot = A_col_ptrs_unscaled.len() - 1;
    auto nnz = isize(proxsuite::linalg::sparse::util::zero_extend(
      A_col_ptrs_unscaled[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_eq+n_in+n_soc,
      n_tot,
      nnz,
      A_col_ptrs_unscaled.ptr_mut(),
      nullptr,
      A_row_indices_unscaled.ptr_mut(),
      A_values_unscaled.ptr_mut(),
    };
  }

  /*!
   * Returns the current (scaled) H matrix of the problem.
   */
  auto H() const -> proxsuite::linalg::sparse::MatRef<T, I>
  {
    auto n_tot = H_col_ptrs.len() - 1;
    auto nnz =
      isize(proxsuite::linalg::sparse::util::zero_extend(H_col_ptrs[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      nnz,
      H_col_ptrs.ptr(),
      nullptr,
      H_row_indices.ptr(),
      H_values.ptr(),
    };
  }
  /*!
   * Returns the current (scaled) H matrix of the problem (mutable form).
   */
  auto H_mut() -> proxsuite::linalg::sparse::MatMut<T, I>
  {
    auto n_tot = H_col_ptrs.len() - 1;
    auto nnz =
      isize(proxsuite::linalg::sparse::util::zero_extend(H_col_ptrs[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      nnz,
      H_col_ptrs.ptr_mut(),
      nullptr,
      H_row_indices.ptr_mut(),
      H_values.ptr_mut(),
    };
  }
  /*!
   * Returns the original (unscaled) H matrix of the problem.
   */
  auto H_unscaled() const -> proxsuite::linalg::sparse::MatRef<T, I>
  {
    auto n_tot = H_col_ptrs_unscaled.len() - 1;
    auto nnz = isize(proxsuite::linalg::sparse::util::zero_extend(
      H_col_ptrs_unscaled[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      nnz,
      H_col_ptrs_unscaled.ptr(),
      nullptr,
      H_row_indices_unscaled.ptr(),
      H_values_unscaled.ptr(),
    };
  }
  /*!
   * Returns the original (unscaled) H matrix of the problem (mutable form).
   */
  auto H_mut_unscaled() -> proxsuite::linalg::sparse::MatMut<T, I>
  {
    auto n_tot = H_col_ptrs_unscaled.len() - 1;
    auto nnz = isize(proxsuite::linalg::sparse::util::zero_extend(
      H_col_ptrs_unscaled[n_tot]));
    return {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      nnz,
      H_col_ptrs_unscaled.ptr_mut(),
      nullptr,
      H_row_indices_unscaled.ptr_mut(),
      H_values_unscaled.ptr_mut(),
    };
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
	static constexpr auto DYN = Eigen::Dynamic;
	using Vec = Eigen::Matrix<T, DYN, 1>;

    ///// SOLUTION STORAGE

    Vec x;
    Vec y;
    Vec z;
	isize n_soc;

	ProxSocpInfo<T> info;

	ProxSocpResults( isize dim=0,isize n_in=0, isize n_soc_ = 0,isize n_eq=0)
			: //
                x(dim),
				n_soc(n_soc_),
                y(n_in+n_soc_+n_eq),
                z(n_in+n_soc_)
                {
        
                x.setZero();
                y.setZero();
                z.setZero();

                info.rho = T(1e-10);
				if(n_soc==0){
					info.mu_eq = T(1e-10);
					info.mu_eq_inv = T(1e10);
				}else{
					info.mu_eq = T(1e-5);
					info.mu_eq_inv = T(1e5);
				}
						
	            info.mu_in = T(1e-6);
				info.mu_soc = T(1.e-4);

	            info.mu_in_inv = T(1e6);
				info.mu_soc_inv = T(1.e4);

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


		info.rho = T(1e-10);
		if(n_soc==0){
			info.mu_eq = T(1e-10);
			info.mu_eq_inv = T(1e10);
		}else{
			info.mu_eq = T(1e-5);
			info.mu_eq_inv = T(1e5);
		}
		info.mu_in = T(1e-6);
		info.mu_soc = T(1.e-4);
		
		info.mu_in_inv = T(1e6);
		info.mu_soc_inv = T(1.e4);

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
template <typename T, typename I>
struct ProxSocpWorkspace;

template <typename T, typename I>
void refactorize_socp(
		ProxSocpWorkspace<T,I>& work,
		ProxSocpResults<T> const& results,
		proxsuite::linalg::sparse::MatMut<T, I> kkt,
		proxsuite::linalg::veg::SliceMut<bool> active_constraints,
		SocpModel<T, I> const& data,
		proxsuite::linalg::veg::dynstack::DynStackMut stack,
		proxsuite::linalg::veg::Tag<T>& xtag) {
		isize n_tot = kkt.nrows();
	T mu_eq_neg = -results.info.mu_eq;
	T mu_in_neg = -results.info.mu_in;
	T mu_soc_neg = -results.info.mu_soc;
	
	if (work.internal.do_ldlt) {
		proxsuite::linalg::sparse::factorize_symbolic_non_zeros(
				work.internal.ldl.nnz_counts.ptr_mut(), 
				work.internal.ldl.etree.ptr_mut(),
				work.internal.ldl.perm_inv.ptr_mut(), 
				work.internal.ldl.perm.ptr_mut(), 
				kkt.symbolic(), 
				stack);
		
		auto _diag = stack.make_new_for_overwrite(xtag, n_tot);
		T* diag = _diag.ptr_mut();

		for (isize i = 0; i < data.dim; ++i) {
			diag[i] = results.info.rho;
		}
		for (isize i = 0; i < data.n_eq; ++i) {
			diag[data.dim + i] = mu_eq_neg;
		}
		for (isize i = 0; i < data.n_in; ++i) {
			diag[(data.dim + data.n_eq) + i] = mu_in_neg;
		}
		for (isize i = 0; i < data.n_soc; ++i) {
			diag[(data.dim + data.n_eq + data.n_in) + i] = mu_soc_neg;
		}

		proxsuite::linalg::sparse::factorize_numeric(
				work.internal.ldl.values.ptr_mut(),
				work.internal.ldl.row_indices.ptr_mut(),
				diag,
				work.internal.ldl.perm.ptr_mut(),
				work.internal.ldl.col_ptrs.ptr(),
				work.internal.ldl.etree.ptr_mut(),
				work.internal.ldl.perm_inv.ptr_mut(),
				kkt.as_const(),
				stack);
			/*
			isize ldl_nnz = 0;
			for (isize i = 0; i < n_tot; ++i) {
				ldl_nnz = util::checked_non_negative_plus(ldl_nnz, isize(ldl_nnz_counts[i]));
			}
			ldl._set_nnz(ldl_nnz);
			*/
	} else {
		*work.internal.matrix_free_kkt = {
				{kkt.as_const(),
		     active_constraints.as_const(),
		     data.dim,
		     data.n_eq,
		     data.n_in,
			 data.n_soc,
		     results.info.rho,
		     T(results.info.mu_eq_inv),// faire un test pour voir si c'est vraiment l'inverse!!!
		     T(results.info.mu_in_inv),
			 T(results.info.mu_soc_inv)}};
		(*work.internal.matrix_free_solver).compute(*work.internal.matrix_free_kkt);
	}
}
/*
template <typename T, typename I>
struct Ldlt {
	proxsuite::linalg::veg::Vec<I> etree;
	proxsuite::linalg::veg::Vec<I> perm;
	proxsuite::linalg::veg::Vec<I> perm_inv;
	proxsuite::linalg::veg::Vec<I> col_ptrs;
	proxsuite::linalg::veg::Vec<I> nnz_counts;
	proxsuite::linalg::veg::Vec<I> row_indices;
	proxsuite::linalg::veg::Vec<T> values;
};
*/


template <typename T, typename I>
struct ProxSocpWorkspace {

	struct /* NOLINT */ {
		// temporary allocations
		proxsuite::linalg::veg::Vec<proxsuite::linalg::veg::mem::byte> storage;// memory of the stack with the requirements req which determines its size.
		//Ldlt<T,I> ldl;
		//bool do_ldlt;
		//bool do_symbolic_fact;

		// persistent allocations

		Eigen::Matrix<T, Eigen::Dynamic, 1> g_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> l_scaled;
		Eigen::Matrix<T, Eigen::Dynamic, 1> u_scaled;
		//proxsuite::linalg::veg::Vec<I> kkt_nnz_counts;

		// stored in unique_ptr because we need a stable address
		std::unique_ptr<detail::AugmentedKktSocp<T, I>> matrix_free_kkt; // view on active part of the KKT which includes the regularizations 
		std::unique_ptr<Eigen::MINRES<
				detail::AugmentedKktSocp<T, I>,
				Eigen::Upper | Eigen::Lower,
				Eigen::IdentityPreconditioner>>
				matrix_free_solver; //eigen based method which takes in entry vector, and performs matrix vector products

		auto stack_mut() -> proxsuite::linalg::veg::dynstack::DynStackMut {
			return {
					proxsuite::linalg::veg::from_slice_mut,
					storage.as_mut(),
			}; 
		}// exploits all available memory in storage 

		// Whether the workspace is dirty
		bool dirty;
		bool proximal_parameter_update;

	} internal;
	//LinSysSolver *linsys_solver;
	qdldl_solver *linsys_solver;
	c_float *rho_vec;  
	//c_float *rho_vec_inv;  
	c_float *xz_tilde;
	//isize lnnz;
	/*!
	 * Constructor using the symbolic factorization.
	 * @param results solver's results.
	 * @param data solver's SocpModel.
	 * @param settings solver's settings.
	 * @param precond_req storage requirements for the solver's preconditioner.
	 * @param H symbolic structure of the quadratic cost input defining the SocpModel.
	 * @param A symbolic structure of the equality constraint matrix input defining the SocpModel.
	 */
	//void setup_symbolic_factorizaton(
	//		ProxSocpResults<T>& results,
	//		SocpModel<T, I>& data,
	//		ProxSocpSettings<T>& settings,
	//		proxsuite::linalg::veg::dynstack::StackReq precond_req,
	//		proxsuite::linalg::sparse::SymbolicMatRef<I> H,
	//		proxsuite::linalg::sparse::SymbolicMatRef<I> AT
	//		){
	//	auto& ldl = internal.ldl;
	//	
	//	auto& storage = internal.storage ;
	//	auto& do_ldlt = internal.do_ldlt;
	//	// persistent allocations
	//	data.dim = H.nrows();
	//	data.n_eq = AT.ncols();
	//	data.H_nnz = H.nnz();
	//	data.A_nnz = AT.nnz();
	//	using namespace proxsuite::linalg::veg::dynstack;
	//	using namespace proxsuite::linalg::sparse::util;
	//	proxsuite::linalg::veg::Tag<I> itag; 
	//	proxsuite::linalg::veg::Tag<T> xtag; 
	//	isize n = H.nrows();
	//	isize m = AT.ncols();
	//	isize n_tot = n + m ;
	//	isize nnz_tot = H.nnz() + AT.nnz();
	//	// form the full kkt matrix
	//	// assuming H, AT are sorted
	//	// and H is upper triangular
	//	{
	//		data.kkt_col_ptrs.resize_for_overwrite(n_tot + 1); // 
	//		data.kkt_row_indices.resize_for_overwrite(nnz_tot);
	//		data.kkt_values.resize_for_overwrite(nnz_tot);
	//		I* kktp = data.kkt_col_ptrs.ptr_mut();
	//		I* kkti = data.kkt_row_indices.ptr_mut();
	//		kktp[0] = 0;
	//		usize col = 0;
	//		usize pos = 0;
	//		auto insert_submatrix = [&](proxsuite::linalg::sparse::SymbolicMatRef<I> m,
	//									bool assert_sym_hi) -> void {
	//			I const* mi = m.row_indices();
	//			isize ncols = m.ncols();
	//			for (usize j = 0; j < usize(ncols); ++j) {
	//				usize col_start = m.col_start(j);
	//				usize col_end = m.col_end(j);
	//				kktp[col + 1] =
	//						checked_non_negative_plus(kktp[col], I(col_end - col_start));
	//				++col;
	//				for (usize p = col_start; p < col_end; ++p) {
	//					usize i = zero_extend(mi[p]);
	//					if (assert_sym_hi) {
	//						VEG_ASSERT(i <= j);
	//					}
	//					kkti[pos] = proxsuite::linalg::veg::nb::narrow<I>{}(i);
	//					++pos;
	//				}
	//			}
	//		};
	//		insert_submatrix(H, true);
	//		insert_submatrix(AT, false);
	//	}
	//	data.kkt_col_ptrs_unscaled = data.kkt_col_ptrs;
	//	data.kkt_row_indices_unscaled = data.kkt_row_indices;
	//	storage.resize_for_overwrite( //
	//			(StackReq::with_len(itag, n_tot) &
	//			proxsuite::linalg::sparse::factorize_symbolic_req( //
	//						itag,                                     //
	//						n_tot,                                    //
	//						nnz_tot,                                  //
	//						proxsuite::linalg::sparse::Ordering::amd))     //
	//					.alloc_req()                               //
	//	); 
	//	ldl.col_ptrs.resize_for_overwrite(n_tot + 1);
	//	ldl.perm_inv.resize_for_overwrite(n_tot);
	//	DynStackMut stack = stack_mut();
	//	bool overflow = false;
	//	{
	//		ldl.etree.resize_for_overwrite(n_tot);
	//		auto etree_ptr = ldl.etree.ptr_mut();
	//		using namespace proxsuite::linalg::veg::literals;
	//		auto kkt_sym = proxsuite::linalg::sparse::SymbolicMatRef<I>{
	//				proxsuite::linalg::sparse::from_raw_parts,
	//				n_tot,
	//				n_tot,
	//				nnz_tot,
	//				data.kkt_col_ptrs.ptr(),
	//				nullptr,
	//				data.kkt_row_indices.ptr(),
	//		};
	//		proxsuite::linalg::sparse::factorize_symbolic_non_zeros( //
	//				ldl.col_ptrs.ptr_mut() + 1,// reimplements col counts to get the matrix free version as well
	//				etree_ptr,
	//				ldl.perm_inv.ptr_mut(),
	//				static_cast<I const*>(nullptr),
	//				kkt_sym,
	//				stack);
	//		
	//		auto pcol_ptrs = ldl.col_ptrs.ptr_mut();
	//		pcol_ptrs[0] = I(0);
	//		using proxsuite::linalg::veg::u64;
	//		u64 acc = 0;
	//		for (usize i = 0; i < usize(n_tot); ++i) {
	//			acc += u64(zero_extend(pcol_ptrs[i + 1]));
	//			if (acc != u64(I(acc))) {
	//				overflow = true;
	//			}
	//			pcol_ptrs[(i + 1)] = I(acc);
	//		}
	//	}
	//	lnnz = isize(zero_extend(ldl.col_ptrs[n_tot]));
	//	// if ldlt is too sparse
	//	// do_ldlt = !overflow && lnnz < (10000000);
	//	do_ldlt = !overflow && lnnz < 10000000;
	//	
	//	internal.do_symbolic_fact = false;
	//}
	/*!
	 * Constructor.
	 * @param qp view on the qp problem.
	 * @param results solver's results.
	 * @param data solver's SocpModel.
	 * @param settings solver's settings.
	 * @param execute_or_not boolean option for execturing or not the preconditioner for scaling the problem (and reduce its ill conditioning).
	 * @param precond preconditioner chosen for the solver.
	 * @param precond_req storage requirements for the solver's preconditioner.
	 */
	template <typename P>
	void setup_impl(
			const SocpView<T, I> socp,
			SocpModel<T, I>& data,
			ProxSocpResults<T>& results,
			const ProxSocpSettings<T>& settings,
			bool execute_or_not,
			P& precond,
			proxsuite::linalg::veg::dynstack::StackReq precond_req) {
		
		//auto& ldl = internal.ldl;
		//
		auto& storage = internal.storage ;
		//auto& do_ldlt = internal.do_ldlt;
		// persistent allocations

		auto& g_scaled = internal.g_scaled;
		auto& l_scaled = internal.l_scaled;
		auto& u_scaled = internal.u_scaled;
		//auto& kkt_nnz_counts = internal.kkt_nnz_counts;

		// stored in unique_ptr because we need a stable address
		auto& matrix_free_solver = internal.matrix_free_solver;
		auto& matrix_free_kkt = internal.matrix_free_kkt;

		data.dim = socp.H.nrows();
		isize m = socp.AT.nrows();//socp.AT.ncols();
		data.H_nnz = socp.H.nnz();
		data.A_nnz = socp.AT.nnz();

		data.g = socp.g.to_eigen();
		data.l = socp.l.to_eigen();
		data.u = socp.u.to_eigen();
		

		using namespace proxsuite::linalg::veg::dynstack;
		using namespace proxsuite::linalg::sparse::util;

		using SR = StackReq;
		proxsuite::linalg::veg::Tag<I> itag;
		proxsuite::linalg::veg::Tag<T> xtag;

		isize n = socp.H.nrows();
		isize n_tot = n + m;
		for (isize i = 0; i < m; i++) {
			if (i < data.n_eq){
				rho_vec[i]     = results.info.mu_eq_inv;
				//std::cout << "i " << rho_vec[i] << std::endl;
				//rho_vec_inv[i] = results.info.mu_eq;
			} else if (i >= data.n_eq && i < data.n_eq + data.n_in){
				rho_vec[i]     = results.info.mu_in_inv;
				//std::cout << "i " << rho_vec[i] << std::endl;
				//rho_vec_inv[i] = results.info.mu_in;
			} else {
				rho_vec[i]     = results.info.mu_soc_inv;
				//std::cout << "i " << rho_vec[i] << std::endl;
				//rho_vec_inv[i] = results.info.mu_soc;
			}
  		}
		//isize nnz_tot = socp.H.nnz() + socp.AT.nnz();
		if (!internal.dirty){// internal.do_symbolic_fact && 
		// form the full kkt matrix
		// assuming H, AT, CT are sorted
		// and H is upper triangular

		//{
		//	data.kkt_col_ptrs.resize_for_overwrite(n_tot + 1);
		//	data.kkt_row_indices.resize_for_overwrite(nnz_tot);
		//	data.kkt_values.resize_for_overwrite(nnz_tot);
		//	I* kktp = data.kkt_col_ptrs.ptr_mut();
		//	I* kkti = data.kkt_row_indices.ptr_mut();
		//	T* kktx = data.kkt_values.ptr_mut();
		//	kktp[0] = 0;
		//	usize col = 0;
		//	usize pos = 0;
		//	auto insert_submatrix = [&](proxsuite::linalg::sparse::MatRef<T, I> m,
		//								bool assert_sym_hi) -> void {
		//		I const* mi = m.row_indices();
		//		T const* mx = m.values();
		//		isize ncols = m.ncols();
		//		for (usize j = 0; j < usize(ncols); ++j) {
		//			usize col_start = m.col_start(j);
		//			usize col_end = m.col_end(j);
		//			kktp[col + 1] =
		//					checked_non_negative_plus(kktp[col], I(col_end - col_start));
		//			++col;
		//			for (usize p = col_start; p < col_end; ++p) {
		//				usize i = zero_extend(mi[p]);
		//				if (assert_sym_hi) {
		//					VEG_ASSERT(i <= j);
		//				}
		//				kkti[pos] = proxsuite::linalg::veg::nb::narrow<I>{}(i); 
		//				kktx[pos] = mx[p];
		//				++pos;
		//			}
		//		}
		//	};
		//	insert_submatrix(socp.H, true);
		//	insert_submatrix(socp.AT, false);
		//}
		//data.kkt_col_ptrs_unscaled = data.kkt_col_ptrs;
		//data.kkt_row_indices_unscaled = data.kkt_row_indices;
		//data.kkt_values_unscaled = data.kkt_values;
		{
			data.H_col_ptrs.resize_for_overwrite(n + 1);
			data.H_row_indices.resize_for_overwrite(socp.H.nnz());
			data.H_values.resize_for_overwrite(socp.H.nnz());
			I* kktp = data.H_col_ptrs.ptr_mut();
			I* kkti = data.H_row_indices.ptr_mut();
			T* kktx = data.H_values.ptr_mut();
			kktp[0] = 0;
			usize col = 0;
			usize pos = 0;
			auto insert_submatrix = [&](proxsuite::linalg::sparse::MatRef<T, I> m,
										bool assert_sym_hi) -> void {
				I const* mi = m.row_indices();
				T const* mx = m.values();
				isize ncols = m.ncols();
				for (usize j = 0; j < usize(ncols); ++j) {
					usize col_start = m.col_start(j);
					usize col_end = m.col_end(j);
					kktp[col + 1] =
							checked_non_negative_plus(kktp[col], I(col_end - col_start));
					++col;
					for (usize p = col_start; p < col_end; ++p) {
						usize i = zero_extend(mi[p]);
						if (assert_sym_hi) {
							VEG_ASSERT(i <= j);
						}
						kkti[pos] = proxsuite::linalg::veg::nb::narrow<I>{}(i); 
						kktx[pos] = mx[p];
						++pos;
					}
				}
			};
			insert_submatrix(socp.H, true);
		}
		data.H_col_ptrs_unscaled = data.H_col_ptrs;
		data.H_row_indices_unscaled = data.H_row_indices;
		data.H_values_unscaled = data.H_values;
		{
			data.A_col_ptrs.resize_for_overwrite(n + 1);
			data.A_row_indices.resize_for_overwrite(socp.AT.nnz());
			data.A_values.resize_for_overwrite(socp.AT.nnz());
			I* kktp = data.A_col_ptrs.ptr_mut();
			I* kkti = data.A_row_indices.ptr_mut();
			T* kktx = data.A_values.ptr_mut();
			kktp[0] = 0;
			usize col = 0;
			usize pos = 0;
			auto insert_submatrix = [&](proxsuite::linalg::sparse::MatRef<T, I> m,
										bool assert_sym_hi) -> void {
				I const* mi = m.row_indices();
				T const* mx = m.values();
				isize ncols = m.ncols();
				for (usize j = 0; j < usize(ncols); ++j) {
					usize col_start = m.col_start(j);
					usize col_end = m.col_end(j);
					kktp[col + 1] =
							checked_non_negative_plus(kktp[col], I(col_end - col_start));
					++col;
					for (usize p = col_start; p < col_end; ++p) {
						usize i = zero_extend(mi[p]);
						if (assert_sym_hi) {
							VEG_ASSERT(i <= j);
						}
						kkti[pos] = proxsuite::linalg::veg::nb::narrow<I>{}(i); 
						kktx[pos] = mx[p];
						++pos;
					}
				}
			};
			insert_submatrix(socp.AT, false);
		}
		data.A_col_ptrs_unscaled = data.A_col_ptrs;
		data.A_row_indices_unscaled = data.A_row_indices;
		data.A_values_unscaled = data.A_values;

		//storage.resize_for_overwrite( //
		//		(StackReq::with_len(itag, n_tot) &
		//		proxsuite::linalg::sparse::factorize_symbolic_req( //
		//					itag,                                     //
		//					n_tot,                                    //
		//					nnz_tot,                                  //
		//					proxsuite::linalg::sparse::Ordering::amd))     //
		//				.alloc_req()                               //
		//);
		//ldl.col_ptrs.resize_for_overwrite(n_tot + 1);
		//ldl.perm_inv.resize_for_overwrite(n_tot);

		//DynStackMut stack = stack_mut();

		//bool overflow = false;
		//{
		//	ldl.etree.resize_for_overwrite(n_tot);
		//	auto etree_ptr = ldl.etree.ptr_mut();
		//	using namespace proxsuite::linalg::veg::literals;
		//	auto kkt_sym = proxsuite::linalg::sparse::SymbolicMatRef<I>{
		//			proxsuite::linalg::sparse::from_raw_parts,
		//			n_tot,
		//			n_tot,
		//			nnz_tot,
		//			data.kkt_col_ptrs.ptr(),
		//			nullptr,
		//			data.kkt_row_indices.ptr(),
		//	};
		//	proxsuite::linalg::sparse::factorize_symbolic_non_zeros( //
		//			ldl.col_ptrs.ptr_mut() + 1,
		//			etree_ptr,
		//			ldl.perm_inv.ptr_mut(),
		//			static_cast<I const*>(nullptr),
		//			kkt_sym,
		//			stack);
		//	auto pcol_ptrs = ldl.col_ptrs.ptr_mut(); 
		//	pcol_ptrs[0] = I(0);//pcol_ptrs +1: pointor towards the nbr of non zero elts per column of the ldlt 
		//	// we need to compute its cumulative sum below to determine if there could be an overflow
		//	using proxsuite::linalg::veg::u64;
		//	u64 acc = 0;
		//	for (usize i = 0; i < usize(n_tot); ++i) {
		//		acc += u64(zero_extend(pcol_ptrs[i + 1]));
		//		if (acc != u64(I(acc))) {
		//			overflow = true;
		//		}
		//		pcol_ptrs[(i + 1)] = I(acc);
		//	}
		//}

		//auto lnnz = isize(zero_extend(ldl.col_ptrs[n_tot]));

		// if ldlt is too sparse
		// do_ldlt = !overflow && lnnz < (10000000);
		//do_ldlt = !overflow && lnnz < 10000000;
	//}
	//else{
	//		T* kktx = data.kkt_values.ptr_mut();
	//		usize pos = 0;	
	//		auto insert_submatrix = [&](proxsuite::linalg::sparse::MatRef<T, I> m) -> void {
	//			T const* mx = m.values();
	//			isize ncols = m.ncols();
	//			for (usize j = 0; j < usize(ncols); ++j) {
	//				usize col_start = m.col_start(j);
	//				usize col_end = m.col_end(j);
	//				for (usize p = col_start; p < col_end; ++p) {
	//					kktx[pos] = mx[p];
	//					++pos;
	//				}
	//			}
	//		};
	//		insert_submatrix(socp.H);
	//		insert_submatrix(socp.AT);
	//		data.kkt_values_unscaled = data.kkt_values;
	//}
#define PROX_SOCP_ALL_OF(...)                                                    \
proxsuite::linalg::veg::dynstack::StackReq::and_(proxsuite::linalg::veg::init_list(__VA_ARGS__))
#define PROX_SOCP_ANY_OF(...)                                                    \
proxsuite::linalg::veg::dynstack::StackReq::or_(proxsuite::linalg::veg::init_list(__VA_ARGS__))
		//  ? --> if
		//auto refactorize_req = 
		//		do_ldlt
		//				? PROX_SOCP_ANY_OF({
		//							proxsuite::linalg::sparse::
		//									factorize_symbolic_req( // symbolic ldl
		//											itag,
		//											n_tot,
		//											nnz_tot,
		//											proxsuite::linalg::sparse::Ordering::user_provided),
		//							PROX_QP_ALL_OF({
		//									SR::with_len(xtag, n_tot), // diag
		//									proxsuite::linalg::sparse::
		//											factorize_numeric_req( // numeric ldl
		//													xtag,
		//													itag,
		//													n_tot,
		//													nnz_tot,
		//													proxsuite::linalg::sparse::Ordering::user_provided),
		//							}),
		//					})
		//				: PROX_SOCP_ALL_OF({
		//							SR::with_len(itag, 0), // compute necessary space for storing n elts of type I (n = 0 here)
		//							SR::with_len(xtag, 0), // compute necessary space for storing n elts of type T (n = 0 here)
		//					});
		auto x_vec = [&](isize n) noexcept -> StackReq {
			return proxsuite::linalg::dense::temp_vec_req(xtag, n);
		};

		//auto ldl_solve_in_place_req = PROX_QP_ALL_OF({
		//		x_vec(n_tot), // tmp
		//		x_vec(n_tot), // err
		//		x_vec(n_tot), // work
		//});
		auto unscaled_primal_dual_residual_req = x_vec(n); // Hx
		//auto line_search_req = PROX_QP_ALL_OF({
		//		x_vec(2 * n_in), // alphas
		//		x_vec(n),        // Cdx_active
		//		x_vec(n_in),     // active_part_z
		//		x_vec(n_in),     // tmp_lo
		//		x_vec(n_in),     // tmp_up
		//});
		// define memory needed for primal_dual_newton_semi_smooth
		//PROX_QP_ALL_OF --> need to store all argument inside
		//PROX_QP_ANY_OF --> au moins un de  ceux en entrée
		//auto primal_dual_newton_semi_smooth_req = PROX_QP_ALL_OF({
		//		x_vec(n_tot), // dw
		//		PROX_SOCP_ANY_OF({
		//				ldl_solve_in_place_req,
		//				//PROX_QP_ALL_OF({
		//				//		SR::with_len(veg::Tag<bool>{}, n_in), // active_set_lo
		//				//		SR::with_len(veg::Tag<bool>{}, n_in), // active_set_up
		//				//		SR::with_len(
		//				//				veg::Tag<bool>{}, n_in), // new_active_constraints
		//				//		(do_ldlt && n_in > 0)
		//				//				? PROX_QP_ANY_OF({
		//				//							proxsuite::linalg::sparse::add_row_req(
		//				//									xtag, itag, n_tot, false, n, n_tot),
		//				//							proxsuite::linalg::sparse::delete_row_req(
		//				//									xtag, itag, n_tot, n_tot),
		//				//					})
		//				//				: refactorize_req,
		//				//}),
		//				//PROX_QP_ALL_OF({
		//				//		x_vec(n),    // Hdx
		//				//		x_vec(m), // Adx
		//				//		x_vec(n_in), // Cdx
		//				//		x_vec(n),    // ATdy
		//				//		x_vec(n),    // CTdz
		//				//}),
		//		}),
		//		//line_search_req, 
		//});

		auto iter_req = PROX_SOCP_ANY_OF({
				PROX_SOCP_ALL_OF(
						{
					x_vec(m), // primal_residual
					//x_vec(n_eq), // primal_residual_eq_scaled
					x_vec(data.n_in), // primal_residual_in_scaled_lo
					//x_vec(n_in), // primal_residual_in_scaled_up
					//x_vec(n_in), // primal_residual_in_scaled_up
					x_vec(n),    // dual_residual_scaled
					PROX_QP_ANY_OF({
									unscaled_primal_dual_residual_req,
									PROX_QP_ALL_OF({
											x_vec(n),    // x_prev
											x_vec(m), // y_prev
											x_vec(data.n_in+data.n_soc), // z_hat
											x_vec(n+m),
											x_vec(data.n_in+data.n_soc), // tmp
											//primal_dual_newton_semi_smooth_req,
									}),
							})}),
				//refactorize_req, // mu_update
		});
		//do_ldlt = false;
		auto req = //
				PROX_SOCP_ALL_OF({
						x_vec(n),                             // g_scaled
						//x_vec(n_eq),                          // b_scaled
						x_vec(data.n_in),                          // l_scaled
						x_vec(m),                          // u_scaled
						//SR::with_len(veg::Tag<bool>{}, n_in), // active constr
						SR::with_len(itag, n_tot),            // kkt nnz counts
						//refactorize_req,
						PROX_SOCP_ANY_OF({
								precond_req,
								PROX_SOCP_ALL_OF({
										//do_ldlt ? PROX_SOCP_ALL_OF({
										//							SR::with_len(itag, n_tot), // perm
										//							SR::with_len(itag, n_tot), // etree
										//							SR::with_len(itag, n_tot), // ldl nnz counts
										//							SR::with_len(itag, lnnz), // ldl row indices
										//							SR::with_len(xtag, lnnz), // ldl values
										//					})
														PROX_SOCP_ALL_OF({
																	SR::with_len(itag, 0),
																	SR::with_len(xtag, 0),
															}),
										iter_req,
								}),
						}),
				});

		storage.resize_for_overwrite(req.alloc_req()); // defines the maximal storage size 
		// storage.resize(n): if it is done twice in a row, the second times it does nothing, as the same resize has been asked
		// preconditioner
		//proxsuite::linalg::sparse::MatMut<T, I>  kkt = data.kkt_mut();
		//auto kkt_top_n_rows = detail::top_rows_mut_unchecked(proxsuite::linalg::veg::unsafe, kkt, n); //  top_rows_mut_unchecked: take a view of sparse matrix for n first lines ; the function assumes all others lines are zeros;
	}
		data.H_col_ptrs = data.H_col_ptrs_unscaled;
		data.H_row_indices = data.H_row_indices_unscaled;
		data.H_values = data.H_values_unscaled;
		data.A_col_ptrs = data.A_col_ptrs_unscaled;
		data.A_row_indices = data.A_row_indices_unscaled;
		data.A_values = data.A_values_unscaled;

		proxsuite::linalg::sparse::MatMut<T, I>  H_scaled = data.H_mut();
		proxsuite::linalg::sparse::MatMut<T, I>  A_scaled = data.A_mut();
		/*
			H AT CT
			A 
			C

			here we store the upper triangular part below

			tirSup(H) AT CT
			0 0 0 
			0 0 0 

			veg::unsafe:  precises that the function has undefined behavior if upper condition is not respected.
		*/

		//proxsuite::linalg::sparse::MatMut<T, I> H_scaled =
		//		detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);
		//proxsuite::linalg::sparse::MatMut<T, I> AT_scaled =
		//		detail::middle_cols_mut(kkt_top_n_rows, n, m, data.A_nnz);
		//std::cout << " AT_scaled " << AT_scaled.to_eigen() <<std::endl;
		g_scaled = data.g;
		l_scaled = data.l;
		u_scaled = data.u;

		SocpViewMut<T, I> socp_scaled = {
				H_scaled,
				{proxsuite::linalg::sparse::from_eigen, g_scaled},
				A_scaled,
				{proxsuite::linalg::sparse::from_eigen, l_scaled},
				{proxsuite::linalg::sparse::from_eigen, u_scaled},
		};
		

		//SparseMat<T, I>  A_scaled = socp_scaled.AT.to_eigen().transpose();
		//T p = 1;
		//auto H_scaled = proxqp::utils::rand::sparse_positive_definite_rand(n, T(10.0), p);
		//auto A_scaled = proxqp::utils::rand::sparse_matrix_rand<T>(m, n, p);
		//SocpViewMut<T, I> socp_view= {
		//		socp_scaled.H,
		//		//{proxsuite::linalg::sparse::from_eigen, H_scaled},
		//		 socp_scaled.g,
		//		 {proxsuite::linalg::sparse::from_eigen, A_scaled},
		//		socp_scaled.l,
		//		socp_scaled.u,
		//};
		DynStackMut stack = stack_mut();
		precond.scale_socp_in_place(socp_scaled, execute_or_not, settings.preconditioner_max_iter, settings.preconditioner_accuracy, stack); // TODO: to debug col start

		//c_int err = load_linsys_solver(linsys_solver_type::QDLDL_SOLVER);
		//c_int m_H = static_cast<c_int>(H_scaled.nrows());
		//c_int n_H = static_cast<c_int>(H_scaled.ncols());
		//c_int nnz_H = static_cast<c_int>(H_scaled.nnz());
		//const csc * H = csc_spalloc(m_H,n_H,nnz_H,c_int(1),c_int(0));
		//H->p = reinterpret_cast<const c_int*>(H_scaled.col_ptrs());
		//H->i = reinterpret_cast<const c_int*>(H_scaled.row_indices());
		//H->x = reinterpret_cast<const c_float*>(H_scaled.values());

		const csc H = H_scaled.to_csc();
		// csc{static_cast<c_int>(_.nnz), static_cast<c_int>(_.nrows), static_cast<c_int>(_.ncols), reinterpret_cast<c_int*>(_.col), reinterpret_cast<c_int*>(_.row), reinterpret_cast<c_float*>(_.val), static_cast<c_int>(-1) };
		//c_int m_A = static_cast<c_int>(A_scaled.nrows());
		//c_int n_A = static_cast<c_int>(A_scaled.ncols());
		//c_int nnz_A = static_cast<c_int>(A_scaled.nnz());
		//const csc * A = csc_spalloc(m_A,n_A,nnz_A,c_int(1),c_int(0));
		const csc A = A_scaled.to_csc();
		//A->p = reinterpret_cast<const c_int*>(A_scaled.col_ptrs());
		//A->i = reinterpret_cast<const c_int*>(A_scaled.row_indices());
		//A->x = reinterpret_cast<const c_float*>(A_scaled.values());
		//A = &socp_scaled.AT.to_csc();

		//c_float *rho_vec_temp ;
		//rho_vec_temp = static_cast<c_float*>(c_malloc(m * sizeof(c_float)));
		//for (isize i = 0; i < m; i++) {
		//	rho_vec_temp[i]     = 1;
		//}
		//auto exitflag = init_linsys_solver(&(linsys_solver),&H,&A,//&(linsys_solver), &H, &A,
        //                        c_float(results.info.rho), rho_vec,
        //                 

		auto exitflag = init_linsys_solver_qdldl(&(linsys_solver),&H,&A,
                                c_float(results.info.rho), rho_vec,
                                 c_int(0));
		//kkt_nnz_counts.resize_for_overwrite(n_tot);
		/*
		proxsuite::linalg::sparse::MatMut<T, I> kkt_active = {
				proxsuite::linalg::sparse::from_raw_parts,
				n_tot,
				n_tot,
				data.H_nnz + data.A_nnz,// these variables are not used for the matrix vector product in augmented KKT with Min res algorithm (to be exact, it should depend of the initial guess)
				kkt.col_ptrs_mut(),
				kkt_nnz_counts.ptr_mut(),
				kkt.row_indices_mut(),
				kkt.values_mut(),
		};
		*/

		//using MatrixFreeSolver = Eigen::MINRES<
		//		detail::AugmentedKktSocp<T, I>,
		//		Eigen::Upper | Eigen::Lower,
		//		Eigen::IdentityPreconditioner>;
		//matrix_free_solver = std::unique_ptr<MatrixFreeSolver>{
		//		new MatrixFreeSolver,
		//};
		//matrix_free_kkt = std::unique_ptr<detail::AugmentedKktSocp<T, I>>{
		//		new detail::AugmentedKktSocp<T, I>{
		//				{
		//						kkt.as_const(),
		//						{},
		//						n,
		//						data.n_eq,
		//						data.n_in,
		//						data.n_soc,
		//						{},
		//						{},
		//						{},
		//						{},
		//				},
		//		}
		//};

		//auto zx = proxsuite::linalg::sparse::util::zero_extend;// ?
		//auto max_lnnz = isize(zx(ldl.col_ptrs[n_tot]));
		//isize ldlt_ntot = do_ldlt ? n_tot : 0;
		//isize ldlt_lnnz = do_ldlt ? max_lnnz : 0;
		//ldl.nnz_counts.resize_for_overwrite(ldlt_ntot);
		//ldl.row_indices.resize_for_overwrite(ldlt_lnnz);
		//ldl.values.resize_for_overwrite(ldlt_lnnz);
		//
		//ldl.perm.resize_for_overwrite(ldlt_ntot);
		//if (do_ldlt) {
		//	// compute perm from perm_inv
		//	for (isize i = 0; i < n_tot; ++i) {
		//		ldl.perm[isize(zx(ldl.perm_inv[i]))] = I(i);
		//	}
		//}

		internal.dirty = false;
	}
	Timer<T> timer;
	ProxSocpWorkspace() = default;

	auto ldl_col_ptrs() const -> I const* {
		return internal.ldl.col_ptrs.ptr();
	}
	auto ldl_col_ptrs_mut() -> I* {
		return internal.ldl.col_ptrs.ptr_mut();
	}
	auto stack_mut() -> proxsuite::linalg::veg::dynstack::DynStackMut {
		return internal.stack_mut();
	}

	void set_dirty() { internal.dirty = true; }
};

} //namespace sparse
} //namespace proxqp
} //namespace proxsuite

#endif /* end of include guard PROXSUITE_QP_SPARSE_ProxSocpWorkspace_HPP */
