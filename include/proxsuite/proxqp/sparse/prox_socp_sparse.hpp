//
// Copyright (c) 2022, INRIA
//
/** \file */

#ifndef PROXSUITE_PROXQP_SPARSE_SOLVER_PROX_SOCP_HPP
#define PROXSUITE_PROXQP_SPARSE_SOLVER_PROX_SOCP_HPP

#include <proxsuite/linalg/dense/core.hpp>
#include <proxsuite/linalg/sparse/core.hpp>
#include <proxsuite/linalg/sparse/factorize.hpp>
#include <proxsuite/linalg/sparse/update.hpp>
#include <proxsuite/linalg/sparse/rowmod.hpp>
#include <proxsuite/linalg/veg/vec.hpp>

#include <proxsuite/proxqp/dense/views.hpp>
#include <proxsuite/proxqp/settings.hpp>
#include <proxsuite/proxqp/results.hpp>
#include <proxsuite/proxqp/sparse/views.hpp>
#include <proxsuite/proxqp/sparse/model.hpp>
#include <proxsuite/proxqp/sparse/workspace_prox_socp.hpp>
#include <proxsuite/proxqp/sparse/utils.hpp>
#include <proxsuite/proxqp/sparse/preconditioner/ruiz_socp.hpp>
#include <proxsuite/proxqp/sparse/preconditioner/identity.hpp>


#include <iostream>
#include <iomanip> 
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

namespace proxsuite {
namespace proxqp {
namespace sparse {
using proxsuite::linalg::veg::isize;
using proxsuite::linalg::veg::usize;
using proxsuite::linalg::veg::i64;
using dense::infty_norm;


isize roundUp(const isize numToRound, isize multiple){
    // x + .5 * N-c_fmod(x + .5 * N, N) //https://osqp.dpldocs.info/source/glob_opts.d.html#L126
    isize tmp = isize(numToRound + isize(0.5 * multiple));
    isize remainder = tmp % multiple;
    return tmp - remainder;
}

template<typename T,typename I>
void print_setup_header_socp(const ProxSocpSettings<T>& settings,ProxSocpResults<T>& results, const SocpModel<T,I>& model){

  print_line();
  std::cout  <<"                      ProxSOCP  -  A Proximal SOCP Solver\n"
             <<"          (c) Antoine Bambade, Adrien Taylor, Justin Carpentier\n"
             <<"                           Inria Paris 2022        \n"
          << std::endl;
  print_line();

  // Print variables and constraints
  std::cout << "problem:  " << std::noshowpos <<std::endl;
  std::cout << "          variables n = " << model.dim << ", equality constraints n_eq = " << model.n_eq <<  ",\n" <<
  "          linear cone constraints n_in = "<< model.n_in << ", second order cone constraints n_soc = " << model.n_soc << ", nnz = " << model.H_nnz + model.A_nnz <<  ",\n" << std::endl;

  // Print Settings
  std::cout << "settings: " << std::endl;
  std::cout  <<"          backend = sparse," << std::endl;
  std::cout  <<"          eps_abs = " << settings.eps_abs <<" eps_rel = " << settings.eps_rel << std::endl;
  std::cout  <<"          eps_prim_inf = " <<settings.eps_primal_inf <<", eps_dual_inf = " << settings.eps_dual_inf << "," << std::endl;

  std::cout  <<"          rho = " <<results.info.rho <<", mu_eq = " << results.info.mu_eq << ", mu_in = " << results.info.mu_in << "," << std::endl;
  std::cout  <<"          mu_soc = " <<results.info.mu_soc << ", max_iter = " << settings.max_iter << "," << std::endl;

  if (settings.compute_preconditioner) {
    std::cout  <<"          scaling: on, " << std::endl;
  } else {
    std::cout  <<"          scaling: off, " << std::endl;
  }
  if (settings.compute_timings) {
    std::cout  <<"          timings: on, " << std::endl;
  } else {
    std::cout  <<"          timings: off, " << std::endl;
  }
  switch (settings.initial_guess)
  {
  case InitialGuessStatus::WARM_START:
	std::cout << "          initial guess: warm start. \n"<< std::endl;
	break;
  case InitialGuessStatus::NO_INITIAL_GUESS:
	std::cout << "          initial guess: initial guess. \n"<< std::endl;
	break;
  case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT:
	std::cout << "          initial guess: warm start with previous result. \n"<< std::endl;
	break;
  case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT:
	std::cout << "          initial guess: cold start with previous result. \n"<< std::endl;
	break;
  case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS:
	std::cout << "          initial guess: equality constrained initial guess. \n"<< std::endl;
  }
}

template <typename T>
using VecMapMut = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1>,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;
template <typename T>
using VecMap = Eigen::Map<
		Eigen::Matrix<T, Eigen::Dynamic, 1> const,
		Eigen::Unaligned,
		Eigen::Stride<Eigen::Dynamic, 1>>;

template <typename T, typename I, typename P>
auto primal_dual_socp_residual(
		VecMapMut<T> primal_residual,
		VecMapMut<T> primal_residual_in_scaled,
		VecMapMut<T> dual_residual_scaled,
		T& primal_feasibility_rhs_0,
		T& dual_feasibility_rhs_0,
		T& dual_feasibility_rhs_1,
		P& precond,
		SocpModel<T, I>& data,
		SocpView<T, I> socp_scaled,
		VecMap<T> x_e,
		VecMap<T> y_e,
		//VectorViewMut<T> x_e,
		//VectorViewMut<T> y_e,
		T& primal_feasibility_eq_lhs,
		T& primal_feasibility_in_lhs,
		T& pri_cone,
		proxsuite::linalg::veg::dynstack::DynStackMut stack) -> proxsuite::linalg::veg::Tuple<T, T> {


    // dual

	isize n = data.dim;
    isize n_eq = data.n_eq;
    isize n_in = data.n_in;
	primal_residual.setZero();
	primal_residual_in_scaled.setZero();
	
	dual_residual_scaled = socp_scaled.g.to_eigen();
	{
		LDLT_TEMP_VEC_UNINIT(T, tmp, n, stack);
		tmp.setZero();
		
		detail::noalias_symhiv_add(tmp, socp_scaled.H.to_eigen(), x_e);
		dual_residual_scaled += tmp; // contains now scaled(g+Hx)

		precond.unscale_dual_residual_in_place({proxqp::from_eigen, tmp});
		dual_feasibility_rhs_0 = infty_norm(tmp); // ||unscaled(Hx)||

		//auto ATy = tmp;
		tmp.setZero();
		
		//detail::noalias_gevmmv_add(
		//		primal_residual, ATy, socp_scaled.AT.to_eigen(), x_e, y_e);// primal_res = AT.T xe ; ATy : AT*y
		detail::noalias_gevmmv_add(
				tmp, primal_residual, socp_scaled.AT.to_eigen(), y_e, x_e);// ATy = A.T ye ; prim_res = A*x
		dual_residual_scaled += tmp; // contains now scaled(g+Hx+ATy)

		precond.unscale_dual_residual_in_place({proxqp::from_eigen, tmp});
		dual_feasibility_rhs_1 = infty_norm(tmp); // ||unscaled(ATy)||
	}
	/*// using unscaled data and then scale at the end
	proxsuite::linalg::sparse::MatMut<T, I>  H_ = data.H_mut_unscaled();
	proxsuite::linalg::sparse::MatMut<T, I>  A_ = data.A_mut_unscaled();
	dual_residual_scaled = data.g;
	{
		LDLT_TEMP_VEC_UNINIT(T, tmp, n, stack);
		tmp.setZero();
		
		precond.unscale_primal_in_place(x_e);
		precond.unscale_dual_in_place(y_e);
		detail::noalias_symhiv_add(tmp, H_.to_eigen(), detail::vec(x_e.to_eigen()));
		dual_feasibility_rhs_0 = infty_norm(tmp); // ||unscaled(Hx)||
		dual_residual_scaled += tmp; // contains now unscaled(g+Hx)
		tmp.setZero();
		detail::noalias_gevmmv_add(
				tmp, primal_residual, A_.to_eigen(), detail::vec(y_e.to_eigen()), detail::vec(x_e.to_eigen()));// ATy = A.T ye ; prim_res = A*x
		dual_feasibility_rhs_1 = infty_norm(tmp); // ||unscaled(ATy)||
		dual_residual_scaled += tmp; // contains now unscaled(g+Hx+ATy)
	}

	T dual_feasibility_lhs = infty_norm(dual_residual_scaled); // ||unscaled(Hx+g+ATy)||
	precond.scale_dual_residual_in_place({proxqp::from_eigen, dual_residual_scaled});// ||scaled(Hx+g+ATy)||
	*/

    // primal 
	auto l = data.l;
	auto u = data.u;

	precond.unscale_primal_residual_in_place({proxqp::from_eigen, primal_residual});
	primal_feasibility_rhs_0 = infty_norm(primal_residual); // ||unscaled(Ax)||

	precond.unscale_dual_residual_in_place({proxqp::from_eigen, dual_residual_scaled});// ||scaled(Hx+g+ATy)||
	T dual_feasibility_lhs = infty_norm(dual_residual_scaled); // ||unscaled(Hx+g+ATy)||
	precond.scale_dual_residual_in_place({proxqp::from_eigen, dual_residual_scaled});// ||scaled(Hx+g+ATy)||

	//std::cout << "Cx " << primal_residual << std::endl;
	primal_residual_in_scaled =
			detail::positive_part(primal_residual.segment(n_eq,n_in) - u.segment(n_eq,n_in)) +
			detail::negative_part(primal_residual.segment(n_eq,n_in) - l);
	primal_residual.head(n_eq) -= u.head(n_eq);

	primal_feasibility_eq_lhs = infty_norm(primal_residual.head(n_eq));
	primal_feasibility_in_lhs = infty_norm(primal_residual_in_scaled);
	
	T primal_feasibility_lhs =
			std::max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	pri_cone = 0;
    isize j = 0;
	isize n_cone = data.dims.rows();
	//std::cout << "aux : " << primal_residual.tail(data.n_soc) -  u.tail(data.n_soc) << std::endl;
	{
		LDLT_TEMP_VEC_UNINIT(T, aux, data.n_soc, stack);
		//std::cout << "primal_residual.tail(data.n_soc) " << primal_residual.tail(data.n_soc) << std::endl;
		//std::cout << "u.tail(data.n_soc) " << u.tail(data.n_soc) << std::endl;
		
		aux = primal_residual.tail(data.n_soc) - u.tail(data.n_soc);
		//std::cout << "aux " << aux << std::endl;
		for (isize it = 0; it < n_cone; ++it) { 
			isize dim_cone = data.dims[it];
			
			T cone_error = std::max(aux.segment(j+1,dim_cone-1).norm() - aux[j],0.);
			//std::cout << "it " << it << " dim cone " << dim_cone << " cone error " << cone_error << std::endl;
			pri_cone = std::max(pri_cone, cone_error);
			j+=dim_cone;
		}
	}
	//std::cout << "primal eq " << primal_feasibility_eq_lhs << " primal in " << primal_feasibility_in_lhs << " primal cone " << pri_cone << std::endl;
	// scaled Ax - b for equality and scaled Ax pour ce qui reste
	precond.scale_primal_residual_in_place(
			{proxqp::from_eigen, primal_residual});

	//precond.scale_primal_in_place(x_e);
	//precond.scale_dual_in_place(y_e);

	primal_feasibility_lhs = std::max(primal_feasibility_lhs,pri_cone);

	return proxsuite::linalg::veg::tuplify(primal_feasibility_lhs, dual_feasibility_lhs);
}

template <typename T,typename I>
void ldl_solve(VectorViewMut<T> sol, VectorView<T> rhs,isize n_tot,
	proxsuite::linalg::sparse::MatMut<T, I> ldl, 
	Eigen::MINRES<detail::AugmentedKktSocp<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>&
			iterative_solver, bool do_ldlt,proxsuite::linalg::veg::dynstack::DynStackMut stack,
			T* ldl_values,I* perm,I*ldl_col_ptrs,I const* perm_inv){
	LDLT_TEMP_VEC_UNINIT(T, work_, n_tot, stack);
	auto rhs_e = rhs.to_eigen();
	auto sol_e = sol.to_eigen();
	auto zx = proxsuite::linalg::sparse::util::zero_extend; 

	if (do_ldlt) {

		for (isize i = 0; i < n_tot; ++i) {
			work_[i] = rhs_e[isize(zx(perm[i]))];
		}

		proxsuite::linalg::sparse::dense_lsolve<T, I>( //
				{proxsuite::linalg::sparse::from_eigen, work_},
				ldl.as_const());

		for (isize i = 0; i < n_tot; ++i) {
			work_[i] /= ldl_values[isize(zx(ldl_col_ptrs[i]))];
		}

		proxsuite::linalg::sparse::dense_ltsolve<T, I>( //
				{proxsuite::linalg::sparse::from_eigen, work_},
				ldl.as_const());

		for (isize i = 0; i < n_tot; ++i) { 
			sol_e[i] = work_[isize(zx(perm_inv[i]))];
		}
	} else {
		work_ = iterative_solver.solve(rhs_e);
		sol_e = work_;
	}
}

template <typename T,typename I>
void ldl_iter_solve_noalias_socp(VectorViewMut<T> sol,
									VectorView<T> rhs,
									VectorView<T> init_guess,
									ProxSocpResults<T>& results,
									SocpModel<T,I>& data,
									isize n_tot,
									proxsuite::linalg::sparse::MatMut<T, I> ldl, 
	Eigen::MINRES<detail::AugmentedKktSocp<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>&
			iterative_solver, bool do_ldlt,proxsuite::linalg::veg::dynstack::DynStackMut stack,
			T* ldl_values,I* perm,I*ldl_col_ptrs,I const* perm_inv,
			ProxSocpSettings<T> const& settings,
			proxsuite::linalg::sparse::MatMut<T, I> kkt){
		auto rhs_e = rhs.to_eigen();
		auto sol_e = sol.to_eigen();

		if (init_guess.dim == sol.dim) {
			sol_e = init_guess.to_eigen();
		} else {
			sol_e.setZero();
		}

		LDLT_TEMP_VEC_UNINIT(T, err, n_tot, stack);
		//proxsuite::linalg::sparse::MatMut<T, I> kkt = data.kkt_mut(); // scaled in the setup

		T prev_err_norm = std::numeric_limits<T>::infinity();

		for (isize solve_iter = 0; solve_iter < settings.nb_iterative_refinement;
				++solve_iter) {

			auto err_x = err.head(data.dim);
			auto err_y = err.segment(data.dim, data.n_eq);
			auto err_z = err.tail(data.n_in);

			auto sol_x = sol_e.head(data.dim);
			auto sol_y = sol_e.segment(data.dim, data.n_eq);
			auto sol_z = sol_e.tail(data.n_in); // removed active set condition

			err = -rhs_e;

			if (solve_iter > 0) {
				T mu_eq_neg = -results.info.mu_eq;
				T mu_in_neg = -results.info.mu_in; 
				detail::noalias_symhiv_add(err, kkt.to_eigen(), sol_e); // replaced kkt_active by kkt
				err_x += results.info.rho * sol_x;
				err_y += mu_eq_neg * sol_y;
                err_z += mu_in_neg * sol_z; // removed active set condition
			}

			T err_norm = infty_norm(err);
			if (err_norm > prev_err_norm / T(2)) {
				break;
			}
			prev_err_norm = err_norm;

			ldl_solve({proxqp::from_eigen, err}, 
					  {proxqp::from_eigen, err},
					  n_tot,
					  ldl,
					  iterative_solver,
					  do_ldlt,
					  stack,
					  ldl_values,
					  perm,
					  ldl_col_ptrs,
					  perm_inv);

			sol_e -= err;
		}
}


template <typename T,typename I>
void qdldl_iter_solve_noalias_socp(VectorViewMut<T> sol,
									VectorView<T> rhs,
									ProxSocpResults<T>& results,
									SocpModel<T,I>& data,
									ProxSocpWorkspace<T,I>& work,
									isize n_tot,
			ProxSocpSettings<T> const& settings,
			proxsuite::linalg::sparse::MatMut<T, I> kkt,
			proxsuite::linalg::veg::dynstack::DynStackMut stack){
		auto rhs_e = rhs.to_eigen();
		auto sol_e = sol.to_eigen();
		LDLT_TEMP_VEC_UNINIT(T, err, n_tot, stack);
		T prev_err_norm = std::numeric_limits<T>::infinity();

		//std::cout << "kkt.to_eigen() " << kkt.to_eigen() << std::endl;

		for (isize solve_iter = 0; solve_iter < settings.nb_iterative_refinement;
				++solve_iter) {

			auto err_x = err.head(data.dim);
			auto err_y = err.segment(data.dim, data.n_eq);
			auto err_z = err.segment(data.dim+data.n_eq, data.n_in);
			auto err_soc = err.tail(data.n_soc);

			auto sol_x = sol_e.head(data.dim);
			auto sol_y = sol_e.segment(data.dim, data.n_eq);
			auto sol_z = sol_e.segment(data.dim+data.n_eq,data.n_in);
			auto sol_soc = sol_e.tail(data.n_soc);

			err = -rhs_e;

			if (solve_iter > 0) {
				T mu_eq_neg = -results.info.mu_eq;
				T mu_in_neg = -results.info.mu_in; 
				T mu_soc_neq = -results.info.mu_soc; 

				detail::noalias_symhiv_add(err, kkt.to_eigen(), sol_e); // replaced kkt_active by kkt
				err_x += results.info.rho * sol_x;
				err_y += mu_eq_neg * sol_y;
                err_z += mu_in_neg * sol_z; // removed active set condition
				err_soc += mu_soc_neq * sol_soc;
			}

			T err_norm = infty_norm(err);
			if (err_norm > prev_err_norm / T(2)) {
				break;
			}
			prev_err_norm = err_norm;

			for (c_int i = 0; i < n_tot; i++) {
				work.xz_tilde[i] = err[i];
			}
			//std::cout << "hello" << std::endl;
			work.linsys_solver->solve(work.linsys_solver,work.xz_tilde);
			for (c_int i = 0; i < n_tot; i++) {
				err[i] = work.xz_tilde[i];
			}		
			//std::cout << "i managed to make one solve" << std::endl;

			sol_e -= err;
		}
}



template<typename T, typename I>
void ldl_solve_in_place_socp(VectorViewMut<T> rhs,
						VectorView<T> init_guess,
						ProxSocpResults<T>& results,
						SocpModel<T,I>& data,
						isize n_tot,
						proxsuite::linalg::sparse::MatMut<T, I> ldl, 
	Eigen::MINRES<detail::AugmentedKktSocp<T, I>,
			Eigen::Upper | Eigen::Lower,
			Eigen::IdentityPreconditioner>&
			iterative_solver, bool do_ldlt,proxsuite::linalg::veg::dynstack::DynStackMut stack,T* ldl_values,I* perm,I*ldl_col_ptrs,I const* perm_inv,
			ProxSocpSettings<T> const& settings,
			proxsuite::linalg::sparse::MatMut<T, I> kkt) {
	LDLT_TEMP_VEC_UNINIT(T, tmp, n_tot, stack);
	ldl_iter_solve_noalias_socp(
		{proxqp::from_eigen, tmp}, 
		rhs.as_const(), 
		init_guess,
		results, 
		data, 
		n_tot, 
		ldl, 
		iterative_solver,
		do_ldlt,
		stack,
		ldl_values,
		perm,
		ldl_col_ptrs,
		perm_inv,
		settings,
		kkt);
	rhs.to_eigen() = tmp;
}

/*
template<typename T>
using DMat = Eigen::Matrix<T, -1, -1>;

template<typename T, typename I>
DMat<T> inner_reconstructed_matrix(proxsuite::linalg::sparse::MatMut<T, I> ldl,bool do_ldlt){
	PROXSUITE::LINALG::VEG_ASSERT(do_ldlt);
	auto ldl_dense = ldl.to_eigen().toDense();
	auto l = DMat<T>(ldl_dense.template triangularView<Eigen::UnitLower>());
	auto lt = l.transpose();
	auto d = ldl_dense.diagonal().asDiagonal();
	auto mat = DMat<T>(l * d * lt);
	return mat;
};

template<typename T,typename I>
DMat<T> reconstructed_matrix(proxsuite::linalg::sparse::MatMut<T, I> ldl,bool do_ldlt,I const* perm_inv,isize n_tot){
	auto mat = inner_reconstructed_matrix(ldl,do_ldlt);
	auto mat_backup = mat;
	for (isize i = 0; i < n_tot; ++i) {
		for (isize j = 0; j < n_tot; ++j) {
			mat(i, j) = mat_backup(perm_inv[i], perm_inv[j]);
		}
	}
	return mat;
};

template<typename T,typename I>
DMat<T>  reconstruction_error(proxsuite::linalg::sparse::MatMut<T, I> ldl,bool do_ldlt,I const* perm_inv,ProxSocpResults<T> results,SocpModel<T,I> data,isize n_tot){
	T mu_eq_neg = -results.info.mu_eq;
	T mu_in_neg = -results.info.mu_in;
	proxsuite::linalg::sparse::MatMut<T, I> kkt = data.kkt_mut();
	auto diff = DMat<T>(
			reconstructed_matrix(ldl,do_ldlt,perm_inv,n_tot) -
			DMat<T>(DMat<T>(kkt.to_eigen())
					.template selfadjointView<Eigen::Upper>()));
	diff.diagonal().head(data.dim).array() -= results.info.rho;
	diff.diagonal().segment(data.dim, data.n_eq).array() -= mu_eq_neg;
	for (isize i = 0; i < data.n_in; ++i) {
		diff.diagonal()[data.dim + data.n_eq + i] -=  mu_in_neg;
	}
	return diff;
};
*/

template<typename T,typename I>
void projection_onto_cones(ProxSocpWorkspace<T, I>& work,
						   SocpModel<T, I>& model,
						   ProxSocpResults<T>& results,
						   proxsuite::linalg::veg::dynstack::DynStackMut stack,
						   VecMap<T> z_prev,
						   VecMapMut<T> y_prev,
						   VecMapMut<T> z_hat){

	LDLT_TEMP_VEC_UNINIT(T, tmp_hat, model.n_in + model.n_soc, stack);
    tmp_hat.head(model.n_in) = z_prev.head(model.n_in) + results.info.mu_in * y_prev.segment(model.n_eq,model.n_in);
	//std::cout << " z_prev.tail(model.n_soc)  " <<  z_prev.tail(model.n_soc) << "  y_prev.tail(model.n_soc) " <<  y_prev.tail(model.n_soc) << std::endl;
    tmp_hat.tail(model.n_soc) = z_prev.tail(model.n_soc) + results.info.mu_soc * y_prev.tail(model.n_soc);
    // projection on linear cone
    z_hat.head(model.n_in) = tmp_hat.head(model.n_in) + detail::positive_part(work.internal.l_scaled - tmp_hat.head(model.n_in))+detail::negative_part(-tmp_hat.head(model.n_in)+work.internal.u_scaled.segment(model.n_eq,model.n_in));//detail::negative_part(work.internal.u_scaled.segment(model.n_eq,model.n_in)-tmp_hat.segment(model.n_eq,model.n_in));
    z_hat.tail(model.n_soc) = tmp_hat.tail(model.n_soc);
	// project over all cones
	//std::cout << "tmp cone " << tmp_hat.tail(model.n_soc) << std::endl;
    isize j = 0;
	isize n_cone = model.dims.rows();
	for (isize it = 0; it < n_cone; ++it) { 
		isize dim_cone = model.dims[it];
		
		T aux_lhs_part = tmp_hat.segment(model.n_in+j+1,dim_cone-1).norm();
		T aux_rhs_part = tmp_hat[model.n_in+j];
		T mean = (aux_lhs_part + aux_rhs_part) * 0.5;
		//std::cout << "it : " << it <<  " dim_cone : " << dim_cone << " aux_lhs_part " << aux_lhs_part << " aux_rhs_part "<< aux_rhs_part << std::endl;
		if (aux_lhs_part <= -aux_rhs_part){
			//std::cout << "project onto zero " << std::endl;
			z_hat.segment(model.n_in+j,dim_cone).setZero();
		} else if (aux_lhs_part > std::abs(aux_rhs_part)){
			T scaling_factor = mean / aux_lhs_part ;
			//std::cout << "scale by " << scaling_factor << std::endl;
			z_hat[model.n_in+j] = mean;
			z_hat.segment(model.n_in+j+1,dim_cone-1) *= scaling_factor;
		}
		j+=dim_cone;
	}
	//std::cout << "z_hat " << z_hat << std::endl;
}

template <typename T, typename I, typename P>
void prox_socp_solve(
		ProxSocpResults<T>& results,
		SocpModel<T, I>& data,
		ProxSocpSettings<T> const& settings,
		ProxSocpWorkspace<T, I>& work,
		P& precond) {
	if (settings.compute_timings){
		work.timer.stop();
		work.timer.start();
	}
	isize n = data.dim;
	isize n_eq = data.n_eq;
	isize n_in = data.n_in;
	isize n_soc = data.n_soc;
	isize m = n_eq+n_in+n_soc;
	isize n_tot = n + m;

	if(work.internal.dirty) // the following is used when a solve has already been executed (and without any intermediary socpmodel update)
	{
		//proxsuite::linalg::sparse::MatMut<T, I> kkt_unscaled = data.kkt_mut_unscaled();
		//auto kkt_top_n_rows = detail::top_rows_mut_unchecked(proxsuite::linalg::veg::unsafe, kkt_unscaled, data.dim);
		//isize m = data.n_eq+data.n_in+data.n_soc;
		//proxsuite::linalg::sparse::MatMut<T, I> H_unscaled = 
		//		detail::middle_cols_mut(kkt_top_n_rows, 0, data.dim, data.H_nnz);
		//proxsuite::linalg::sparse::MatMut<T, I> AT_unscaled =
		//		detail::middle_cols_mut(kkt_top_n_rows, data.dim, m, data.A_nnz);
		//SparseMat<T, I> H_triu = H_unscaled.to_eigen().template triangularView<Eigen::Upper>();
		//sparse::SocpView<T, I> socp = {
		//		{proxsuite::linalg::sparse::from_eigen, H_triu},
		//		{proxsuite::linalg::sparse::from_eigen, data.g},
		//		{proxsuite::linalg::sparse::from_eigen, AT_unscaled.to_eigen()},
		//		{proxsuite::linalg::sparse::from_eigen, data.l},
		//		{proxsuite::linalg::sparse::from_eigen, data.u}};
		//results.cleanup(); // for the moment no initial guess

		// TODO MODIFY INITIAL GUESS ACCORDING TO y and Z which are not the same
		//switch (settings.initial_guess) { // the following is used when one solve has already been executed
		//			//case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS:{
		//			//	results.cleanup(); 
		//			//	break;
		//			//}
		//			//case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT:{
		//			//	// keep solutions but restart ProxSocpWorkspace and results
		//			//	results.cold_start();
		//			//	precond.scale_primal_in_place({proxsuite::proxqp::from_eigen, results.x});
		//			//	precond.scale_dual_in_place_eq({proxsuite::proxqp::from_eigen,results.y});
		//			//	precond.scale_dual_in_place_in({proxsuite::proxqp::from_eigen,results.z});
		//			//	break;
		//			//}
		//			case InitialGuessStatus::NO_INITIAL_GUESS:{
		//				results.cleanup(); 
		//				break;
		//			}
		//			//case InitialGuessStatus::WARM_START:{
		//			//	results.cold_start(); // because there was already a solve, precond was already computed if set so
		//			//	precond.scale_primal_in_place({proxsuite::proxqp::from_eigen, results.x}); // it contains the value given in entry for warm start
		//			//	precond.scale_dual_in_place_eq({proxsuite::proxqp::from_eigen,results.y});
		//			//	precond.scale_dual_in_place_in({proxsuite::proxqp::from_eigen,results.z});
		//			//	break;
		//			//}
		//			//case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT:{
		//			//	// keep ProxSocpWorkspace and results solutions except statistics
		//			//	results.cleanup_statistics();
		//			//	precond.scale_primal_in_place({proxsuite::proxqp::from_eigen, results.x});
		//			//	precond.scale_dual_in_place_eq({proxsuite::proxqp::from_eigen,results.y});
		//			//	precond.scale_dual_in_place_in({proxsuite::proxqp::from_eigen,results.z});
		//			//	break;
		//			//}
		//}
		results.cleanup(); 
		work.rho_vec     = static_cast<c_float*>(c_malloc(m * sizeof(c_float)));
		//work.rho_vec_inv = static_cast<c_float*>(c_malloc(m * sizeof(c_float)));
		work.xz_tilde = static_cast<c_float*>(c_calloc(n + m, sizeof(c_float)));
		proxsuite::linalg::sparse::MatRef<T, I>  H_unscaled = data.H_unscaled();
		proxsuite::linalg::sparse::MatRef<T, I>  A_unscaled = data.A_unscaled();

		SocpView<T, I> socp_unscaled = {
				H_unscaled,
				{proxsuite::linalg::sparse::from_eigen, data.g},
				A_unscaled,
				{proxsuite::linalg::sparse::from_eigen, data.l},
				{proxsuite::linalg::sparse::from_eigen, data.u},
		};
		work.setup_impl(socp_unscaled, data,results, settings, false, precond, P::scale_socp_in_place_req(proxsuite::linalg::veg::Tag<T>{}, data.dim, m));
	}else{
		// the following is used for a first solve after initializing or updating the Qp object 
		/* TODO MODIFY INITIAL GUESS ACCORDING TO y and Z which are not the same
		switch (settings.initial_guess) {
					case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS:{
						break;
					}
					case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT:{
						precond.scale_primal_in_place({proxsuite::proxqp::from_eigen, results.x}); // meaningful for when there is an upate of the model and one wants to warm start with previous result
						precond.scale_dual_in_place_eq({proxsuite::proxqp::from_eigen,results.y});
						precond.scale_dual_in_place_in({proxsuite::proxqp::from_eigen,results.z});
						break;
					}
					case InitialGuessStatus::NO_INITIAL_GUESS:{
						break;
					}
					case InitialGuessStatus::WARM_START:{
						precond.scale_primal_in_place({proxsuite::proxqp::from_eigen, results.x});
						precond.scale_dual_in_place_eq({proxsuite::proxqp::from_eigen,results.y});
						precond.scale_dual_in_place_in({proxsuite::proxqp::from_eigen,results.z});
						break;
					}
					case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT:{
						precond.scale_primal_in_place({proxsuite::proxqp::from_eigen, results.x}); // meaningful for when there is an upate of the model and one wants to warm start with previous result
						precond.scale_dual_in_place_eq({proxsuite::proxqp::from_eigen,results.y});
						precond.scale_dual_in_place_in({proxsuite::proxqp::from_eigen,results.z});
						break;
					}
		}	
		*/
	}
	if (settings.verbose){
		sparse::print_setup_header_socp(settings,results, data);
	}
	using namespace proxsuite::linalg::veg::literals;
	namespace util = proxsuite::linalg::sparse::util;
	auto zx = util::zero_extend;

	proxsuite::linalg::veg::dynstack::DynStackMut stack = work.stack_mut();


	const T machine_eps = std::numeric_limits<T>::epsilon();
	T fact(0);
	T fact_mean(0);
	T fact_eq(0);
	T fact_mean_eq(0);
	T fact_in(0);
	T fact_mean_in(0);
	T fact_cone(0);
	T fact_mean_cone(0);
	T fact_dual(0);
	T fact_mean_dual(0);
	isize n_mean(0);
	isize freq_mu_update(settings.check_termination);
	T aux_fact(0);
	
	VectorViewMut<T> x{proxqp::from_eigen, results.x};
	VectorViewMut<T> y{proxqp::from_eigen, results.y};
	VectorViewMut<T> z{proxqp::from_eigen, results.z};

	//proxsuite::linalg::sparse::MatMut<T, I> kkt = data.kkt_mut();
	//auto kkt_top_n_rows = detail::top_rows_mut_unchecked(proxsuite::linalg::veg::unsafe, kkt, n);
	//proxsuite::linalg::sparse::MatMut<T, I> H_scaled =
	//		detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);
	//proxsuite::linalg::sparse::MatMut<T, I> AT_scaled =
	//		detail::middle_cols_mut(kkt_top_n_rows, n, m, data.A_nnz);

	proxsuite::linalg::sparse::MatMut<T, I> H_scaled = data.H_mut();
	proxsuite::linalg::sparse::MatMut<T, I> AT_scaled = data.A_mut();

	auto& g_scaled_e = work.internal.g_scaled;
	auto& l_scaled_e = work.internal.l_scaled;
	auto& u_scaled_e = work.internal.u_scaled;

	SocpViewMut<T, I> socp_scaled = {
			H_scaled,
			{proxsuite::linalg::sparse::from_eigen, g_scaled_e},
			AT_scaled,
			{proxsuite::linalg::sparse::from_eigen, l_scaled_e},
			{proxsuite::linalg::sparse::from_eigen, u_scaled_e},
	};

	T const primal_feasibility_rhs_1_eq = infty_norm(data.u);
	T const primal_feasibility_rhs_1_in_l = infty_norm(data.l);
	T const dual_feasibility_rhs_2 = infty_norm(data.g);
	T primal_feasibility_eq_lhs(1);
	T primal_feasibility_in_lhs(1);
	T pri_cone(1);

	//auto ldl_col_ptrs = work.internal.ldl.col_ptrs.ptr_mut();
	//proxsuite::linalg::veg::Tag<I> itag;
	//proxsuite::linalg::veg::Tag<T> xtag;
	//bool do_ldlt = work.internal.do_ldlt;
	//isize ldlt_ntot = do_ldlt ? n_tot : 0;
	//auto _perm = stack.make_new_for_overwrite(itag, ldlt_ntot);
	//I const* perm_inv = work.internal.ldl.perm_inv.ptr_mut();
	//I* perm = _perm.ptr_mut();
	//if (do_ldlt) {
	//	// compute perm from perm_inv
	//	for (isize i = 0; i < n_tot; ++i) {
	//		perm[isize(zx(perm_inv[i]))] = I(i);
	//	}
	//}
	//auto& iterative_solver = *work.internal.matrix_free_solver.get();
	//I* etree = work.internal.ldl.etree.ptr_mut();
	//I* ldl_nnz_counts =  work.internal.ldl.nnz_counts.ptr_mut();
	//I* ldl_row_indices = work.internal.ldl.row_indices.ptr_mut();
	//T* ldl_values = work.internal.ldl.values.ptr_mut();
	//proxsuite::linalg::veg::SliceMut<bool> active_constraints=results.active_constraints.as_mut();
	//proxsuite::linalg::sparse::MatMut<T, I> ldl = {
	//		proxsuite::linalg::sparse::from_raw_parts,
	//		n_tot,
	//		n_tot,
	//		0,
	//		ldl_col_ptrs,
	//		do_ldlt ? ldl_nnz_counts : nullptr,
	//		ldl_row_indices,
	//		ldl_values,
	//};
	auto x_e = x.to_eigen();
	auto y_e = y.to_eigen();
	auto z_e = z.to_eigen();

	/* TODO add initial guess options here
	LDLT_TEMP_VEC_UNINIT(T, rhs, n_tot, stack);
	LDLT_TEMP_VEC_UNINIT(T, no_guess, 0, stack);

	rhs.head(n) = -g_scaled_e;
	rhs.segment(n, n_eq) = b_scaled_e;
	rhs.segment(n + n_eq, n_in).setZero();

	ldl_solve_in_place_socp({proxqp::from_eigen, rhs}, {proxqp::from_eigen, no_guess},results,data, n_tot, ldl, iterative_solver,do_ldlt,stack,ldl_values,perm,ldl_col_ptrs,perm_inv,settings);
	x_e = rhs.head(n);
	y_e = rhs.tail(n_eq+n_in);
	*/

	x_e.setZero();
	y_e.setZero();
	z_e.setZero(); // check if warm start or not
	//sparse::refactorize_socp<T,I>(
	//		work,
	//		results,
	//		kkt,
	//		active_constraints,
	//		data,
	//		stack,
	//		xtag);
	T new_mu_soc = results.info.mu_soc;
	T new_mu_soc_inv = results.info.mu_soc_inv;
	T new_mu_in = results.info.mu_in;
	T new_mu_in_inv = results.info.mu_in_inv;
	T new_mu_eq = results.info.mu_eq;
	T new_mu_eq_inv = results.info.mu_eq_inv;

	for (isize iter = 0; iter < settings.max_iter; ++iter) {
		results.info.iter += 1;
		T primal_feasibility_eq_rhs_0(0);

		T dual_feasibility_rhs_0(0);
		T dual_feasibility_rhs_1(0);

		LDLT_TEMP_VEC_UNINIT(T, primal_residual, m, stack);
		LDLT_TEMP_VEC_UNINIT(T, primal_residual_in_scaled, n_in, stack);
		LDLT_TEMP_VEC_UNINIT(T, dual_residual_scaled, n, stack);

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		auto is_primal_feasible = [&](T primal_feasibility_lhs) -> bool {
			T rhs_pri = settings.eps_abs;
			if (settings.eps_rel != 0) {
				rhs_pri += settings.eps_rel * std::max({
																					primal_feasibility_eq_rhs_0,//||Ax||
																					primal_feasibility_rhs_1_eq,//||u||
																					primal_feasibility_rhs_1_in_l,//||l||
																			});
			}
			return primal_feasibility_lhs <= rhs_pri;
		};
		auto is_dual_feasible = [&](T dual_feasibility_lhs) -> bool {
			T rhs_dua = settings.eps_abs;
			if (settings.eps_rel != 0) {
				rhs_dua += settings.eps_rel * std::max({
																					dual_feasibility_rhs_0,//||Hx||
																					dual_feasibility_rhs_1,//||ATy||
																					dual_feasibility_rhs_2//||g||
																			});
			}

			return dual_feasibility_lhs <= rhs_dua;
		};
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		VEG_BIND(
				auto,
				(primal_feasibility_lhs, dual_feasibility_lhs),
				primal_dual_socp_residual(
						primal_residual,
						primal_residual_in_scaled,
						dual_residual_scaled,
						primal_feasibility_eq_rhs_0,
						dual_feasibility_rhs_0,
						dual_feasibility_rhs_1,
						precond,
						data,
						socp_scaled.as_const(),
						//{proxqp::from_eigen,x_e},
						//{proxqp::from_eigen,y_e},
						detail::vec(x_e),
						detail::vec(y_e),
						primal_feasibility_eq_lhs,
						primal_feasibility_in_lhs,
						pri_cone,
						stack));
		if (settings.verbose){
			//LDLT_TEMP_VEC_UNINIT(T, aux, n, stack);
			//aux.setZero();
			//detail::noalias_symhiv_add(aux, socp_scaled.H.to_eigen(), x_e);
			//precond.unscale_dual_residual_in_place({proxqp::from_eigen, aux});
			//precond.unscale_primal_in_place({proxqp::from_eigen, x_e});
			//precond.unscale_dual_in_place({proxqp::from_eigen, y_e});
			//aux *= 0.5;
			//aux += data.g;
			//results.info.objValue = (aux).dot(x_e);
			T tmp_ = fact_mean / std::max(n_mean,isize(1));
			std::cout << "\033[1;32m[iteration " << iter + 1  << "]\033[0m" << std::endl;
			std::cout << std::scientific << std::setw(2) << std::setprecision(2) << 
			"| primal residual=" << primal_feasibility_lhs << "| dual residual=" << dual_feasibility_lhs << " | mu_soc=" << results.info.mu_soc << " | rho=" << results.info.rho << //std::endl;
			" fact " << fact << " fact mean " <<  tmp_ << " mu update fact bound " << settings.mu_update_fact_bound << " freq_mu_update " << freq_mu_update << std::endl;
			std::cout << "results.info.mu_updates "<<results.info.mu_updates<<" mu_eq "<<results.info.mu_eq<<  "mu_in " << results.info.mu_in << " fact_dual " << fact_mean_dual / std::max(n_mean,isize(1)) << " fact_in " << fact_mean_in / std::max(n_mean,isize(1))<<std::endl;
			results.info.pri_res = primal_feasibility_lhs;
			results.info.dua_res = dual_feasibility_lhs;
			//precond.scale_primal_in_place(VectorViewMut<T>{from_eigen, x_e});
			//precond.scale_dual_in_place(VectorViewMut<T>{from_eigen, y_e});
		}
		if (is_primal_feasible(primal_feasibility_lhs) &&
			is_dual_feasible(dual_feasibility_lhs)) {
			break;
		}

		LDLT_TEMP_VEC_UNINIT(T, x_prev_e, n, stack);
		LDLT_TEMP_VEC_UNINIT(T, y_prev_e, m, stack);
		//LDLT_TEMP_VEC_UNINIT(T, z_prev_e, n_in+n_soc, stack);
		LDLT_TEMP_VEC_UNINIT(T, z_hat, n_in+n_soc, stack);
		//LDLT_TEMP_VEC(T, dw_prev, n_tot, stack);

		x_prev_e = x_e;
		y_prev_e = y_e;
		//z_prev_e = z_e;
		//std::cout << "x_prev_e " << x_prev_e << " y_prev_e " << y_prev_e << " z_prev_e " << z_prev_e << std::endl;

		projection_onto_cones( work,
							   data,
							   results,
							   stack,
							   detail::vec(z_e),
							   y_prev_e,
							   z_hat);
		//std::cout << "z_hat " << z_hat << std::endl;
		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		
		LDLT_TEMP_VEC_UNINIT(T, dw, n_tot, stack);
		dw.setZero();
		// Linear solving step
	
		///
		//
		// [ H + rho I    AeqT    A_inT	     A_socT]      -[ H×x + g + AT×y               ]
		// [ A_eq      -µ_eq I      0            0 ]      -[ A_eq×x -  b                  ]
		// [ A_in         0        -µ_in I       0 ]    = -[ A_in×x -  z_in               ]
		// [ A_soc        0         0     -µ_soc I ] dw = -[ A_soc×x - u_soc - z_soc      ]

		//auto rhs = dw;
		//std::cout << "primal_residual before " << primal_residual << std::endl;

		//std::cout << "work.internal.u_scaled " << work.internal.u_scaled << std::endl;
		//std::cout << "dual_residual_scaled " << dual_residual_scaled << std::endl;
		//rhs.head(n) = -dual_residual_scaled; // H×x + g + AT×y
		//rhs.tail(m) = -primal_residual ; // n_eq first constraints contains already A_eq x - b scaled 
		//std::cout << " rhs " << rhs << std::endl;
		//ldl_solve_in_place_socp(
		//		{proxqp::from_eigen, rhs},  
		//		{proxqp::from_eigen, dw_prev},
		//		results,
		//		data,
		//		n_tot,
		//		ldl,
		//		iterative_solver,
		//		do_ldlt,
		//		stack,
		//		ldl_values,
		//		perm,
		//		ldl_col_ptrs,
		//		perm_inv,
		//		settings,
		//		kkt);

		// recalculate the residuals to be more accurate and robust
		
		//dual_residual_scaled = socp_scaled.g.to_eigen();
		//primal_residual.setZero();
		//{
		//	detail::noalias_symhiv_add(dw.head(n), socp_scaled.H.to_eigen(), x_e);
		//	dual_residual_scaled += dw.head(n); // contains now scaled(g+Hx)
		//	dw.head(n).setZero();
		//	detail::noalias_gevmmv_add(
		//			dw.head(n), primal_residual, socp_scaled.AT.to_eigen(), y_e, x_e);// ATy = A.T ye ; prim_res = A*x
		//	dual_residual_scaled += dw.head(n); // contains now scaled(g+Hx+ATy)
		//}
		//primal_residual.head(n_eq) -= work.internal.u_scaled.head(n_eq);
		primal_residual.tail(n_in+n_soc) -= z_hat; // contains now scaled(Ax - ze) 
		primal_residual.tail(n_soc) -= work.internal.u_scaled.tail(n_soc); // contains now A_soc×x - u_soc - z_soc 


		for (c_int i = 0; i < data.dim; i++) {
			// Cycle over part related to x variables
			work.xz_tilde[i] = -dual_residual_scaled[i];
		}
		for (c_int i = 0; i < m; i++) {
			// Cycle over dual variable in the first step (nu)
			work.xz_tilde[i + data.dim] = -primal_residual[i];
		}

		work.linsys_solver->solve(work.linsys_solver,work.xz_tilde);
		for (c_int i = 0; i < m+data.dim; i++) {
			// Cycle over dual variable in the first step (nu)
			dw[i] = work.xz_tilde[i];
		}
		//qdldl_iter_solve_noalias_socp({proxqp::from_eigen, dw},  
		//							{proxqp::from_eigen, rhs},
		//							results,data, work,n_tot,settings, kkt,stack);
		
		//std::cout << "dw " << dw << std::endl;
		//std::cout << "dw " << dw << std::endl;
	    z_e.head(n_in) = z_hat.head(n_in) + 0.1 * results.info.mu_in * dw.segment(n+n_eq,n_in);
        z_e.tail(n_soc) = z_hat.tail(n_soc) + settings.tau * results.info.mu_soc * dw.tail(n_soc);
        x_e += dw.head(n);
        y_e += dw.tail(n_eq+n_in+n_soc);


		T primal_feasibility_eq_lhs_new(primal_feasibility_eq_lhs);
		T primal_feasibility_in_lhs_new(primal_feasibility_in_lhs);
		T pri_cone_new(pri_cone);
		VEG_BIND(
				auto,
				(primal_feasibility_lhs_new, dual_feasibility_lhs_new),
				primal_dual_socp_residual(
						primal_residual,
						primal_residual_in_scaled,
						dual_residual_scaled,
						primal_feasibility_eq_rhs_0,
						dual_feasibility_rhs_0,
						dual_feasibility_rhs_1,
						precond,
						data,
						socp_scaled.as_const(),
						//{proxqp::from_eigen,x_e},
						//{proxqp::from_eigen,y_e},
						detail::vec(x_e),
						detail::vec(y_e),
						primal_feasibility_eq_lhs_new,
						primal_feasibility_in_lhs_new,
						pri_cone_new,
						stack));
		bool prim_feas = is_primal_feasible(primal_feasibility_lhs_new);
		bool dual_feas = is_dual_feasible(dual_feasibility_lhs_new);
		if (prim_feas &&
			dual_feas) {
			break;
		}

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		// mu update
		
		fact = (primal_feasibility_lhs_new ) / (primal_feasibility_lhs  + machine_eps);
		fact_mean += fact;

		fact_eq = (primal_feasibility_eq_lhs_new ) / (primal_feasibility_eq_lhs  + machine_eps);
		fact_mean_eq += fact_eq;

		fact_in = (primal_feasibility_in_lhs_new ) / (primal_feasibility_in_lhs  + machine_eps);
		fact_mean_in += fact_in;

		fact_cone = (pri_cone_new ) / (pri_cone  + machine_eps);
		fact_mean_cone += fact_cone;

		fact_dual = (dual_feasibility_lhs_new)/(dual_feasibility_lhs+machine_eps);
		fact_mean_dual += fact_dual;

		n_mean+=1;
		bool change(false);

		if (iter % freq_mu_update == 0 && iter >=freq_mu_update){
			
			
			bool increase_penalty = (!prim_feas) ;
			bool decrease_penalty = prim_feas && (!dual_feas) || dual_feas && !prim_feas;
			
			aux_fact = fact_mean / n_mean;
			if (aux_fact>= settings.mu_update_fact_bound){

					T aux_dual = fact_mean_dual / n_mean ;
					bool real_increase =  results.info.mu_soc_inv!=1.E5 && results.info.mu_in_inv!=1.E7;
					if (decrease_penalty && aux_dual >= settings.mu_update_fact_bound ){
						//new_mu_in = std::min(results.info.mu_in*100,1.E-4);
						//new_mu_in_inv = std::max(results.info.mu_in_inv*0.01,1.E4);
						new_mu_soc = std::min(results.info.mu_soc*100,1.E-1);
						new_mu_soc_inv = std::max(results.info.mu_soc_inv*0.01,1.E1);
						if (n_soc==0){
							new_mu_eq = 1.E-9;
							new_mu_eq_inv = 1.E9;
						}
						fact_mean = 0;
						n_mean = 0;
						change = std::abs(results.info.mu_soc_inv-new_mu_soc_inv)>machine_eps;// ||std::abs(results.info.mu_in_inv-new_mu_in_inv)>machine_eps;
					} else if (increase_penalty){
						T aux_in = fact_mean_in/n_mean;
						T aux_soc = fact_mean_cone/n_mean;

						if (aux_soc>=settings.mu_update_fact_bound){
							new_mu_soc = std::max(results.info.mu_soc*0.1,1.E-5);// std::min(results.info.mu_soc*10,1.E-1);
							new_mu_soc_inv = std::min(results.info.mu_soc_inv*10,1.E5);//std::max(results.info.mu_soc_inv*0.1,1.E1);
							fact_mean = 0;
							n_mean = 0;
							
						}
						if (aux_in >=settings.mu_update_fact_bound && false){
							new_mu_in = std::max(results.info.mu_in*0.1,1.E-7);
							new_mu_in_inv = std::min(results.info.mu_in_inv*10,1.E7);
						}
						change = std::abs(results.info.mu_soc_inv-new_mu_soc_inv)>machine_eps ||std::abs(results.info.mu_in_inv-new_mu_in_inv)>machine_eps;
					}
			}

			freq_mu_update = std::max(roundUp(iter,settings.check_termination),settings.check_termination);
		}

		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		if (change){
			if (true ){
				//std::cout <<"update " << "new_mu_eq = " << new_mu_eq << " and  results.info.mu_eq= " <<  results.info.mu_eq  << " new_mu_in= "<<new_mu_in<<" results.info.mu_in= "<< results.info.mu_in<<std::endl;
				results.info.mu_soc = new_mu_soc;
				results.info.mu_soc_inv = new_mu_soc_inv;
				results.info.mu_in = new_mu_in;
				results.info.mu_in_inv = new_mu_in_inv;
				results.info.mu_eq = new_mu_eq;
				results.info.mu_eq_inv = new_mu_eq_inv;
				++results.info.mu_updates;
				for (isize i = 0; i < m; i++) {
					if (i < data.n_eq){
						work.rho_vec[i]     = results.info.mu_eq_inv;
					} else if (i >= data.n_eq && i < data.n_eq + data.n_in){
						work.rho_vec[i]     = results.info.mu_in_inv;
					} else {
						work.rho_vec[i]     = results.info.mu_soc_inv;
					}
				}
				update_linsys_solver_rho_vec_qdldl(work.linsys_solver, work.rho_vec);
			}
		}
		//if (results.info.mu_soc != new_mu_soc) {
		//	{ ++results.info.mu_updates; }
		//	//refactorize();
		//	isize w_values = 1;// un seul elt non nul
		//	T alpha  = 0;
		//	for (isize j=0; j<n_soc;++j){
		//		I row_index = j+n+n_eq+n_in;
		//		alpha = results.info.mu_soc - new_mu_soc;
		//		T value = 1;
		//		proxsuite::linalg::sparse::VecRef<T, I> w{
		//				proxsuite::linalg::veg::from_raw_parts,
		//				n+n_eq+n_in+n_soc,
		//				w_values,
		//				&row_index, // &: adresse de row index
		//				&value,
		//		};
		//		ldl= rank1_update(
		//			ldl,
		//			etree,
		//			perm_inv,
		//			w,
		//			alpha,
		//			stack);
		//	}
		//}


	}

	LDLT_TEMP_VEC_UNINIT(T, aux, n, stack);
	aux.setZero();					
	detail::noalias_symhiv_add(aux, socp_scaled.H.to_eigen(), x_e);
	precond.unscale_dual_residual_in_place({proxqp::from_eigen, aux}); 
	precond.unscale_primal_in_place({proxqp::from_eigen, x_e});
	precond.unscale_dual_in_place({proxqp::from_eigen, y_e});

	aux *= 0.5;
	aux += data.g;
	results.info.objValue = (aux).dot(x_e);

	if (settings.compute_timings) {
		results.info.solve_time = work.timer.elapsed().user; // in microseconds
		results.info.run_time =
				results.info.solve_time + results.info.setup_time;
		if (settings.verbose) {
			std::cout << "-------------------SOLVER STATISTICS-------------------" << std::endl;
			std::cout << "total iter:   " << results.info.iter << std::endl;
			std::cout << "mu updates:   " << results.info.mu_updates << std::endl;
			std::cout << "objective:    " << results.info.objValue << std::endl;
			switch (results.info.status) { 
						case QPSolverOutput::PROXQP_SOLVED:{
							std::cout << "status:       " << "Solved" << std::endl;
							break;
						}
						case QPSolverOutput::PROXQP_MAX_ITER_REACHED:{
							std::cout << "status:       " << "Maximum number of iterations reached" << std::endl;
							break;
						}
						case QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE:{
							std::cout << "status:       " << "Primal infeasible" << std::endl;
							break;
						}
						case QPSolverOutput::PROXQP_DUAL_INFEASIBLE:{
							std::cout << "status:       " << "Dual infeasible" << std::endl;
							break;
						}
			}
			std::cout << "run time:     " << results.info.solve_time << std::endl;
			std::cout << "--------------------------------------------------------" << std::endl;
		}
	} else {
		if (settings.verbose) {
			std::cout << "-------------------SOLVER STATISTICS-------------------" << std::endl;
			std::cout << "total iter:   " << results.info.iter << std::endl;
			std::cout << "mu updates:   " << results.info.mu_updates << std::endl;
			std::cout << "objective:    " << results.info.objValue << std::endl;
			switch (results.info.status) { 
						case QPSolverOutput::PROXQP_SOLVED:{
							std::cout << "status:       " << "Solved." << std::endl;
							break;
						}
						case QPSolverOutput::PROXQP_MAX_ITER_REACHED:{
							std::cout << "status:       " << "Maximum number of iterations reached" << std::endl;
							break;
						}
						case QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE:{
							std::cout << "status:       " << "Primal infeasible" << std::endl;
							break;
						}
						case QPSolverOutput::PROXQP_DUAL_INFEASIBLE:{
							std::cout << "status:       " << "Dual infeasible" << std::endl;
							break;
						}
			}
			std::cout << "--------------------------------------------------------" << std::endl;
		}
	}

	work.set_dirty();

    if (work.linsys_solver) {
      if (work.linsys_solver->free) {
        work.linsys_solver->free(work.linsys_solver);
      }
    }
	c_free(work.rho_vec);
    //c_free(work.rho_vec_inv);

}

template <typename T,typename I>
using SparseMat = Eigen::SparseMatrix<T, Eigen::ColMajor, I>;
template <typename T>
using VecRef = Eigen::Ref<Eigen::Matrix<T, DYN, 1> const>;
template <typename T>
using MatRef = Eigen::Ref<Eigen::Matrix<T, DYN, DYN> const>;
template <typename T>
using Vec = Eigen::Matrix<T, DYN, 1>;

template <typename T,typename I>
void prox_socp_update_proximal_parameters(
		ProxSocpResults<T>& results,
		ProxSocpWorkspace<T,I>& work,
		std::optional<T> rho_new,
		std::optional<T> mu_eq_new,
		std::optional<T> mu_in_new,
		std::optional<T> mu_soc_new) {

	if (rho_new != std::nullopt) {
		results.info.rho = rho_new.value();
		work.internal.proximal_parameter_update=true;
	}
	if (mu_eq_new != std::nullopt) {
		results.info.mu_eq = mu_eq_new.value();
		results.info.mu_eq_inv = T(1) / results.info.mu_eq;
		work.internal.proximal_parameter_update=true;
	}
	if (mu_in_new != std::nullopt) {
		results.info.mu_in = mu_in_new.value();
		results.info.mu_in_inv = T(1) / results.info.mu_in;
		work.internal.proximal_parameter_update=true;
	}
	if (mu_soc_new != std::nullopt) {
		results.info.mu_soc = mu_soc_new.value();
		results.info.mu_soc_inv = T(1) / results.info.mu_soc;
		work.internal.proximal_parameter_update=true;
	}
}

template <typename T>
void prox_socp_warm_starting(
		std::optional<VecRef<T>> x_wm,
		std::optional<VecRef<T>> y_wm,
		ProxSocpResults<T>& results,
		ProxSocpSettings<T>& settings) {
	isize m = results.y.rows();
	if (m!=0){
			if(x_wm != std::nullopt && y_wm != std::nullopt){
					results.x = x_wm.value().eval();
					results.y = y_wm.value().eval();
					settings.initial_guess = InitialGuessStatus::WARM_START;
			}
	}  else {
		// m = 0
		if(x_wm != std::nullopt ){
					results.x = x_wm.value().eval();
					settings.initial_guess = InitialGuessStatus::WARM_START;
		}
	}	
}

template <typename T, typename I, typename P>
void prox_socp_setup(
		SocpView<T, I> socp,
		ProxSocpResults<T>& results,
		SocpModel<T, I>& data,
		ProxSocpWorkspace<T, I>& work,
		ProxSocpSettings<T>& settings,
		P& precond,
		PreconditionerStatus& preconditioner_status) {
	isize n = socp.H.nrows();
	//isize m = socp.AT.ncols();
	isize m = socp.AT.nrows();
	

	if (results.x.rows() != n) {
		results.x.resize(n);
		results.x.setZero();
	}
	if (results.y.rows() != m) {
		results.y.resize(m);
		results.y.setZero();
	}
	isize n_cones = data.n_in+data.n_soc;
	if (results.z.rows() != n_cones) {
		results.z.resize(n_cones);
		results.z.setZero();
	}
	//if (results.active_constraints.len() != data.n_in) {
	//	results.active_constraints.resize(data.n_in);
	//	for (isize i = 0; i < data.n_in; ++i) {
	//		results.active_constraints[i] = true;
	//	}
	//}
	work.rho_vec     = static_cast<c_float*>(c_malloc(m * sizeof(c_float)));
	//work.rho_vec_inv = static_cast<c_float*>(c_malloc(m * sizeof(c_float)));
	work.xz_tilde = static_cast<c_float*>(c_calloc(n + m, sizeof(c_float)));
	bool execute_preconditioner_or_not = false;
	switch (preconditioner_status)
	{
	case PreconditionerStatus::EXECUTE:
		execute_preconditioner_or_not = true;
		break;
	case PreconditionerStatus::IDENTITY:
		execute_preconditioner_or_not = false;
		break;
	case PreconditionerStatus::KEEP:
		// keep previous one
		execute_preconditioner_or_not = false;
		break;
	}
	work.setup_impl(
			socp, 
			data, 
			results,
			settings,
			execute_preconditioner_or_not,
			precond,
			P::scale_socp_in_place_req(proxsuite::linalg::veg::Tag<T>{}, n, m));
	/*
	if (!settings.constant_update && false){
		const isize n = data.dim;
		Eigen::SelfAdjointEigenSolver<SparseMat<T, I>> es(n);

		proxsuite::linalg::sparse::MatMut<T, I> kkt = data.kkt_mut();

		auto kkt_top_n_rows = detail::top_rows_mut_unchecked(proxsuite::linalg::veg::unsafe, kkt, n);
		proxsuite::linalg::sparse::MatMut<T, I> H_scaled =
			detail::middle_cols_mut(kkt_top_n_rows, 0, n, data.H_nnz);
		SparseMat<T, I> H_triu = (H_scaled.to_eigen().template triangularView<Eigen::Upper>()).transpose();
		es.compute(H_triu,Eigen::EigenvaluesOnly);
		auto lambda = es.eigenvalues();
		T min_lam = std::max(lambda[0],T(1.E-12));
		T max_lam = lambda[n-1];
		results.info.kappa = T(max_lam/(min_lam));
		T kappa_used(max_lam/T(1.E-12));
		if (results.info.kappa>=1.E3){
			settings.mu_update_factor = pow(kappa_used, 0.01);
		}else{
			settings.mu_update_factor = pow(kappa_used, 0.05);
		}
		settings.mu_update_inv_factor = T(1./settings.mu_update_factor);
	}
	*/
	/* TODO
	switch (settings.initial_guess) { // the following is used when initiliazing the Qp object or updating it
                case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS:{
					
					if (work.internal.proximal_parameter_update){
						results.cleanup_all_except_prox_parameters(); 
					}else{
						results.cleanup(); 
					}
                    break;
                }
                case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT:{
					// keep solutions but restart workspace and results
					
					if (work.internal.proximal_parameter_update){
						results.cleanup_statistics(); 
					}else{
						results.cold_start(); 
					}
                    break;
                }
                case InitialGuessStatus::NO_INITIAL_GUESS:{
					
					if (work.internal.proximal_parameter_update){
						results.cleanup_all_except_prox_parameters(); 
					}else{
						results.cleanup(); 
					}
                    break;
                }
				case InitialGuessStatus::WARM_START:{
					
					if (work.internal.proximal_parameter_update){
						results.cleanup_all_except_prox_parameters(); 
					}else{
						results.cleanup(); 
					}
                    break;
                }
                case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT:{
                    // keep workspace and results solutions except statistics
					
					results.cleanup_statistics(); // always keep prox parameters (changed or previous ones)
                    break;
                }
	}
	*/
}
///// QP object
template <typename T,typename I>
struct SOCP {
public:
	ProxSocpResults<T> results;
	ProxSocpSettings<T> settings;
	SocpModel<T,I> model;
	ProxSocpWorkspace<T,I> work;
    preconditioner::RuizSocpEquilibration<T, I> ruiz;

	SOCP(isize _dim, isize _m, isize _n_eq, isize _n_in, Eigen::Matrix<isize, Eigen::Dynamic, 1> dims_)
			: results(_dim, _n_in, dims_.sum() , _n_eq),
				settings(),
				model(_dim,_n_eq,_n_in, dims_),
				work(),ruiz(_dim,_m,_n_eq,_n_in,1e-3,10,preconditioner::Symmetry::UPPER) {
			work.timer.stop();
			//work.internal.do_symbolic_fact=true;
				}

	void init(
			const SparseMat<T, I>& H,
			const Vec<T>& g,
			const SparseMat<T, I>& A,
			const Vec<T>& u,
			const Vec<T>& l,
			bool compute_preconditioner_=true,
			std::optional<T> rho = std::nullopt,
			std::optional<T> mu_eq = std::nullopt,
			std::optional<T> mu_in = std::nullopt,
			std::optional<T> mu_soc = std::nullopt) {
		if (settings.compute_timings){
			work.timer.stop();
			work.timer.start();
		}
		work.internal.proximal_parameter_update= false;
		PreconditionerStatus preconditioner_status;
		if (compute_preconditioner_){
			preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
		}else{
			preconditioner_status = proxsuite::proxqp::PreconditionerStatus::IDENTITY;
		}
		//settings.compute_preconditioner = compute_preconditioner_;
		SparseMat<T, I> H_triu = H.template triangularView<Eigen::Upper>();
		//SparseMat<T, I> AT = A.transpose();
		//std::cout << "AT " << AT << std::endl;
		sparse::SocpView<T, I> socp = {
				{proxsuite::linalg::sparse::from_eigen, H_triu},
				{proxsuite::linalg::sparse::from_eigen, g},
				{proxsuite::linalg::sparse::from_eigen, A},
				{proxsuite::linalg::sparse::from_eigen, l},
				{proxsuite::linalg::sparse::from_eigen, u}};
		prox_socp_update_proximal_parameters(results, work, rho, mu_eq, mu_in,mu_soc);
		prox_socp_setup(
                socp,
                results,
                model,
                work,
				settings,
				ruiz,
                preconditioner_status);
		if (settings.compute_timings){
			results.info.setup_time += work.timer.elapsed().user; // in microseconds
		}
	};

	void solve() {
		prox_socp_solve( //
				results,
				model,
				settings,
				work,
                ruiz);
	};

	/*!
	 * Solves the QP problem using PROXQP algorithm and a warm start.
	 * @param x primal warm start.
	 * @param y dual warm start.
	 */
	void solve(std::optional<VecRef<T>> x,
			std::optional<VecRef<T>> y) {
		auto start = std::chrono::high_resolution_clock::now();
		prox_socp_warm_starting(x, y, results, settings);
		sparse::prox_socp_solve( //
				results,
				model,
				settings,
				work,
                ruiz);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration =
				std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		results.info.solve_time = T(duration.count());
		//results.info.run_time = results.info.solve_time + results.info.setup_time;
	};
	void cleanup() {
		results.cleanup();
	}
};



} // namespace sparse
} // namespace proxqp
} // namespace proxsuite


#endif /* end of include guard PROXSUITE_PROXQP_SPARSE_SOLVER_PROX_SOCP_HPP */
