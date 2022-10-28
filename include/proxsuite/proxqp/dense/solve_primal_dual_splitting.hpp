//
// Copyright (c) 2022, INRIA
//
/** \file */

#ifndef PROXSUITE_PROXQP_DENSE_SOLVER_PROX_SOCP_HPP
#define PROXSUITE_PROXQP_DENSE_SOLVER_PROX_SOCP_HPP

#include <proxsuite/linalg/dense/core.hpp>
#include <proxsuite/linalg/veg/vec.hpp>
#include <proxsuite/proxqp/dense/workspace_primal_dual_splitting.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz_socp.hpp>
#include <proxsuite/proxqp/sparse/preconditioner/identity.hpp>
#include <iostream>
#include <iomanip> 
#include <cmath>

namespace proxsuite {
namespace proxqp {
namespace dense {
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

template<typename T>
void print_setup_header_primal_dual_splitting(const PrimalDualSplittingSettings<T>& settings,PrimalDualSplittingResults<T>& results, const SocpModel<T>& model){

  print_line();
  std::cout  <<"                           ProxSOCP  -  A Proximal SOCP Solver\n"
             <<"               (c) Antoine Bambade, Adrien Taylor, Justin Carpentier\n"
             <<"                                Inria Paris 2022        \n"
          << std::endl;
  print_line();

  // Print variables and constraints
  std::cout << "problem:  " << std::noshowpos <<std::endl;
  std::cout << "          variables n = " << model.dim <<  ",\n" <<
  "          linear cone constraints n_in = "<< model.n_in << std::endl;

  // Print Settings
  std::cout << "settings: " << std::endl;
  std::cout  <<"          backend = dense," << std::endl;
  std::cout  <<"          eps_abs = " << settings.eps_abs <<" eps_rel = " << settings.eps_rel << std::endl;
  std::cout  <<"          eps_prim_inf = " <<settings.eps_primal_inf <<", eps_dual_inf = " << settings.eps_dual_inf << "," << std::endl;

  std::cout  <<"          rho = " <<results.info.rho << ", mu_in = " << results.info.mu_in << "," << std::endl;
  std::cout  <<"          max_iter = " << settings.max_iter << "," << std::endl;

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
void primal_dual_socp_residual(
					   const SocpModel<T>& model,
                       PrimalDualSplittingResults<T>& results,
                       PrimalDualSplittingWorkspace<T>& work,
                       preconditioner::RuizSocpEquilibration<T>& ruiz,
                       T& primal_feasibility_lhs,
                       T& primal_feasibility_rhs,
					   T& dual_feasibility_lhs,
					   T& dual_feasibility_rhs_0,
					   T& dual_feasibility_rhs_1){
    // dual

	isize n = model.dim;
    isize n_in = model.n_in;
	work.primal_residual_scaled.setZero();
	work.tmp_x.setZero();
    work.tmp_y.setZero();
	work.dual_residual_scaled = work.g_scaled;

	{
		work.tmp_x = work.H_scaled.template selfadjointView<Eigen::Lower>() * results.x;
		work.dual_residual_scaled += work.tmp_x ; // contains now scaled(g+Hx)

		ruiz.unscale_dual_residual_in_place({proxqp::from_eigen, work.tmp_x});
		dual_feasibility_rhs_0 = infty_norm(work.tmp_x); // ||unscaled(Hx)||

		work.tmp_x = work.A_scaled.transpose() * results.y;
		work.dual_residual_scaled += work.tmp_x ; // contains now scaled(g+Hx+ATy)

		ruiz.unscale_dual_residual_in_place({proxqp::from_eigen, work.tmp_x });
		dual_feasibility_rhs_1 = infty_norm(work.tmp_x ); // ||unscaled(ATy)||
	}


	ruiz.unscale_dual_residual_in_place(
			{proxqp::from_eigen, work.dual_residual_scaled}); // contains now unscaled(Hx+g+ATy)
	dual_feasibility_lhs = infty_norm(work.dual_residual_scaled); // ||unscaled(Hx+g+ATy)||
	ruiz.scale_dual_residual_in_place({proxqp::from_eigen, work.dual_residual_scaled});// ||scaled(Hx+g+ATy)||

    // primal 

	work.primal_residual_scaled = work.A_scaled * results.x;

	ruiz.unscale_primal_residual_in_place(
			{proxqp::from_eigen, work.primal_residual_scaled});
	primal_feasibility_rhs = infty_norm(work.primal_residual_scaled); // ||unscaled(Ax)||
	work.tmp_y = proxsuite::proxqp::dense::negative_part(work.primal_residual_scaled - model.u);
	//work.primal_residual_scaled.head(n_eq) -= u.head(n_in); equality taken into accounts in the inequalities

	primal_feasibility_lhs= infty_norm(work.tmp_y.head(n_in));

	//T pri_cone = 0;
    //isize j = 0;
	//isize n_cone = model.dims.rows();
	//{
	//	isize m = n+n_in;
	//	work.tmp_y.tail(model.n_soc) = work.primal_residual_scaled.tail(model.n_soc) - u.tail(model.n_soc);
	//	for (isize it = 0; it < n_cone; ++it) { 
	//		isize dim_cone = model.dims[it];
	//		
	//		T cone_error = std::max(work.tmp_y.segment(m+j+1,dim_cone-1).norm() - work.tmp_y[m+j],0.);
	//		pri_cone = std::max(pri_cone, cone_error);
	//		j+=dim_cone;
	//	}
	//}
	// scaled Ax - b for equality and scaled Ax pour ce qui reste
	//ruiz.scale_primal_residual_in_place(
	//		{proxqp::from_eigen, work.primal_residual_scaled});
	//primal_feasibility_lhs = std::max(primal_feasibility_lhs,pri_cone);

}


template<typename T>
void projection_onto_cones(PrimalDualSplittingWorkspace<T>& work,
						   SocpModel<T>& model,
                           PrimalDualSplittingSettings<T> const&  settings,
						   PrimalDualSplittingResults<T>& results){
	//isize m_ = model.dim+model.n_eq+model.n_in;
	
	work.u_prev = work.u;
    work.tmp_y = -settings.alpha_over_relaxed * results.y+ (1-settings.alpha_over_relaxed)*work.u_prev + work.v/(results.info.admm_step_size );
    
	//std::cout << "-settings.alpha_over_relaxed * results.y " << -settings.alpha_over_relaxed * results.y << std::endl;
	//std::cout << "(1-settings.alpha_over_relaxed)*work.u_prev " << (1-settings.alpha_over_relaxed)*work.u_prev << std::endl;
	//std::cout << "work.v/(results.info.admm_step_size ) " << work.v/(results.info.admm_step_size ) << std::endl;
	//std::cout << "rhs for proj " << work.tmp_y  << std::endl;
	
	// projection on linear cone
    work.u.head(model.n_in) = proxsuite::proxqp::dense::positive_part(work.tmp_y.head(model.n_in));
    //work.u.tail(model.n_soc) = work.tmp_y.tail(model.n_soc); // check if it is good
	// project over all cones
    //isize j = 0;
	//isize n_cone = model.dims.rows();
	//for (isize it = 0; it < n_cone; ++it) { 
	//	isize dim_cone = model.dims[it];
	//	T aux_lhs_part = work.tmp_y.segment(m_+j+1,dim_cone-1).norm();
	//	T aux_rhs_part = work.tmp_y[m_+j];
	//	T mean = (aux_lhs_part + aux_rhs_part) * 0.5;
	//	if (aux_lhs_part <= -aux_rhs_part){
	//		work.u.segment(model.n_in+j,dim_cone).setZero();
	//	} else if (aux_lhs_part > std::abs(aux_rhs_part)){
	//		T scaling_factor = mean / aux_lhs_part ;
	//		work.u[model.n_in+j] = mean;
	//		work.u.segment(model.n_in+j+1,dim_cone-1) *= scaling_factor;
	//	}
	//	j+=dim_cone;
	//}
	
	//std::cout << "u " << work.u.head(model.n_in) << std::endl;
}

/*!
 * Setups and performs the first factorization of the regularized KKT matrix of
 * the problem.
 *
 * @param work workspace of the solver.
 * @param model QP problem model as defined by the user (without any scaling
 * performed).
 * @param results solution results.
 */
template<typename T>
void
prox_socp_setup_factorization(PrimalDualSplittingWorkspace<T>& work,
                    const SocpModel<T>& model,
                    PrimalDualSplittingResults<T>& results)
{

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    work.ldl_stack.as_mut(),
  };
  isize m = model.n_in;
  work.kkt.topLeftCorner(model.dim, model.dim) = work.H_scaled;
  work.kkt.topLeftCorner(model.dim, model.dim).diagonal().array() +=
    results.info.rho;
  work.kkt.block(0, model.dim, model.dim,m) =
    work.A_scaled.transpose();
  work.kkt.block(model.dim, 0, m, model.dim) = work.A_scaled;
  //work.kkt.bottomRightCorner(m, m).setZero();
  work.kkt.diagonal()
    .segment(model.dim, m)
    .setConstant(-results.info.mu_in);
  work.ldl.factorize(work.kkt, stack);
}

template<typename T>
void
socp_refactorize(const SocpModel<T>& model,
            PrimalDualSplittingResults<T>& results,
            PrimalDualSplittingWorkspace<T>& work,
            T rho_new)
{

  if (rho_new == results.info.rho) {
    return;
  }

  work.dw.setZero();
  work.kkt.diagonal().head(model.dim).array() +=
    rho_new - results.info.rho;
  work.kkt.diagonal().tail(model.n_in).array() =
    -results.info.mu_in;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
  };
  work.ldl.factorize(work.kkt, stack);

}

template<typename T>
void
socp_iterative_solve_with_permut_fact( //
  const PrimalDualSplittingSettings<T>& settings,
  const SocpModel<T>& model,
  PrimalDualSplittingResults<T>& results,
  PrimalDualSplittingWorkspace<T>& work,
  T eps,
  bool upper_part)
{

  work.err.setZero();
  work.dw.setZero();
  i32 it = 0;
  i32 it_stability = 0;
  T preverr(0);
  T err(0);
  if (upper_part){
	work.dw.head(model.dim) = work.rhs.head(model.dim);
  }else{
	work.dw.tail(model.n_in) = work.rhs.tail(model.n_in);
  }
  
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
  };
  if (upper_part){
	work.ldl.primal_solve_in_place(work.dw.head(model.dim), stack);
	work.err.head(model.dim) = work.rhs.head(model.dim);
	work.err.head(model.dim).noalias() -=
		work.H_scaled.template selfadjointView<Eigen::Lower>() *
		work.dw.head(model.dim);
	work.err.head(model.dim) -=
		results.info.rho * work.dw.head(model.dim);
	preverr = infty_norm(work.err.head(model.dim));
  }else{
	//std::cout << "work.dw.tail(model.n_in) " << work.dw.tail(model.n_in) << std::endl;
	work.ldl.dual_solve_in_place(work.dw.tail(model.n_in),model.dim, stack);
	//std::cout << "G reconstructed " << work.ldl.dbg_reconstructed_matrix_low(model.dim,model.n_in) << std::endl;
	//std::cout << "dw.tail " << work.dw.tail(model.n_in) << std::endl; 
	work.tmp_y = -work.dw.tail(model.n_in);
	work.ldl.dual_multiply_in_place(work.tmp_y,model.dim,stack);
	//std::cout << "work.tmp_y " << work.tmp_y  << std::endl;
	work.err.tail(model.n_in) = work.rhs.tail(model.n_in);
	work.err.tail(model.n_in) -= work.tmp_y;
	preverr = infty_norm(work.err.tail(model.n_in));
  }
  err = preverr;
  ++it;
  //to put in debuger mode
  if (settings.verbose) {
          std::cout << "infty_norm(res) " <<preverr
                                                  << std::endl;
  }
  //std::cout << "d " << work.ldl.d() << std::endl;
  
  while (err >= eps) {

    if (it >= settings.nb_iterative_refinement) {
      break;
    }

    ++it;
	if (upper_part){
		work.ldl.primal_solve_in_place(work.err.head(model.dim), stack);
		work.dw.head(model.dim) += work.err.head(model.dim);
	}else{
		work.ldl.dual_solve_in_place(work.err.tail(model.n_in),model.dim, stack);
		work.dw.tail(model.n_in) += work.err.tail(model.n_in);
	}

    work.err.setZero();

	if (upper_part){
		work.err.head(model.dim) = work.rhs.head(model.dim);
		work.err.head(model.dim).noalias() -=
			work.H_scaled.template selfadjointView<Eigen::Lower>() *
			work.dw.head(model.dim);
		work.err.head(model.dim) -=
			results.info.rho * work.dw.head(model.dim);
		err = infty_norm(work.err.head(model.dim));
	}else{
		work.err.tail(model.n_in) = work.rhs.tail(model.n_in);
		work.tmp_y = -work.dw.tail(model.n_in);
		work.ldl.dual_multiply_in_place(work.tmp_y,model.dim,stack);
		work.err.tail(model.n_in) -= work.tmp_y;
		err = infty_norm(work.err.tail(model.n_in));
	}

    if (err > preverr) {
      it_stability += 1;

    } else {
      it_stability = 0;
    }
    if (it_stability == 2) {
      break;
    }
    preverr = err;
    // to put in debug mode
    if (settings.verbose) {
            std::cout << "err "
                                                    <<
    infty_norm(work.err) << std::endl;
    }
    
  }

  if (err >=
      std::max(eps, settings.eps_refact)) {
    socp_refactorize(model, results, work, results.info.rho);
    it = 0;
    it_stability = 0;

    work.dw = work.rhs;

	if (upper_part){
		work.ldl.primal_solve_in_place(work.dw.head(model.dim), stack);
		work.err.head(model.dim) = work.rhs.head(model.dim);
		work.err.head(model.dim).noalias() -=
			work.H_scaled.template selfadjointView<Eigen::Lower>() *
			work.dw.head(model.dim);
		work.err.head(model.dim) -=
			results.info.rho * work.dw.head(model.dim);
		preverr = infty_norm(work.err.head(model.dim));
	}else{
		work.err.tail(model.n_in) = work.rhs.tail(model.n_in);
		work.ldl.dual_solve_in_place(work.dw.tail(model.n_in),model.dim, stack);
		work.tmp_y = -work.dw.tail(model.n_in);
		work.ldl.dual_multiply_in_place(work.tmp_y,model.dim,stack);
		work.err.tail(model.n_in) -= work.tmp_y;
		preverr = infty_norm(work.err.tail(model.n_in));
	}

    err = preverr;
    ++it;
    // to put in debug mode
    //if (settings.verbose) {
    //        std::cout << "infty_norm(res) "
    //                                                <<
    //infty_norm(work.err) << std::endl;
    //}
    
    while (err >= eps) {

      if (it >= settings.nb_iterative_refinement) {
        break;
      }
      ++it;
	  if (upper_part){
	  	work.ldl.primal_solve_in_place(work.err.head(model.dim), stack);
	  	work.dw.head(model.dim) += work.err.head(model.dim);
	  }else{
	  	work.ldl.dual_solve_in_place(work.err.tail(model.n_in),model.dim, stack);
	  	work.dw.tail(model.n_in) += work.err.tail(model.n_in);
	  }
  
	  work.err.setZero();
  
	  if (upper_part){
	  	work.err.head(model.dim) = work.rhs.head(model.dim);
	  	work.err.head(model.dim).noalias() -=
	  		work.H_scaled.template selfadjointView<Eigen::Lower>() *
	  		work.dw.head(model.dim);
	  	work.err.head(model.dim) -=
	  		results.info.rho * work.dw.head(model.dim);
	  	err = infty_norm(work.err.head(model.dim));
	  }else{
	  	work.err.tail(model.n_in) = work.rhs.tail(model.n_in);
	  	work.tmp_y = -work.dw.tail(model.n_in);
	  	work.ldl.dual_multiply_in_place(work.tmp_y,model.dim,stack);
	  	work.err.tail(model.n_in) -= work.tmp_y;
	  	err = infty_norm(work.err.tail(model.n_in));
	  }
      if (err > preverr) {
        it_stability += 1;

      } else {
        it_stability = 0;
      }
      if (it_stability == 2) {
        break;
      }
      preverr = err;
      // to put in debug mode
      //if (settings.verbose) {
      //        std::cout << "infty_norm(res) "
      //                                                <<
      //infty_norm(work.err) << std::endl;
      //}
      
    }
  }
  work.rhs.setZero();
}

template<typename T>
void
socp_mu_update(const SocpModel<T>& model,
          PrimalDualSplittingResults<T>& results,
          PrimalDualSplittingWorkspace<T>& work,
		  T mu_in_old,
		  T mu_in_new)
{
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
  };

  isize n = model.dim;
  isize n_in = model.n_in;

  if ((n_in) == 0) {
    return;
  }

  work.tmp_y.setConstant(mu_in_old - mu_in_new);

  {
    auto _indices = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, n_in);
    isize* indices = _indices.ptr_mut();
    for (isize k = 0; k < n_in; ++k) {
      indices[k] = n + k;
    }
    work.ldl.diagonal_update_clobber_indices(
      indices,n_in, work.tmp_y, stack);
  }

}

template <typename T>
T compute_admm_step_size(T m,T M,T eps){
    return sqrt(m*M) * pow((M/m),eps);
}

template <typename T>
void update_gk(PrimalDualSplittingWorkspace<T>& work,
               PrimalDualSplittingResults<T>& results,
			   SocpModel<T>& model,
			   PrimalDualSplittingSettings<T> const& settings
			   ){
    bool upper_part = true;
    work.rhs.head(model.dim) = -work.g_scaled + results.info.rho * work.x_prev;
	//std::cout << "rhs for g_k " << work.rhs.head(model.dim) << std::endl;
    socp_iterative_solve_with_permut_fact( //
    settings,
    model,
    results,
    work,
    T(1.e-4),
	upper_part);
	work.g_k = work.A_scaled * work.dw.head(model.dim);
	work.g_k += results.info.mu_in * work.y_prev ;
	work.g_k -= work.u_scaled;
	//std::cout << "g_k " << work.g_k << std::endl;
}

template <typename T>
void update_y(PrimalDualSplittingWorkspace<T>& work,
               PrimalDualSplittingResults<T>& results,
			   SocpModel<T>& model,
			   PrimalDualSplittingSettings<T> const& settings
			   ){
    bool upper_part = false;
    work.rhs.tail(model.n_in) = work.g_k - results.info.admm_step_size * work.u + work.v;
    //std::cout << "rhs for y " << work.rhs.tail(model.n_in) << std::endl;
	socp_iterative_solve_with_permut_fact( //
    settings,
    model,
    results,
    work,
    T(1.e-4),
	upper_part);
	results.y = -work.dw.tail(model.n_in);
	//std::cout << "y " << results.y << std::endl;
}

template <typename T>
void update_x(PrimalDualSplittingWorkspace<T>& work,
               PrimalDualSplittingResults<T>& results,
			   SocpModel<T>& model,
			   PrimalDualSplittingSettings<T> const& settings
			   ){
    bool upper_part = true;
    work.rhs.head(model.dim) = -work.A_scaled.transpose() * results.y;
	work.rhs.head(model.dim) -= work.g_scaled ;
	work.rhs.head(model.dim) += results.info.rho * work.x_prev;
    //std::cout << "rhs for x " << work.rhs.head(model.dim)  << std::endl;
	socp_iterative_solve_with_permut_fact( //
    settings,
    model,
    results,
    work,
    T(1.e-4),
	upper_part);
	results.x = work.dw.head(model.dim);
	//std::cout << "x " << results.x << std::endl;
}

template <typename T>
void inner_residuals(PrimalDualSplittingWorkspace<T>& work,
               PrimalDualSplittingResults<T>& results,
			   SocpModel<T>& model,
			   PrimalDualSplittingSettings<T> const& settings,
			   T& inner_dual_residual,
			   T& inner_primal_residual
			   ){
	proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
    };
	work.tmp_y = -results.y;
	//std::cout << "rhs for Gy " << work.tmp_y << std::endl;
	work.ldl.dual_multiply_in_place(work.tmp_y,model.dim,stack); // computes (A * H_rho^-1 * A.T + mu I + admm_step_size I) y
	//std::cout << "Gy " << work.tmp_y << std::endl;
	work.tmp_y += work.g_k + work.v + results.info.admm_step_size * results.y ; // so we remove admm_step_size * y as we want the residuals withou it 
	inner_dual_residual = infty_norm(work.tmp_y);
	inner_primal_residual = infty_norm(proxsuite::proxqp::dense::positive_part(results.y));
	//std::cout << "inner_dual_residual " << inner_dual_residual << std::endl;
	//std::cout << "inner_primal_residual " << inner_primal_residual << std::endl;
}

template <typename T>
void prox_inner_primal_dual_residual(
					   const SocpModel<T>& model,
                       PrimalDualSplittingResults<T>& results,
                       PrimalDualSplittingWorkspace<T>& work,
                       T& primal_feasibility_lhs){
    // primal 
	//std::cout << " work.A_scaled * results.x " <<  work.A_scaled * results.x << std::endl;
	//std::cout << " results.info.mu_in * work.y_prev  " << results.info.mu_in * work.y_prev  <<std::endl;
	//std::cout << " work.u_scaled " << work.u_scaled<< std::endl;
	work.tmp_y = work.A_scaled * results.x + results.info.mu_in * work.y_prev - work.u_scaled ;
	//std::cout << "tmp " << work.tmp_y  << std::endl;
	work.primal_residual_scaled = proxsuite::proxqp::dense::negative_part(work.tmp_y) - results.info.mu_in * results.y;

	primal_feasibility_lhs = infty_norm(work.primal_residual_scaled );

}

template <typename T>
void solve_inner_problem(PrimalDualSplittingWorkspace<T>& work,
               PrimalDualSplittingResults<T>& results,
			   SocpModel<T>& model,
			   PrimalDualSplittingSettings<T> const& settings,
			   T eps_k
			   ){
	T inner_dual_residual(10.);
	T inner_primal_residual(10.);
	T primal_dual_prox_residual(10);
	T new_mu_inner(results.info.admm_step_size);
	for (isize iter = 0; iter < settings.max_iter_inner_loop; ++iter) {
		results.info.iter +=1;
		update_y(work,results,model,settings);
		projection_onto_cones( work,
							   model,
                               settings,
							   results);
		work.v += settings.tau * results.info.admm_step_size * (-settings.alpha_over_relaxed * results.y - work.u + (1-settings.alpha_over_relaxed) * work.u_prev);
		//std::cout << "v " << work.v << std::endl;
		inner_residuals(work,
						results,
						model,
						settings,
						inner_dual_residual,
						inner_primal_residual);

		if (iter % settings.check_termination == 0 && iter >= settings.check_termination){
			//std::cout << "coucou there can be a intern mu update" << std::endl;
			if (inner_primal_residual > settings.mu_update_fact_bound * inner_dual_residual){
					results.info.eps_current += settings.eps_update;
					new_mu_inner = compute_admm_step_size(results.info.mu_in,results.info.max_eig,results.info.eps_current );
					{ ++results.info.mu_updates; }
					socp_mu_update(model,
					       results,
					       work,
						   results.info.mu_in + results.info.admm_step_size, 
					       results.info.mu_in + new_mu_inner);
					results.info.admm_step_size = new_mu_inner;
			} else if (inner_primal_residual < settings.mu_update_fact_bound_inv * inner_dual_residual){
					results.info.eps_current -= settings.eps_update;
					new_mu_inner = compute_admm_step_size(results.info.mu_in,results.info.max_eig,results.info.eps_current );
					{ ++results.info.mu_updates; }
					socp_mu_update(model,
					       results,
					       work,
						   results.info.mu_in + results.info.admm_step_size, 
					       results.info.mu_in + new_mu_inner);
					results.info.admm_step_size = new_mu_inner;
			}
		}
		update_x(work,results,model,settings);
		prox_inner_primal_dual_residual(model,results,work,primal_dual_prox_residual); 
		
		if (settings.verbose) {
		std::cout << "\033[1;34m[inner iteration " << iter + 1 << "]\033[0m"
					<< std::endl;
		std::cout << std::scientific << std::setw(2) << std::setprecision(2)
					<< "| inner residual=" << primal_dual_prox_residual << " | eps_current=" << results.info.eps_current
					<< " | inner_primal_residual =" << inner_primal_residual
					<< " | inner_dual_residual =" << inner_dual_residual
					<< " | admm step size =" << results.info.admm_step_size
					<< std::endl;
		}
		
		if (primal_dual_prox_residual <= eps_k){
			break;
		}
	}

}


template <typename T>
void prox_socp_solve(
		PrimalDualSplittingResults<T>& results,
		SocpModel<T>& model,
		PrimalDualSplittingSettings<T> const& settings,
		PrimalDualSplittingWorkspace<T>& work,
		preconditioner::RuizSocpEquilibration<T>& ruiz) {
	if (settings.compute_timings){
		work.timer.stop();
		work.timer.start();
	}
	if(work.dirty) // the following is used when a solve has already been executed (and without any intermediary socpmodel update)
	{
		work.cleanup(); // for the moment no initial guess only
        results.cleanup();
		work.H_scaled = model.H;
		work.g_scaled = model.g;
		work.A_scaled = model.A;

		work.u_scaled = model.u;
		isize m = model.n_in;
		work.u_scaled.head(model.n_in) =
			(model.u.head(model.n_in).array() <= T(1.E20))
			.select(model.u.head(model.n_in),
					Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() +
						T(1.E20));
		//work.l_scaled =
		//	(model.l.array() >= T(-1.E20))
		//	.select(model.l,
		//			Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() -
		//				T(1.E20));
		prox_socp_setup_equilibration(work, settings, ruiz, false); // reuse previous equilibration
		prox_socp_setup_factorization(work, model, results);
        
	}else{
		prox_socp_setup_factorization(work, model, results);
    }
	if (settings.verbose){
		print_setup_header_primal_dual_splitting(settings,results, model);
	}
	isize m = model.n_in;
    proxsuite::linalg::veg::dynstack::DynStackMut stack{
                        proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
                    };
    results.info.max_eig = work.ldl.power_iteration(model.dim,m,stack,settings.power_iteration_max_iter,settings.power_iteration_accuracy);
    results.info.admm_step_size = compute_admm_step_size(results.info.mu_in,results.info.max_eig,results.info.eps_current);
	T new_mu_ext = results.info.mu_in+results.info.admm_step_size;
    T new_admm_step_size(0);
    socp_mu_update(model,results,work,results.info.mu_in,new_mu_ext);
	new_mu_ext = results.info.mu_in;
	T new_mu_inv_ext = T(1)/new_mu_ext;

    const T machine_eps = std::numeric_limits<T>::epsilon();
	T eps_k(0.1);
	isize freq_mu_update(settings.check_termination);
	for (isize iter = 0; iter < settings.max_iter; ++iter) {
		//results.info.iter += 1;

		T dual_feasibility_rhs_0(0);
		T dual_feasibility_rhs_1(0);
		T primal_feasibility_rhs(0);
		T primal_feasibility_lhs(0);
		T dual_feasibility_lhs(0);

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		auto is_primal_feasible = [&](T primal_feasibility_lhs) -> bool {
			T rhs_pri = settings.eps_abs;
			if (settings.eps_rel != 0) {
				rhs_pri += settings.eps_rel * std::max({
																					primal_feasibility_rhs,//||Ax||
																					work.primal_feasibility_rhs_1_in_u,//||u||
																					work.primal_feasibility_rhs_1_in_l,//||l||
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
																					work.dual_feasibility_rhs_2//||g||
																			});
			}

			return dual_feasibility_lhs <= rhs_dua;
		};
		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

		primal_dual_socp_residual(
					   model,
                       results,
                       work,
                       ruiz,
                       primal_feasibility_lhs,
                       primal_feasibility_rhs,
					   dual_feasibility_lhs,
					   dual_feasibility_rhs_0,
					   dual_feasibility_rhs_1);

		if (settings.verbose){

			ruiz.unscale_primal_in_place(VectorViewMut<T>{ from_eigen, results.x });
			ruiz.unscale_dual_in_place(
				VectorViewMut<T>{ from_eigen, results.y });

			results.info.objValue = 0;
			for (Eigen::Index j = 0; j < model.dim; ++j) {
			results.info.objValue +=
				0.5 * (results.x(j) * results.x(j)) * model.H(j, j);
			results.info.objValue +=
				results.x(j) * T(model.H.col(j)
									.tail(model.dim - j - 1)
									.dot(results.x.tail(model.dim - j - 1)));
			}
			results.info.objValue += (model.g).dot(results.x);
			std::cout << "\033[1;32m[iteration " << iter + 1  << "]\033[0m" << std::endl;
			std::cout << std::scientific << std::setw(2) << std::setprecision(2) << 
			"| primal residual=" << primal_feasibility_lhs << "| dual residual=" << dual_feasibility_lhs << " | mu_in=" << results.info.mu_in << " | rho=" << results.info.rho << std::endl;
			results.info.pri_res = primal_feasibility_lhs;
			results.info.dua_res = dual_feasibility_lhs;
			ruiz.scale_primal_in_place(VectorViewMut<T>{from_eigen, results.x});
			ruiz.scale_dual_in_place(VectorViewMut<T>{from_eigen, results.y});
		}
		if (is_primal_feasible(primal_feasibility_lhs) &&
			is_dual_feasible(dual_feasibility_lhs)) {
			break;
		}	

        work.x_prev = results.x;
        work.y_prev = results.y;

		if (iter % settings.check_termination == 0 && iter >=settings.check_termination){
			
			if (primal_feasibility_lhs >= settings.mu_update_fact_bound * dual_feasibility_lhs){
					new_mu_ext = std::max(settings.mu_update_factor * results.info.mu_in,1.e-9);
					new_mu_inv_ext = std::min(settings.mu_update_factor_inv * results.info.mu_in_inv,1.e9);
			} else if (primal_feasibility_lhs <= settings.mu_update_fact_bound_inv * dual_feasibility_lhs){
					new_mu_ext = std::min(settings.mu_update_factor_inv * results.info.mu_in,1.e-9);
					new_mu_inv_ext = std::min(settings.mu_update_factor * results.info.mu_in_inv,1.e9);
			}

			//freq_mu_update = std::max(roundUp(iter,settings.check_termination),settings.check_termination);
		}

		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		
		if (results.info.mu_in != new_mu_ext) {
			{ ++results.info.mu_updates; }
			socp_mu_update(model,
					       results,
					       work,
						   results.info.mu_in + results.info.admm_step_size, 
					       new_mu_ext + results.info.admm_step_size);
			results.info.mu_in = new_mu_ext;
			results.info.mu_in_inv = new_mu_inv_ext;
		}
        // update g_k
		update_gk(work,results,model,settings);
		solve_inner_problem(work,
							results,
							model,
							settings,
							eps_k
				);
		eps_k = std::max(eps_k * 0.1,1.e-9);
	}

	ruiz.unscale_primal_in_place(VectorViewMut<T>{ from_eigen, results.x });
	ruiz.unscale_dual_in_place(
		VectorViewMut<T>{ from_eigen, results.y });
	
	results.info.objValue = 0;
	for (Eigen::Index j = 0; j < model.dim; ++j) {
	results.info.objValue +=
		0.5 * (results.x(j) * results.x(j)) * model.H(j, j);
	results.info.objValue +=
		results.x(j) * T(model.H.col(j)
							.tail(model.dim - j - 1)
							.dot(results.x.tail(model.dim - j - 1)));
	}
	results.info.objValue += (model.g).dot(results.x);

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

	work.dirty = true;


}

template <typename T>
void update_proximal_parameters(
		PrimalDualSplittingResults<T>& results,
		PrimalDualSplittingWorkspace<T>& work,
		std::optional<T> rho_new,
		std::optional<T> mu_in_new) {

	if (rho_new != std::nullopt) {
		results.info.rho = rho_new.value();
		work.proximal_parameter_update=true;
	}
	if (mu_in_new != std::nullopt) {
		results.info.mu_in = mu_in_new.value();
		results.info.mu_in_inv = T(1) / results.info.mu_in;
		work.proximal_parameter_update=true;
	}
}

template <typename T>
void prox_socp_warm_starting(
		std::optional<VecRef<T>> x_wm,
		std::optional<VecRef<T>> y_wm,
		PrimalDualSplittingResults<T>& results,
		PrimalDualSplittingSettings<T>& settings) {
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

template<typename T>
void
prox_socp_setup_equilibration(PrimalDualSplittingWorkspace<T>& work,
                    const PrimalDualSplittingSettings<T>& settings,
                    preconditioner::RuizSocpEquilibration<T>& ruiz,
                    bool execute_preconditioner)
{
  
  dense::SocpViewMut<T> socp_scaled{
    { from_eigen, work.H_scaled }, { from_eigen, work.g_scaled },
    { from_eigen, work.A_scaled }, { from_eigen, work.u_scaled }
  };
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    work.ldl_stack.as_mut(),
  };
  ruiz.scale_socp_in_place(socp_scaled,
                         execute_preconditioner,
                         settings.preconditioner_max_iter,
                         settings.preconditioner_accuracy,
                         stack);
  work.correction_guess_rhs_g = infty_norm(work.g_scaled);
}

template <typename T>
void    setup(
		const Mat<T>& H,
		const VecRef<T>g,
		const Mat<T>& A,
		const VecRef<T> u,
		PrimalDualSplittingResults<T>& results,
		SocpModel<T>& model,
		PrimalDualSplittingWorkspace<T>& work,
		const PrimalDualSplittingSettings<T>& settings,
		preconditioner::RuizSocpEquilibration<T>& ruiz,
		PreconditionerStatus& preconditioner_status) {
  // for the moment no initial guess
  //results.cleanup();
  work.cleanup();
  /*
  switch (settings.initial_guess) {
    case InitialGuessStatus::EQUALITY_CONSTRAINED_INITIAL_GUESS: {
      if (work.proximal_parameter_update) {
        results.cleanup_all_except_prox_parameters();
      } else {
        results.cleanup();
      }
      work.cleanup();
      break;
    }
    case InitialGuessStatus::COLD_START_WITH_PREVIOUS_RESULT: {
      // keep solutions but restart workspace and results
      if (work.proximal_parameter_update) {
        results.cleanup_statistics();
      } else {
        results.cold_start();
      }
      work.cleanup();
      break;
    }
    case InitialGuessStatus::NO_INITIAL_GUESS: {
      if (work.proximal_parameter_update) {
        results.cleanup_all_except_prox_parameters();
      } else {
        results.cleanup();
      }
      work.cleanup();
      break;
    }
    case InitialGuessStatus::WARM_START: {
      if (work.proximal_parameter_update) {
        results
          .cleanup_all_except_prox_parameters(); // the warm start is given at
                                                 // the solve function
      } else {
        results.cleanup();
      }
      work.cleanup();
      break;
    }
    case InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT: {
      if (work.socp_refactorize || work.proximal_parameter_update) {
        work.cleanup(); // meaningful for when there is an upate of the model
                          // and one wants to warm start with previous result
        work.socp_refactorize = true;
      }
      results.cleanup_statistics();
      break;
    }
  }
  */
  model.H = H ; //Eigen::
      //Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      //  H);
   // else model.H remains initialzed to a matrix with zero elements
  model.g = g;

  model.A = A; //Eigen::
      //Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
      //  A);
  // else model.A remains initialized to a matrix with zero elements or zero
    // shape
  model.u = u;
   // else model.u remains initialized to a matrix with zero elements or zero
    // shape

  work.H_scaled = model.H;
  work.g_scaled = model.g;
  work.A_scaled = model.A;

  work.u_scaled = model.u;
  //isize m = model.n_eq+model.n_in;
  //work.u_scaled.segment(model.n_eq,model.n_in) =
  //  (model.u.segment(model.n_eq,model.n_in).array() <= T(1.E20))
  //    .select(model.u.segment(model.n_eq,model.n_in),
  //            Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() +
  //              T(1.E20));
  //work.l_scaled =
  //  (model.l.array() >= T(-1.E20))
  //    .select(model.l,
  //            Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() -
  //              T(1.E20));

  work.primal_feasibility_rhs_1_in_u = infty_norm(work.u_scaled);
  work.dual_feasibility_rhs_2 = infty_norm(model.g);

  switch (preconditioner_status) {
    case PreconditionerStatus::EXECUTE:
      prox_socp_setup_equilibration(work, settings, ruiz, true);
      break;
    case PreconditionerStatus::IDENTITY:
      prox_socp_setup_equilibration(work, settings, ruiz, false);
      break;
    case PreconditionerStatus::KEEP:
      // keep previous one
      prox_socp_setup_equilibration(work, settings, ruiz, false);
      break;
  }
}
///// QP object
template <typename T>
struct SOCP {
public:
	dense::PrimalDualSplittingResults<T> results;
	dense::PrimalDualSplittingSettings<T> settings;
	dense::SocpModel<T> model;
	dense::PrimalDualSplittingWorkspace<T> work;
    preconditioner::RuizSocpEquilibration<T> ruiz;

	SOCP(isize _dim, isize _m, isize _n_in)
			: results(_dim, _m),
				settings(),
				model(_dim,_m),
				work(_dim,_m),ruiz(_dim,_n_in, _m,1e-3,10) {
			work.timer.stop();
				}

	void init(
			const Mat<T>& H,
			const VecRef<T> g,
			const Mat<T>& A,
			const VecRef<T> u,
			bool compute_preconditioner_=true,
			std::optional<T> rho = std::nullopt,
			std::optional<T> mu_in = std::nullopt) {
		if (settings.compute_timings){
			work.timer.stop();
			work.timer.start();
		}
		work.proximal_parameter_update= false;
		PreconditionerStatus preconditioner_status;
		if (compute_preconditioner_){
			preconditioner_status = proxsuite::proxqp::PreconditionerStatus::EXECUTE;
		}else{
			preconditioner_status = proxsuite::proxqp::PreconditionerStatus::IDENTITY;
		}
		//settings.compute_preconditioner = compute_preconditioner_;
		update_proximal_parameters(results, work, rho, mu_in);

		setup(
                H,g,A,u,
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
		prox_socp_solve( //
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



} // namespace dense
} // namespace proxqp
} // namespace proxsuite


#endif /* end of include guard PROXSUITE_PROXQP_DENSE_SOLVER_PROX_SOCP_HPP */
