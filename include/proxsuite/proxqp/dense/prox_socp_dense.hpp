//
// Copyright (c) 2022, INRIA
//
/** \file */

#ifndef PROXSUITE_PROXQP_DENSE_SOLVER_PROX_SOCP_HPP
#define PROXSUITE_PROXQP_DENSE_SOLVER_PROX_SOCP_HPP

#include <proxsuite/linalg/dense/core.hpp>
#include <proxsuite/linalg/veg/vec.hpp>
#include <proxsuite/proxqp/dense/workspace_prox_socp.hpp>
#include <proxsuite/proxqp/dense/preconditioner/ruiz_socp.hpp>
#include <proxsuite/proxqp/sparse/preconditioner/identity.hpp>
#include <iostream>
#include <iomanip> 

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
void print_setup_header_socp(const ProxSocpSettings<T>& settings,ProxSocpResults<T>& results, const SocpModel<T>& model){

  print_line();
  std::cout  <<"                           ProxSOCP  -  A Proximal SOCP Solver\n"
             <<"               (c) Antoine Bambade, Adrien Taylor, Justin Carpentier\n"
             <<"                                Inria Paris 2022        \n"
          << std::endl;
  print_line();

  // Print variables and constraints
  std::cout << "problem:  " << std::noshowpos <<std::endl;
  std::cout << "          variables n = " << model.dim << ", equality constraints n_eq = " << model.n_eq <<  ",\n" <<
  "          linear cone constraints n_in = "<< model.n_in << ", second order cone constraints n_soc = " << model.n_soc << ",\n" << std::endl;

  // Print Settings
  std::cout << "settings: " << std::endl;
  std::cout  <<"          backend = dense," << std::endl;
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
void primal_dual_socp_residual(
					   const SocpModel<T>& model,
                       ProxSocpResults<T>& results,
                       ProxSocpWorkspace<T>& work,
                       preconditioner::RuizSocpEquilibration<T>& ruiz,
                       T& primal_feasibility_lhs,
                       T& primal_feasibility_rhs,
					   T& dual_feasibility_lhs,
					   T& dual_feasibility_rhs_0,
					   T& dual_feasibility_rhs_1){
    // dual

	isize n = model.dim;
    isize n_eq = model.n_eq;
    isize n_in = model.n_in;
	work.primal_residual.setZero();
	work.dw.setZero();
	work.dual_residual_scaled = work.g_scaled;

	{
		work.dw.head(n) = work.H_scaled.template selfadjointView<Eigen::Lower>() * results.x;
		work.dual_residual_scaled += work.dw.head(n) ; // contains now scaled(g+Hx)

		ruiz.unscale_dual_residual_in_place({proxqp::from_eigen, work.dw.head(n)});
		dual_feasibility_rhs_0 = infty_norm(work.dw.head(n)); // ||unscaled(Hx)||

		work.dw.head(n) = work.A_scaled.transpose() * results.y;
		work.dual_residual_scaled += work.dw.head(n) ; // contains now scaled(g+Hx+ATy)

		ruiz.unscale_dual_residual_in_place({proxqp::from_eigen, work.dw.head(n) });
		dual_feasibility_rhs_1 = infty_norm(work.dw.head(n) ); // ||unscaled(ATy)||
	}


	ruiz.unscale_dual_residual_in_place(
			{proxqp::from_eigen, work.dual_residual_scaled}); // contains now unscaled(Hx+g+ATy)
	dual_feasibility_lhs = infty_norm(work.dual_residual_scaled); // ||unscaled(Hx+g+ATy)||
	ruiz.scale_dual_residual_in_place({proxqp::from_eigen, work.dual_residual_scaled});// ||scaled(Hx+g+ATy)||

    // primal 
	auto l = model.l;
	auto u = model.u;

	work.primal_residual = work.A_scaled * results.x;

	ruiz.unscale_primal_residual_in_place(
			{proxqp::from_eigen, work.primal_residual});
	primal_feasibility_rhs = infty_norm(work.primal_residual); // ||unscaled(Ax)||
	work.dw.segment(n+n_eq,n_in) =
			proxsuite::proxqp::dense::positive_part(work.primal_residual.segment(n_eq,n_in) - u.segment(n_eq,n_in)) +
			proxsuite::proxqp::dense::negative_part(work.primal_residual.segment(n_eq,n_in) - l);
	work.primal_residual.head(n_eq) -= u.head(n_eq);

	T primal_feasibility_eq_lhs = infty_norm(work.primal_residual.head(n_eq));
	T primal_feasibility_in_lhs = infty_norm(work.dw.segment(n+n_eq,n_in));
	primal_feasibility_lhs =
			std::max(primal_feasibility_eq_lhs, primal_feasibility_in_lhs);

	T pri_cone = 0;
    isize j = 0;
	isize n_cone = model.dims.rows();
	{
		isize m = n+n_eq+n_in;
		work.dw.tail(model.n_soc) = work.primal_residual.tail(model.n_soc) - u.tail(model.n_soc);
		for (isize it = 0; it < n_cone; ++it) { 
			isize dim_cone = model.dims[it];
			
			T cone_error = std::max(work.dw.segment(m+j+1,dim_cone-1).norm() - work.dw[m+j],0.);
			pri_cone = std::max(pri_cone, cone_error);
			j+=dim_cone;
		}
	}
	// scaled Ax - b for equality and scaled Ax pour ce qui reste
	ruiz.scale_primal_residual_in_place(
			{proxqp::from_eigen, work.primal_residual});

	primal_feasibility_lhs = std::max(primal_feasibility_lhs,pri_cone);

}


template<typename T>
void projection_onto_cones(ProxSocpWorkspace<T>& work,
						   SocpModel<T>& model,
						   ProxSocpResults<T>& results){
	isize m_ = model.dim+model.n_eq+model.n_in;
	work.dw.setZero();
    work.dw.segment(model.dim+model.n_eq,model.n_in) = results.z.head(model.n_in) + results.info.mu_in * results.y.segment(model.n_eq,model.n_in);
    work.dw.tail(model.n_soc) = results.z.tail(model.n_soc) + results.info.mu_soc * results.y.tail(model.n_soc);
    // projection on linear cone
    work.z_hat.head(model.n_in) = work.dw.segment(model.dim+model.n_eq,model.n_in)+ proxsuite::proxqp::dense::positive_part(work.l_scaled - work.dw.segment(model.dim+model.n_eq,model.n_in))+proxsuite::proxqp::dense::negative_part(-work.dw.segment(model.dim+model.n_eq,model.n_in)+work.u_scaled.segment(model.n_eq,model.n_in));//proxsuite::proxqp::dense::negative_part(work.internal.u_scaled.segment(model.n_eq,model.n_in)-tmp_hat.segment(model.n_eq,model.n_in));
    work.z_hat.tail(model.n_soc) = work.dw.tail(model.n_soc);
	// project over all cones
    isize j = 0;
	isize n_cone = model.dims.rows();
	for (isize it = 0; it < n_cone; ++it) { 
		isize dim_cone = model.dims[it];
		T aux_lhs_part = work.dw.segment(m_+j+1,dim_cone-1).norm();
		T aux_rhs_part = work.dw[m_+j];
		T mean = (aux_lhs_part + aux_rhs_part) * 0.5;
		if (aux_lhs_part <= -aux_rhs_part){
			work.z_hat.segment(model.n_in+j,dim_cone).setZero();
		} else if (aux_lhs_part > std::abs(aux_rhs_part)){
			T scaling_factor = mean / aux_lhs_part ;
			work.z_hat[model.n_in+j] = mean;
			work.z_hat.segment(model.n_in+j+1,dim_cone-1) *= scaling_factor;
		}
		j+=dim_cone;
	}
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
prox_socp_setup_factorization(ProxSocpWorkspace<T>& work,
                    const SocpModel<T>& model,
                    ProxSocpResults<T>& results)
{

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut,
    work.ldl_stack.as_mut(),
  };
  isize m = model.n_eq+model.n_in+model.n_soc;
  work.kkt.topLeftCorner(model.dim, model.dim) = work.H_scaled;
  work.kkt.topLeftCorner(model.dim, model.dim).diagonal().array() +=
    results.info.rho;
  work.kkt.block(0, model.dim, model.dim,m) =
    work.A_scaled.transpose();
  work.kkt.block(model.dim, 0, m, model.dim) = work.A_scaled;
  //work.kkt.bottomRightCorner(m, m).setZero();
  work.kkt.diagonal()
    .segment(model.dim, model.n_eq)
    .setConstant(-results.info.mu_eq);
  work.kkt.diagonal()
    .segment(model.dim+model.n_eq, model.n_in)
    .setConstant(-results.info.mu_in);
  work.kkt.diagonal()
    .segment(model.dim+model.n_eq+model.n_in, model.n_soc)
    .setConstant(-results.info.mu_soc);

  work.ldl.factorize(work.kkt, stack);
}

template<typename T>
void
socp_refactorize(const SocpModel<T>& model,
            ProxSocpResults<T>& results,
            ProxSocpWorkspace<T>& work,
            T rho_new)
{

  if (!work.constraints_changed && rho_new == results.info.rho) {
    return;
  }

  work.dw.setZero();
  work.kkt.diagonal().head(model.dim).array() +=
    rho_new - results.info.rho;
  work.kkt.diagonal().segment(model.dim, model.n_eq).array() =
    -results.info.mu_eq;
  work.kkt.diagonal().segment(model.dim+model.n_eq, model.n_in).array() =
    -results.info.mu_in;
  work.kkt.diagonal().segment(model.dim+model.n_eq+model.n_in, model.n_soc).array() =
    -results.info.mu_soc;

  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
  };
  work.ldl.factorize(work.kkt, stack);

  work.constraints_changed = false;

}

template<typename T>
void
socp_iterative_residual(const SocpModel<T>& model,
                   ProxSocpResults<T>& results,
                   ProxSocpWorkspace<T>& work,
                   isize inner_pb_dim)
{
  isize m = model.n_eq+model.n_in+model.n_soc;
  work.err = work.rhs;

  work.err.head(model.dim).noalias() -=
    work.H_scaled.template selfadjointView<Eigen::Lower>() *
    work.dw.head(model.dim);
  work.err.head(model.dim) -=
    results.info.rho * work.dw.head(model.dim);

  // PERF: fuse {A, C}_scaled multiplication operations
  work.err.head(model.dim).noalias() -=
    work.A_scaled.transpose() *
    work.dw.tail(m);
  work.err.tail(m).noalias() -=
    work.A_scaled * work.dw.head(model.dim);
  work.err.segment(model.dim, model.n_eq) +=
    work.dw.segment(model.dim, model.n_eq) * results.info.mu_eq;
  work.err.segment(model.dim+model.n_eq, model.n_in) +=
    work.dw.segment(model.dim+model.n_eq, model.n_in) * results.info.mu_in;
  work.err.tail(model.n_soc) +=
    work.dw.tail(model.n_soc) * results.info.mu_soc;
}

template<typename T>
void
socp_iterative_solve_with_permut_fact( //
  const ProxSocpSettings<T>& settings,
  const SocpModel<T>& model,
  ProxSocpResults<T>& results,
  ProxSocpWorkspace<T>& work,
  T eps,
  isize inner_pb_dim)
{

  work.err.setZero();
  i32 it = 0;
  i32 it_stability = 0;

  work.dw = work.rhs;
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
  };
  work.ldl.solve_in_place(work.dw, stack);
  socp_iterative_residual(model, results, work, inner_pb_dim);

  ++it;
  T preverr = infty_norm(work.err);
  //to put in debuger mode
  //if (settings.verbose) {
  //        std::cout << "infty_norm(res) " <<preverr
  //                                                << std::endl;
  //}
  //std::cout << "d " << work.ldl.d() << std::endl;
  
  while (infty_norm(work.err) >= eps) {

    if (it >= settings.nb_iterative_refinement) {
      break;
    }

    ++it;
    work.ldl.solve_in_place(work.err, stack);
    work.dw += work.err;

    work.err.setZero();
    socp_iterative_residual(model, results, work, inner_pb_dim);

    if (infty_norm(work.err) > preverr) {
      it_stability += 1;

    } else {
      it_stability = 0;
    }
    if (it_stability == 2) {
      break;
    }
    preverr = infty_norm(work.err);
    // to put in debug mode
    //if (settings.verbose) {
    //        std::cout << "infty_norm(res) "
    //                                                <<
    //infty_norm(work.err) << std::endl;
    //}
    
  }

  if (infty_norm(work.err) >=
      std::max(eps, settings.eps_refact)) {
    socp_refactorize(model, results, work, results.info.rho);
    it = 0;
    it_stability = 0;

    work.dw = work.rhs;
    work.ldl.solve_in_place(work.dw, stack);

    socp_iterative_residual(model, results, work, inner_pb_dim);

    preverr = infty_norm(work.err);
    ++it;
    // to put in debug mode
    //if (settings.verbose) {
    //        std::cout << "infty_norm(res) "
    //                                                <<
    //infty_norm(work.err) << std::endl;
    //}
    
    while (infty_norm(work.err) >= eps) {

      if (it >= settings.nb_iterative_refinement) {
        break;
      }
      ++it;
      work.ldl.solve_in_place(work.err, stack);
      work.dw += work.err;

      work.err.setZero();
      socp_iterative_residual(model, results, work, inner_pb_dim);

      if (infty_norm(work.err) > preverr) {
        it_stability += 1;

      } else {
        it_stability = 0;
      }
      if (it_stability == 2) {
        break;
      }
      preverr = infty_norm(work.err);
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
          ProxSocpResults<T>& results,
          ProxSocpWorkspace<T>& work,
		  T mu_soc_new)
{
  proxsuite::linalg::veg::dynstack::DynStackMut stack{
    proxsuite::linalg::veg::from_slice_mut, work.ldl_stack.as_mut()
  };

  isize n = model.dim;
  isize n_eq = model.n_eq;
  isize n_in = model.n_in;
  isize n_soc = model.n_soc;

  if ((n_soc) == 0) {
    return;
  }

  work.dw.tail(n_soc).setConstant(results.info.mu_soc - mu_soc_new);

  {
    auto _indices = stack.make_new_for_overwrite(
      proxsuite::linalg::veg::Tag<isize>{}, n_soc);
    isize* indices = _indices.ptr_mut();
    for (isize k = 0; k < n_soc; ++k) {
      indices[k] = n + n_eq + n_in + k;
    }
    work.ldl.diagonal_update_clobber_indices(
      indices,n_soc, work.dw.tail(n_soc), stack);
  }

  work.constraints_changed = true;
}

template<typename T>
void
linear_system_solving(const ProxSocpSettings<T>& settings,
                                    const SocpModel<T>& model,
                                    ProxSocpResults<T>& results,
                                    ProxSocpWorkspace<T>& work,
                                    T eps = T(1.E-6))
{

  /* MUST BE
   *  dual_residual_scaled = Hx + rho * (x-x_prev) + A.T y + C.T z
   *  primal_residual_eq_scaled = Ax-b+mu_eq (y_prev-y)
   *  primal_residual_in_scaled_up = Cx-u+mu_in(z_prev)
   *  primal_residual_in_scaled_low = Cx-l+mu_in(z_prev)
   */

  work.dual_residual_scaled = work.g_scaled;
  work.dual_residual_scaled.noalias() += work.H_scaled.template selfadjointView<Eigen::Lower>() * results.x;
  work.dual_residual_scaled.noalias() += work.A_scaled.transpose() * results.y;
  work.primal_residual = work.A_scaled * results.x;
  
  work.primal_residual.head(model.n_eq) -= work.u_scaled.head(model.n_eq);
  isize m = model.n_in + model.n_eq + model.n_soc;
  work.rhs.setZero();
  work.dw.setZero();
  work.primal_residual.tail(model.n_in+model.n_soc) -= work.z_hat; // contains now scaled(Ax - ze) 
  work.primal_residual.tail(model.n_soc) -= work.u_scaled.tail(model.n_soc); // contains now A_soc×x - u_soc - z_soc 

  work.rhs.head(model.dim) = -work.dual_residual_scaled; // H×x + g + AT×y
  work.rhs.tail(m) = -work.primal_residual ; // n_eq first constraints contains already A_eq x - b scaled 

  socp_iterative_solve_with_permut_fact( //
    settings,
    model,
    results,
    work,
    eps,
    model.dim+m);
  
}


template <typename T>
void prox_socp_solve(
		ProxSocpResults<T>& results,
		SocpModel<T>& model,
		ProxSocpSettings<T> const& settings,
		ProxSocpWorkspace<T>& work,
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
		work.l_scaled = model.l;
		isize m = model.n_eq+model.n_in+model.n_soc;
		work.u_scaled.segment(model.n_eq,model.n_in) =
			(model.u.segment(model.n_eq,model.n_in).array() <= T(1.E20))
			.select(model.u.segment(model.n_eq,model.n_in),
					Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() +
						T(1.E20));
		work.l_scaled =
			(model.l.array() >= T(-1.E20))
			.select(model.l,
					Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() -
						T(1.E20));
		prox_socp_setup_equilibration(work, settings, ruiz, false); // reuse previous equilibration
		prox_socp_setup_factorization(work, model, results);
	}else{
		prox_socp_setup_factorization(work, model, results);
	}
	if (settings.verbose){
		print_setup_header_socp(settings,results, model);
	}
	isize m = model.n_eq+model.n_in+model.n_soc;
	const T machine_eps = std::numeric_limits<T>::epsilon();
	T fact_mean(0);
	T fact(0);
	isize n_mean(0);
	isize freq_mu_update(settings.check_termination);
	T aux_fact(0);
	T new_mu_soc = results.info.mu_soc;
	T new_mu_soc_inv = results.info.mu_soc_inv;
	for (isize iter = 0; iter < settings.max_iter; ++iter) {
		results.info.iter += 1;

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
			T tmp_ = fact_mean / std::max(n_mean,isize(1));
			std::cout << "\033[1;32m[iteration " << iter + 1  << "]\033[0m" << std::endl;
			std::cout << std::scientific << std::setw(2) << std::setprecision(2) << 
			"| primal residual=" << primal_feasibility_lhs << "| dual residual=" << dual_feasibility_lhs << " | mu_soc=" << results.info.mu_soc << " | rho=" << results.info.rho << std::endl;
			results.info.pri_res = primal_feasibility_lhs;
			results.info.dua_res = dual_feasibility_lhs;
			ruiz.scale_primal_in_place(VectorViewMut<T>{from_eigen, results.x});
			ruiz.scale_dual_in_place(VectorViewMut<T>{from_eigen, results.y});
		}
		if (is_primal_feasible(primal_feasibility_lhs) &&
			is_dual_feasible(dual_feasibility_lhs)) {
			break;
		}	

		projection_onto_cones( work,
							   model,
							   results);
		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

		// Linear solving step
	
		///
		//
		// [ H + rho I    AeqT    A_inT	     A_socT]      -[ H×x + g + AT×y               ]
		// [ A_eq      -µ_eq I      0            0 ]      -[ A_eq×x -  b                  ]
		// [ A_in         0        -µ_in I       0 ]    = -[ A_in×x -  z_in               ]
		// [ A_soc        0         0     -µ_soc I ] dw = -[ A_soc×x - u_soc - z_soc      ]

		linear_system_solving(settings,
                            model,
                            results,
                            work);
	    results.z.head(model.n_in) = work.z_hat.head(model.n_in) + settings.tau * results.info.mu_in * work.dw.segment(model.dim+model.n_eq,model.n_in);
        results.z.tail(model.n_soc) = work.z_hat.tail(model.n_soc) + settings.tau * results.info.mu_soc * work.dw.tail(model.n_soc);
        results.x += work.dw.head(model.dim);
        results.y += work.dw.tail(m);
		T primal_feasibility_lhs_new(0);
		T dual_feasibility_lhs_new(0);
		primal_dual_socp_residual(
					   model,
                       results,
                       work,
                       ruiz,
                       primal_feasibility_lhs_new,
                       primal_feasibility_rhs,
					   dual_feasibility_lhs_new,
					   dual_feasibility_rhs_0,
					   dual_feasibility_rhs_1);
		if (is_primal_feasible(primal_feasibility_lhs_new) &&
			is_dual_feasible(dual_feasibility_lhs_new)) {
			//std::cout << "primal_feasibility_lhs_new " << primal_feasibility_lhs_new << " dual_feasibility_lhs_new " << dual_feasibility_lhs_new << std::endl;
			break;
		}

		// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
		// mu update
		
		fact = (primal_feasibility_lhs_new ) / (primal_feasibility_lhs  + machine_eps);
		fact_mean += fact;
		n_mean+=1;

		if (iter % freq_mu_update == 0 && iter >=freq_mu_update){
			
			aux_fact = fact_mean / n_mean;
			if (aux_fact>= settings.mu_update_fact_bound){
					new_mu_soc = 0.1;
					new_mu_soc_inv = 10.;
					fact_mean = 0;
					n_mean = 0;
			}

			freq_mu_update = std::max(roundUp(iter,settings.check_termination),settings.check_termination);
		}

		// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
		
		if (results.info.mu_soc != new_mu_soc) {
			{ ++results.info.mu_updates; }
			socp_mu_update(model,
					       results,
					       work,
					       new_mu_soc);
		}

		results.info.mu_soc = new_mu_soc;
		results.info.mu_soc_inv = new_mu_soc_inv;
	}

	ruiz.unscale_primal_in_place(VectorViewMut<T>{ from_eigen, results.x });
	ruiz.unscale_dual_in_place(
		VectorViewMut<T>{ from_eigen, results.y });
	
	//std::cout << " dual " << infty_norm(model.H * results.x + model.g + model.A.transpose() * results.y ) << std::endl;
	//std::cout << "x " << results.x << std::endl;
	//std::cout << "y " << results.y <<std::endl;
	//std::cout<< "g "  << model.g << std::endl;
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
void prox_socp_update_proximal_parameters(
		ProxSocpResults<T>& results,
		ProxSocpWorkspace<T>& work,
		std::optional<T> rho_new,
		std::optional<T> mu_eq_new,
		std::optional<T> mu_in_new,
		std::optional<T> mu_soc_new) {

	if (rho_new != std::nullopt) {
		results.info.rho = rho_new.value();
		work.proximal_parameter_update=true;
	}
	if (mu_eq_new != std::nullopt) {
		results.info.mu_eq = mu_eq_new.value();
		results.info.mu_eq_inv = T(1) / results.info.mu_eq;
		work.proximal_parameter_update=true;
	}
	if (mu_in_new != std::nullopt) {
		results.info.mu_in = mu_in_new.value();
		results.info.mu_in_inv = T(1) / results.info.mu_in;
		work.proximal_parameter_update=true;
	}
	if (mu_soc_new != std::nullopt) {
		results.info.mu_soc = mu_soc_new.value();
		results.info.mu_soc_inv = T(1) / results.info.mu_soc;
		work.proximal_parameter_update=true;
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

template<typename T>
void
prox_socp_setup_equilibration(ProxSocpWorkspace<T>& work,
                    const ProxSocpSettings<T>& settings,
                    preconditioner::RuizSocpEquilibration<T>& ruiz,
                    bool execute_preconditioner)
{
  
  dense::SocpViewMut<T> socp_scaled{
    { from_eigen, work.H_scaled }, { from_eigen, work.g_scaled },
    { from_eigen, work.A_scaled }, { from_eigen, work.l_scaled },
    { from_eigen, work.u_scaled }
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
void prox_socp_setup(
		const SparseMat<T>& H,
		const VecRef<T>g,
		const SparseMat<T>& A,
		const VecRef<T> u,
		const VecRef<T> l,
		ProxSocpResults<T>& results,
		SocpModel<T>& model,
		ProxSocpWorkspace<T>& work,
		const ProxSocpSettings<T>& settings,
		preconditioner::RuizSocpEquilibration<T>& ruiz,
		PreconditionerStatus& preconditioner_status) {
  // for the moment no initial guess
  results.cleanup();
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
  model.H = Eigen::
      Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
        H);
   // else model.H remains initialzed to a matrix with zero elements
  model.g = g;

  model.A = Eigen::
      Matrix<T, Eigen::Dynamic, Eigen::Dynamic, to_eigen_layout(rowmajor)>(
        A);
  // else model.A remains initialized to a matrix with zero elements or zero
    // shape
  model.u = u;
   // else model.u remains initialized to a matrix with zero elements or zero
    // shape
  model.l = l;
  // else model.l remains initialized to a matrix with zero elements or zero
    // shape

  work.H_scaled = model.H;
  work.g_scaled = model.g;
  work.A_scaled = model.A;

  work.u_scaled = model.u;
  work.l_scaled = model.l;
  isize m = model.n_eq+model.n_in+model.n_soc;
  work.u_scaled.segment(model.n_eq,model.n_in) =
    (model.u.segment(model.n_eq,model.n_in).array() <= T(1.E20))
      .select(model.u.segment(model.n_eq,model.n_in),
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() +
                T(1.E20));
  work.l_scaled =
    (model.l.array() >= T(-1.E20))
      .select(model.l,
              Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(model.n_in).array() -
                T(1.E20));

  work.primal_feasibility_rhs_1_in_u = infty_norm(work.u_scaled);
  work.primal_feasibility_rhs_1_in_l = infty_norm(work.l_scaled);
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
	dense::ProxSocpResults<T> results;
	dense::ProxSocpSettings<T> settings;
	dense::SocpModel<T> model;
	dense::ProxSocpWorkspace<T> work;
    preconditioner::RuizSocpEquilibration<T> ruiz;

	SOCP(isize _dim, isize _m, isize _n_eq, isize _n_in, Eigen::Matrix<isize, Eigen::Dynamic, 1> dims_)
			: results(_dim, _m, _n_eq),
				settings(),
				model(_dim,_n_eq,_n_in, dims_),
				work(_dim, _n_eq, _n_in, dims_),ruiz(_dim,_m,_n_eq,_n_in,1e-3,10) {
			work.timer.stop();
				}

	void init(
			const SparseMat<T>& H,
			const VecRef<T> g,
			const SparseMat<T>& A,
			const VecRef<T> u,
			const VecRef<T> l,
			bool compute_preconditioner_=true,
			std::optional<T> rho = std::nullopt,
			std::optional<T> mu_eq = std::nullopt,
			std::optional<T> mu_in = std::nullopt,
			std::optional<T> mu_soc = std::nullopt) {
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
		prox_socp_update_proximal_parameters(results, work, rho, mu_eq, mu_in,mu_soc);

		prox_socp_setup(
                H,g,A,u,l,
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
		//std::cout << " dual " << infty_norm(model.H * results.x + model.g + model.A.transpose() * results.y ) << std::endl;
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
