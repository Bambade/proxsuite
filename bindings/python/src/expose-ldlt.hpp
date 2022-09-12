//
// Copyright (c) 2022 INRIA
//
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <proxsuite/proxqp/dense/wrapper.hpp>
#include <proxsuite/proxqp/sparse/wrapper.hpp>

#include <Eigen/SparseCholesky>

namespace proxsuite {
namespace linalg {
using proxsuite::linalg::veg::isize;

namespace dense {

namespace python {

template <typename T>
void dense_iterative_solve( //
		proxqp::dense::VecRef<T> rhs,
		proxqp::dense::VecRefMut<T> sol,
		proxqp::dense::MatRef<T> mat,
		T eps,
		isize max_it,
    const bool verbose) {
	isize it = 0;
	sol = rhs;
	proxsuite::linalg::dense::Ldlt<T> ldl{};
  ldl.reserve_uninit(rhs.rows());
  proxsuite::linalg::veg::Vec<unsigned char> ldl_stack;
  ldl_stack.resize_for_overwrite(
    proxsuite::linalg::veg::dynstack::StackReq(

      proxsuite::linalg::dense::Ldlt<T>::factorize_req(rhs.rows()) |
      proxsuite::linalg::dense::Ldlt<T>::solve_in_place_req(rhs.rows()))
      .alloc_req());
	proxsuite::linalg::veg::dynstack::DynStackMut stack{
		proxsuite::linalg::veg::from_slice_mut, ldl_stack.as_mut()
	};
	ldl.factorize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(mat), stack);
	ldl.solve_in_place(sol, stack);

	auto res = (mat * sol - rhs).eval();
	while (proxsuite::proxqp::dense::infty_norm(res) >= eps) {
		it += 1;
    if (verbose){
      //std::cout << "it: " << it << "; err: " << proxsuite::proxqp::dense::infty_norm(res)<< std::endl;
    }
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res,stack);
		sol += res;
		res = (mat * sol - rhs);
	}
}

template<typename T>
void
DenseIterativeSolve(pybind11::module_ m)
{
  m.def(
    "dense_iterative_solve",&dense_iterative_solve<T>,
    "Function for solving a linear system using PROXQP dense linear solver directly "
    "with at most max_it step of iterative refinement (which are executed if the minimal accuracy eps"
	 "is not reached). A Ldlt factorization is realised in order to solve the system.",
    pybind11::arg_v("rhs", std::nullopt, "the rhs term of the linear system to solve."),
    pybind11::arg_v("sol", std::nullopt, "the solution of the linear system (changed in place)."),
    pybind11::arg_v(
      "mat", std::nullopt, "The matrix (symetric and invertible) in dense format to factorize in order to solve the linear system."),
    pybind11::arg_v("eps", T(1.E-6), "The minimal accuracy desired for the residual."),
    pybind11::arg_v(
      "mat_it", 5, "The maximum number of iterative refinement step."),
    pybind11::arg_v(
      "verbose", false, "verbose argument for printing iterative refinement steps with associated residual error."));
}

} // namespace python
} // namespace dense

namespace sparse {
namespace python {

template <typename T,typename I>
void sparse_iterative_solve( //
		proxqp::sparse::VecRef<T> rhs,
		proxqp::sparse::VecRefMut<T> sol,
		proxqp::sparse::SparseMat<T,I> mat,
		T eps,
		isize max_it,
    const bool verbose) {
	isize it = 0;
	sol = rhs;
	proxsuite::proxqp::sparse::Ldlt<T,I> ldl;
  ldl.reserve_uninit(rhs.rows());
  proxsuite::linalg::veg::Vec<unsigned char> ldl_stack;
  ldl_stack.resize_for_overwrite(
    proxsuite::linalg::veg::dynstack::StackReq(

      proxsuite::linalg::dense::Ldlt<T>::factorize_req(rhs.rows()) |
      proxsuite::linalg::dense::Ldlt<T>::solve_in_place_req(rhs.rows()))
      .alloc_req());
	proxsuite::linalg::veg::dynstack::DynStackMut stack{
		proxsuite::linalg::veg::from_slice_mut, ldl_stack.as_mut()
	};
	ldl.factorize(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(mat), stack);
	ldl.solve_in_place(sol, stack);

	auto res = (mat * sol - rhs).eval();
	while (proxsuite::proxqp::dense::infty_norm(res) >= eps) {
		it += 1;
    if (verbose){
      //std::cout << "it: " << it << "; err: " << proxsuite::proxqp::dense::infty_norm(res)<< std::endl;
    }
		if (it >= max_it) {
			break;
		}
		res = -res;
		ldl.solve_in_place(res,stack);
		sol += res;
		res = (mat * sol - rhs);
	}
}


template<typename T, typename I>
auto
to_eigen(proxsuite::linalg::sparse::MatRef<T, I> a) noexcept
  -> Eigen::Matrix<T, -1, -1>
{
  return a.to_eigen();
}
template<typename I>
auto
to_eigen_perm(proxsuite::linalg::veg::Slice<I> perm)
  -> Eigen::PermutationMatrix<-1, -1, I>
{
  Eigen::PermutationMatrix<-1, -1, I> perm_eigen;
  perm_eigen.indices().resize(perm.len());
  std::memmove( //
    perm_eigen.indices().data(),
    perm.ptr(),
    proxsuite::linalg::veg::usize(perm.len()) * sizeof(I));
  // copie perm.ptr() vers perm_eigen ...
  //  proxsuite::linalg::veg::usize(perm.len()) * sizeof(I) :taille de la zone à
  //  copier
  return perm_eigen;
}


template<typename T, typename I>
auto
reconstruct_with_perm(proxsuite::linalg::veg::Slice<I> perm_inv,
                      proxsuite::linalg::sparse::MatRef<T, I> ld)
  -> Eigen::Matrix<T, -1, -1, Eigen::ColMajor>
{
  using Mat = Eigen::Matrix<T, -1, -1, Eigen::ColMajor>;
  Mat ld_eigen = to_eigen(ld);
  auto perm_inv_eigen = to_eigen_perm(perm_inv);
  Mat l = ld_eigen.template triangularView<Eigen::UnitLower>();
  Mat d = ld_eigen.diagonal().asDiagonal();
  Mat ldlt = l * d * l.transpose();
  return perm_inv_eigen.inverse() * ldlt * perm_inv_eigen;
}


template <typename T,typename I>
void sparse_factorization( //
		proxqp::sparse::SparseMat<T,I> mat_) {
	
	proxsuite::proxqp::sparse::SparseMat<T, I> Mat = mat_.template triangularView<Eigen::Upper>();
	proxsuite::linalg::sparse::MatRef<T, I> mat = {proxsuite::linalg::sparse::from_eigen, Mat};
	proxsuite::proxqp::sparse::Ldlt<T,I> ldl;
	proxsuite::linalg::veg::Vec<proxsuite::linalg::veg::mem::byte> storage;
	isize n_tot = mat.nrows();
	isize nnz_tot = mat.nnz();
	proxsuite::linalg::veg::Vec<I> kkt_col_ptrs;
	proxsuite::linalg::veg::Vec<I> kkt_row_indices;
	proxsuite::linalg::veg::Vec<T> kkt_values;
	proxsuite::linalg::veg::Vec<I> kkt_nnz_counts;
	using namespace proxsuite::linalg::veg::dynstack;
	using namespace proxsuite::linalg::sparse::util;

	proxsuite::linalg::veg::Tag<I> itag; 
	proxsuite::linalg::veg::Tag<T> xtag; 

	using SR = StackReq;

	isize lnnz =0;
	{


		kkt_col_ptrs.resize_for_overwrite(n_tot + 1);
		kkt_row_indices.resize_for_overwrite(nnz_tot);
		kkt_values.resize_for_overwrite(nnz_tot);

		I* kktp = kkt_col_ptrs.ptr_mut();
		I* kkti = kkt_row_indices.ptr_mut();
		T* kktx = kkt_values.ptr_mut();

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

		insert_submatrix(mat, true);
		
		storage.resize_for_overwrite( //
				(StackReq::with_len(itag, n_tot) &
				proxsuite::linalg::sparse::factorize_symbolic_req( //
							itag,                                     //
							n_tot,                                    //
							nnz_tot,                                  //
							proxsuite::linalg::sparse::Ordering::amd))     //
						.alloc_req()                               //
		);

		ldl.col_ptrs.resize_for_overwrite(n_tot + 1);
		ldl.perm_inv.resize_for_overwrite(n_tot);

		auto stack = proxsuite::linalg::veg::dynstack::DynStackMut{
		proxsuite::linalg::veg::tags::from_slice_mut, storage.as_mut()
		};

		bool overflow = false;
		
		ldl.etree.resize_for_overwrite(n_tot);
		auto etree_ptr = ldl.etree.ptr_mut();

		using namespace proxsuite::linalg::veg::literals;
		auto kkt_sym = proxsuite::linalg::sparse::SymbolicMatRef<I>{
				proxsuite::linalg::sparse::from_raw_parts,
				n_tot,
				n_tot,
				nnz_tot,
				kkt_col_ptrs.ptr(),
				nullptr,
				kkt_row_indices.ptr(),
		};
		proxsuite::linalg::sparse::factorize_symbolic_non_zeros( //
				ldl.col_ptrs.ptr_mut() + 1,
				etree_ptr,
				ldl.perm_inv.ptr_mut(),
				static_cast<I const*>(nullptr),
				kkt_sym,
				stack);

		auto pcol_ptrs = ldl.col_ptrs.ptr_mut(); 
		pcol_ptrs[0] = I(0);//pcol_ptrs +1: pointor towards the nbr of non zero elts per column of the ldlt 
		// we need to compute its cumulative sum below to determine if there could be an overflow

		using proxsuite::linalg::veg::u64;
		u64 acc = 0;

		for (usize i = 0; i < usize(n_tot); ++i) {
			acc += u64(zero_extend(pcol_ptrs[i + 1]));
			if (acc != u64(I(acc))) {
				overflow = true;
			}
			pcol_ptrs[(i + 1)] = I(acc);
		}
		

		lnnz = isize(zero_extend(ldl.col_ptrs[n_tot]));

	}
#define PROX_SOCP_ALL_OF(...)                                                    \
proxsuite::linalg::veg::dynstack::StackReq::and_(proxsuite::linalg::veg::init_list(__VA_ARGS__))
#define PROX_SOCP_ANY_OF(...)                                                    \
proxsuite::linalg::veg::dynstack::StackReq::or_(proxsuite::linalg::veg::init_list(__VA_ARGS__))
		//  ? --> if
		auto refactorize_req = 
				 PROX_SOCP_ANY_OF({
									proxsuite::linalg::sparse::
											factorize_symbolic_req( // symbolic ldl
													itag,
													n_tot,
													nnz_tot,
													proxsuite::linalg::sparse::Ordering::user_provided),
									PROX_QP_ALL_OF({
											SR::with_len(xtag, n_tot), // diag
											proxsuite::linalg::sparse::
													factorize_numeric_req( // numeric ldl
															xtag,
															itag,
															n_tot,
															nnz_tot,
															proxsuite::linalg::sparse::Ordering::user_provided),
									}),
							});

		auto x_vec = [&](isize n) noexcept -> StackReq {
			return proxsuite::linalg::dense::temp_vec_req(xtag, n);
		};

		auto ldl_solve_in_place_req = PROX_QP_ALL_OF({
				x_vec(n_tot), // tmp
				x_vec(n_tot), // err
				x_vec(n_tot), // work
		});

		auto req = //
				PROX_SOCP_ALL_OF({
						SR::with_len(itag, n_tot),            // kkt nnz counts
						refactorize_req,
						ldl_solve_in_place_req,
										PROX_SOCP_ALL_OF({
																	SR::with_len(itag, n_tot), // perm
																	SR::with_len(itag, n_tot), // etree
																	SR::with_len(itag, n_tot), // ldl nnz counts
																	SR::with_len(itag, lnnz), // ldl row indices
																	SR::with_len(xtag, lnnz), // ldl values
															})
				});

	storage.resize_for_overwrite(req.alloc_req());
	auto stack = proxsuite::linalg::veg::dynstack::DynStackMut{
      proxsuite::linalg::veg::tags::from_slice_mut, storage.as_mut()
    };
	kkt_nnz_counts.resize_for_overwrite(n_tot);
	auto zx = proxsuite::linalg::sparse::util::zero_extend;// ?
	auto max_lnnz = isize(zx(ldl.col_ptrs[n_tot]));
	isize ldlt_ntot = n_tot;
	isize ldlt_lnnz = max_lnnz;

	ldl.nnz_counts.resize_for_overwrite(ldlt_ntot);
	ldl.row_indices.resize_for_overwrite(ldlt_lnnz);
	ldl.values.resize_for_overwrite(ldlt_lnnz);
	
	ldl.perm.resize_for_overwrite(ldlt_ntot);
	if (true) {
		// compute perm from perm_inv
		for (isize i = 0; i < n_tot; ++i) {
			ldl.perm[isize(zx(ldl.perm_inv[i]))] = I(i);
		}
	}

	////////////////////////
	
	////////////////////////********************** try another way
	 auto nnz =
      isize(proxsuite::linalg::sparse::util::zero_extend(kkt_col_ptrs[n_tot]));
	proxsuite::linalg::sparse::MatMut<T, I>  kkt = {
      proxsuite::linalg::sparse::from_raw_parts,
      n_tot,
      n_tot,
      nnz,
      kkt_col_ptrs.ptr_mut(),
      nullptr,
      kkt_row_indices.ptr_mut(),
      kkt_values.ptr_mut(),
    };
	//auto ldl_col_ptrs = ldl.col_ptrs.ptr_mut();
	bool do_ldlt = true;

	ldlt_ntot = do_ldlt ? n_tot : 0;
	auto _perm = stack.make_new_for_overwrite(itag, ldlt_ntot);
	I const* perm_inv = ldl.perm_inv.ptr_mut();
	I* perm = _perm.ptr_mut();

	if (do_ldlt) {
		// compute perm from perm_inv
		for (isize i = 0; i < n_tot; ++i) {
			perm[isize(zx(perm_inv[i]))] = I(i);
		}
	}

	// FOR DEBUG	
	//I* etree = ldl.etree.ptr_mut();
	//I* ldl_nnz_counts =  ldl.nnz_counts.ptr_mut();
	//I* ldl_row_indices = ldl.row_indices.ptr_mut();
	//T* ldl_values = ldl.values.ptr_mut();
	//proxsuite::linalg::sparse::MatMut<T, I> ld = {
	//		proxsuite::linalg::sparse::from_raw_parts,
	//		n_tot,
	//		n_tot,
	//		0,
	//		ldl_col_ptrs,
	//		do_ldlt ? ldl_nnz_counts : nullptr,
	//		ldl_row_indices,
	//		ldl_values,
	//};

    proxsuite::linalg::sparse::factorize_symbolic_non_zeros(
      ldl.nnz_counts.ptr_mut(),
      ldl.etree.ptr_mut(),
      ldl.perm_inv.ptr_mut(),
      ldl.perm.ptr_mut(),
      kkt.symbolic(),
      stack);

	proxsuite::linalg::sparse::factorize_numeric(
			ldl.values.ptr_mut(),
			ldl.row_indices.ptr_mut(),
			nullptr,//diag, // ?
			ldl.perm.ptr_mut(),
			ldl.col_ptrs.ptr(),
			ldl.etree.ptr_mut(),
			ldl.perm_inv.ptr_mut(),
			kkt.as_const(), // ?
			stack);

	// for debug
	//Eigen::Matrix<T, -1, -1, Eigen::ColMajor> a = reconstruct_with_perm(ldl.perm_inv.as_ref(), ld.as_const()) ;
	//assert((a -
    //     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,  Eigen::ColMajor>(mat)).norm() < T(1e-10) );

}


template <typename T>
void sparse_eigen_factorization( //
		Eigen::SparseMatrix<T> mat_) {
		
		 Eigen::SimplicialLDLT<Eigen::SparseMatrix<T>> solverLDLT;
		 solverLDLT.compute(mat_);
}


template<typename T,typename I>
void
SparseIterativeSolve(pybind11::module_ m)
{
  m.def(
    "dense_iterative_solve",&sparse_iterative_solve<T,I>,
    "Function for solving a linear system using PROXQP sparse linear solver directly "
    "with at most max_it step of iterative refinement (which are executed if the minimal accuracy eps"
	 "is not reached). A Ldlt factorization is realised in order to solve the system.",
    pybind11::arg_v("rhs", std::nullopt, "the rhs term of the linear system to solve."),
    pybind11::arg_v("sol", std::nullopt, "the solution of the linear system (changed in place)."),
    pybind11::arg_v(
      "mat", std::nullopt, "The matrix (symetric and invertible) in sparse format to factorize in order to solve the linear system."),
    pybind11::arg_v("eps", T(1.E-6), "The minimal accuracy desired for the residual."),
    pybind11::arg_v(
      "mat_it", 5, "The maximum number of iterative refinement step."),
    pybind11::arg_v(
      "verbose", false, "verbose argument for printing iterative refinement steps with associated residual error."));
}


template<typename T,typename I>
void
SparseFactorization(pybind11::module_ m)
{
  m.def(
    "sparse_factorization",&sparse_factorization<T,I>,
    "Function for solving a linear system using PROXQP sparse linear solver directly "
    "with at most max_it step of iterative refinement (which are executed if the minimal accuracy eps"
	 "is not reached). A Ldlt factorization is realised in order to solve the system.",
    pybind11::arg_v("mat", std::nullopt, "the matrix to factorize (should be symmetric and invertible)."));
}

template<typename T>
void
SparseEigenFactorization(pybind11::module_ m)
{
  m.def(
    "sparse_eigen_factorization",&sparse_eigen_factorization<T>,
    "Function for solving a linear system using PROXQP sparse linear solver directly "
    "with at most max_it step of iterative refinement (which are executed if the minimal accuracy eps"
	 "is not reached). A Ldlt factorization is realised in order to solve the system.",
    pybind11::arg_v("mat", std::nullopt, "the matrix to factorize (should be symmetric and invertible)."));
}

} // namespace python
} // namespace sparse

} // namespace linalg
} // namespace proxsuite
