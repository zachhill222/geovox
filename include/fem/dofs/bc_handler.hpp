#pragma once

#include <Eigen/SparseCore>
#include <functional>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//a class to help set essential boundary conditions for problems
	//for a problem Ax=b with row x[i] corresponding to a dof with an essential boundary condition
	//we can simply set the i-th row of A to the corresponding identity row and the i-th value of b
	//to the prescribed boundary condition. this does not change krylov spaces for iterative methods.
	//we can also row reduce the i-th column of A and subtract the corresponding terms from each row of b.

	//this class is a natural place to store the boundary condition logic. each boundary condition can be stored as
	//a predicate/function pair. The predicate determines which dofs qualify for the BC and the function determines
	//its value. Both should take the same DOF as their sole argument.

	//the natural BCs are already incorporated into the matrix A and the periodic BCs are already incorporated
	//into the DOFs.
	template <typename Dof_type>
	struct BCHandler
	{
		using DOF_t    = Dof_type;
		using SpMat_t  = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
		using Vec_t    = Eigen::VectorXd;

		struct EssentialBC
		{
			std::function<bool(DOF_t)>   pred;
			std::function<double(DOF_t)> fun;

			template<typename Pred, typename Fun>
			EssentialBC(Pred&& pred_, Fun&& fun_) : 
				pred(std::forward<Pred>(pred_)), 
				fun(std::forward<Fun>(fun_))
				{
					static_assert(std::is_same_v<std::invoke_result_t<Pred,DOF_t>, bool>,
						"EssentialBC - first argument (pred) must be callable with DOF_t and return bool.");

					static_assert(std::is_same_v<std::invoke_result_t<Fun,DOF_t>, double>,
						"EssentialBC - second argument (fun) must be callable with DOF_t and return double.");
				}
		};


		//store BCs
		std::vector<EssentialBC> essential_bcs;

		template<typename Pred, typename Fun>
		inline void add_essential(Pred&& pred, Fun&& fun) {
			essential_bcs.emplace_back(std::forward<Pred>(pred), std::forward<Fun>(fun));
		}

		//apply all dirichlet BCs to the given matrix and vector.
		//the active BCs in the correct order must also be supplied
		void apply(SpMat_t& mat, Vec_t& rhs, const std::vector<DOF_t>& dofs) const {
			assert(mat.rows() == rhs.size());
			assert(mat.rows() == dofs.size());
			assert(A.isCompressed());

			const auto outer_ptr = mat.outerIndexPtr();
			const auto inner_ptr = mat.innerIndexPtr();
			const auto val_ptr   = mat.valuePtr();

			#ifdef _OPENMP
			#pragma omp parallel for //note pred and fun must be thread safe
			#endif
			for (size_t r=0; r<dofs.size(); ++r) {
				const DOF_t dof = dofs[r];
				for (const auto& bc : essential_bcs) {
					if (bc.pred(dof)) {
						//set row to the identity
						const int r_start = outer_ptr[r];
						const int r_end   = outer_ptr[r+1];

						#pragma omp simd
						for (int idx=r_start; idx<r_end; ++idx) {
							val_ptr[idx] = (inner_ptr[idx] == static_cast<int>(r)) ? 1.0 : 0.0;
						}

						//set rhs
						rhs[r] = bc.fun(dof);
						break; //at most one BC per dof
					}
				}
			}
		}
	};
}