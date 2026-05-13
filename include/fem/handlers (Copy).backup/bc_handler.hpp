#pragma once

#include <Eigen/SparseCore>
#include <functional>
#include <cassert>
#include <span>

#include "util/compatibility.hpp"

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
	template<typename Dof_type>
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

		//cache boundary dof indices and which BC they belong to
		std::vector<uint64_t> boundary_dofs;
		std::vector<uint64_t> bc_idx;
		uint64_t compressed_dof_size=0;

		//call this whenever the compressed dof list changes or the list of boundary conditions are updated
		template<typename Container_t>
		inline void cache(const Container_t& dofs) {cache(as_span(dofs));}

		void cache(std::span<const DOF_t> dofs) {
			boundary_dofs.clear();
			bc_idx.clear();
			compressed_dof_size = dofs.size();

			for (size_t r=0; r<dofs.size(); ++r) {
				const DOF_t dof = dofs[r];
				for (uint64_t b_idx=0; b_idx<essential_bcs.size(); ++b_idx) {
					const auto& bc = essential_bcs[b_idx];
					if (bc.pred(dof)) {
						boundary_dofs.push_back(r);
						bc_idx.push_back(b_idx);
						break;
					}
				}
			}
		}

		//add boundary condition
		template<typename Pred, typename Fun>
		inline void add_essential(Pred&& pred, Fun&& fun) {
			essential_bcs.emplace_back(std::forward<Pred>(pred), std::forward<Fun>(fun));
		}

		//apply all dirichlet BCs to the given matrix and vector.
		//the active BCs in the correct order must also be supplied
		inline void apply(SpMat_t& mat, Vec_t& rhs, const std::vector<DOF_t>& dofs) const {
			apply(mat,rhs,as_span(dofs));
		}

		void apply(SpMat_t& mat, Vec_t& rhs, std::span<const DOF_t> dofs) const {
			assert(mat.rows()  == rhs.size());
			assert(mat.rows()  == static_cast<int>(dofs.size()));
			assert(dofs.size() == compressed_dof_size);
			assert(mat.isCompressed());

			if (essential_bcs.empty()) {return;}

			const auto outer_ptr = mat.outerIndexPtr();
			const auto inner_ptr = mat.innerIndexPtr();
			const auto val_ptr   = mat.valuePtr();

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (size_t r=0; r<boundary_dofs.size(); ++r) {
				const int row_idx = static_cast<int>(boundary_dofs[r]);
				const DOF_t dof   = dofs[boundary_dofs[r]];
				const auto& bc    = essential_bcs[bc_idx[r]];
				
				//set row to the identity
				const int r_start = outer_ptr[row_idx];
				const int r_end   = outer_ptr[row_idx+1];

				#pragma omp simd
				for (int idx=r_start; idx<r_end; ++idx) {
					val_ptr[idx] = (inner_ptr[idx] == row_idx) ? 1.0 : 0.0;
				}

				//set rhs
				rhs[row_idx] = bc.fun(dof);
			}
		}

		//apply all dirichlet BCs to the given vector.
		//can be used for the rhs or for the result of a matrix-vector multiply
		template<typename ContainerA_t, typename ContainerB_t>
		inline void apply(ContainerA_t& vec, const ContainerB_t& dofs) const {
			apply(as_span(vec), as_span(dofs));
		}

		void apply(std::span<double> vec, std::span<const DOF_t> dofs) const {
			assert(dofs.size() == compressed_dof_size);
			assert(vec.size()  == compressed_dof_size);

			if (essential_bcs.empty()) {return;}

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (size_t r=0; r<boundary_dofs.size(); ++r) {
				const uint64_t row_idx = boundary_dofs[r];
				const DOF_t dof        = dofs[row_idx];
				const auto& bc         = essential_bcs[bc_idx[r]];
				vec[row_idx]           = bc.fun(dof);
			}
		}

		//set copy the values of x to y that correspond to a boundary dof
		//this is useful when computing y=Ax without forming A but when the BCs
		//still need to be applied
		
		template<typename ContainerA_t, typename ContainerB_t>
		inline void apply_matvec(ContainerA_t& y, const ContainerB_t& x) const {
			apply_matvec(as_span(y), as_span(x));
		}

		void apply_matvec(std::span<double> y, std::span<const double> x) const {
			assert(y.size()==x.size());
			assert(y.size()==compressed_dof_size);
			
			#pragma omp simd
			for (size_t i=0; i<boundary_dofs.size(); ++i) {
				const uint64_t idx = boundary_dofs[i];
				y[idx] = x[idx];
			}
		}
	};


	
}