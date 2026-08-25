#pragma once

#include "gutil.hpp"

#include "util/util.hpp"

#include <vector>
#include <span>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace GV {


	/////////////////////////////////////////////////////////////////////////////
	/// Track essential boundary conditions
	/////////////////////////////////////////////////////////////////////////////
	template<typename T>
	struct BcHandler {
		std::vector<T> 		vals{};
		std::vector<size_t> dof_numbers{};


		/// A factory method to assign homogeneous Dirichlet BCs to
		/// the dofs. Note that the order of the BCs are not deterministic,
		/// so it may be worthwhile to sort the dof_numbers afterwards.
		/// It may be useful to have a non-zero offset when working with a larger
		/// system involving several dofhandlers.
		template<typename Predicate, std::random_access_iterator I>
		[[nodiscard]] static BcHandler ComputeDirichletBC(I dof_begin, I dof_end, Predicate pred, const size_t offset=0) noexcept {
			BcHandler result;

			GUTIL_OMP(parallel)
			{
				std::vector<size_t> thread_dof_numbers;
				OmpIteratorRange    range(dof_begin, dof_end);
				for (I it=range.begin; it!=range.end; ++it) {
					if (pred(*it)) {thread_dof_numbers.push_back(offset + std::distance(dof_begin,it));}
				}

				GUTIL_OMP(critical)
				{
					result.dof_numbers.insert(result.dof_numbers.end(), 
						std::make_move_iterator(thread_dof_numbers.begin()),
						std::make_move_iterator(thread_dof_numbers.end()));
				}
			}

			result.vals.assign(result.dof_numbers.size(), T{0});
			return result;
		}
	};


	/////////////////////////////////////////////////////////////////////////////
	/// Given a matrix in CSR format, replace the specified rows with the corresponding
	/// identity rows. Note that it may be convenient to mix the index type of the matrix
	/// representation and the rows to replace.
	///
	/// Recall that CSR storage has an outer_ptr, inner_ptr and val_ptr where
	/// outer_ptr = [o0,o1,o2,...,oN] such that [outer_ptr[i], outer_ptr[i+1]) are the range
	/// of values in inner_ptr and val_ptr that correspond to row i. If A(i,j) is non-zero,
	/// then there is a k in [outer_ptr[i], outer_ptr[i+1]) such that inner_ptr[k] = j and val_ptr[k] = A(i,j).
	/////////////////////////////////////////////////////////////////////////////
	template<typename T, typename I1, typename I2>
	inline constexpr void csr_set_identity_row(const I1* outer_ptr, const I1* inner_ptr, T* val_ptr, std::span<const I2> rows) noexcept {
		GUTIL_ASSERT(outer_ptr && inner_ptr && val_ptr);

		GUTIL_OMP(parallel)
		{
			OmpIndexRange range(rows.size());

			for (size_t idx=range.begin; idx<range.end; ++idx) {
				const I1 r       = static_cast<I1>(rows[idx]);
				const I1 r_begin = outer_ptr[r];
				const I1 r_end   = outer_ptr[r+1];
				
				GUTIL_SIMD()
				for (I1 k=r_begin; k<r_end; ++k) {
					val_ptr[k] = (inner_ptr[k] == r) ? T{1} : T{0};
				}
			}
		}
	}


	/////////////////////////////////////////////////////////////////////////////
	/// Given a symmetric matrix in CSR format, set the specified rows and columns to the
	/// corresponding identity rows/columns. When handling the columns, it is
	/// best if the inner_ptr is sorted (within each row), but this may not be
	/// always guaranteed.
	///
	/// To avoid race conditions, we do this in two passes. Pass 1 zeros the columns
	/// while pass 2 sets the rows. 
	///
	/// This function only implements pass 1. Call csr_set_identity_row for pass 2.
	/////////////////////////////////////////////////////////////////////////////
	template<bool InnerSorted=false, typename T, typename I1, typename I2>
	inline constexpr void csr_set_symmetric_identity_row_col(const I1* outer_ptr, const I1* inner_ptr, T* val_ptr, std::span<const I2> rows) noexcept {
		GUTIL_ASSERT(outer_ptr && inner_ptr && val_ptr);

		//pass 1: handle the columns
		GUTIL_OMP(parallel)
		{
			OmpIndexRange range(rows.size());

			for (size_t idx=range.begin; idx<range.end; ++idx) {
				const I1 r       = static_cast<I1>(rows[idx]);
				const I1 r_begin = outer_ptr[r];
				const I1 r_end   = outer_ptr[r+1];

				for (I1 k=r_begin; k<r_end; ++k) {
					const I1 j       = inner_ptr[k]; if (j==r) {continue;}
					const I1 j_begin = outer_ptr[j];
					const I1 j_end   = outer_ptr[j+1];

					if constexpr (InnerSorted) {
						const I1* first = inner_ptr + j_begin;
						const I1* last  = inner_ptr + j_end;
						const I1* it    = std::lower_bound(first, last, r);
						if (it!=last && *it==r) {
							//note the pointer difference gives the index
							val_ptr[it - inner_ptr] = T{0};
						}
					}
					else {
						const I1* first = inner_ptr + j_begin;
						const I1* last  = inner_ptr + j_end;
						const I1* it    = std::find(first, last, r);
						if (it!=last) {
							GUTIL_ASSERT(*it==r);
							//note the pointer difference gives the index
							val_ptr[it - inner_ptr] = T{0};
						}
					}
				}
			}
		}

		//pass 2: set the rows
		csr_set_identity_row(outer_ptr, inner_ptr, val_ptr, rows);
	}


	/////////////////////////////////////////////////////////////////////////////
	/// Given a symmetric matrix in CSR format and vectors x and b, eliminate the
	/// specified values (rows) by row-reduction. Note that the known values of x
	/// must be set ahead of time.
	///
	/// We do this in three passes. Pass 1 updates b(i) with b(i) - sum_j A(i,j)*x(j)
	/// for all i not in the provided rows and sets b(r)=x(r) for all r in the
	/// provided rows. Passes 2 and 3 update the matrix A.
	/////////////////////////////////////////////////////////////////////////////
	template<bool InnerSorted=false, typename T, typename I1, typename I2>
	inline constexpr void csr_row_reduce_symmetric(const I1* outer_ptr, const I1* inner_ptr, T* val_ptr, std::span<const T> x, std::span<T> b, std::span<const I2> rows) noexcept {
		GUTIL_ASSERT(outer_ptr && inner_ptr && val_ptr);
		GUTIL_ASSERT(x.size()==b.size());

		std::vector<T> x_reduced(rows.size());
		GUTIL_OMP(parallel)
		{
			OmpIndexRange range(rows.size());

			//pass 1: subtract
			for (size_t idx=range.begin; idx<range.end; ++idx) {
				const I1 r       = static_cast<I1>(rows[idx]);
				const I1 r_begin = outer_ptr[r];
				const I1 r_end   = outer_ptr[r+1];
				const T  x_r     = x[r];

				for (I1 k=r_begin; k<r_end; ++k) {
					const I1 j = inner_ptr[k];
					if (j==r) {
						b[j] = x_r;
					}
					else {
						GUTIL_OMP(atomic) b[j] -= val_ptr[k]*x_r;
					}
				}
			}
		}

		//passes 2 and 3: set the rows and columns of A to the identity
		csr_set_symmetric_identity_row_col<InnerSorted>(outer_ptr, inner_ptr, val_ptr, rows);
	}



#ifdef EIGEN_MAJOR_VERSION
	///////////////////////////////////////////////////////////////////
	/// Convenience wrappers to call the CSR-level BC-elimination methods
	/// directly on Eigen::SparseMatrix objects. All require the matrix to
	/// already be compressed (via makeCompressed() / setFromTriplets()).
	///
	/// NOTE on InnerSorted: Eigen documents compressed-mode inner indices
	/// as sorted by increasing index, but its own source comments on
	/// collapseDuplicates() (used by setFromTriplets()) note this isn't
	/// strictly guaranteed on that specific path -- left as an explicit
	/// choice here rather than defaulted to true, since this codebase
	/// builds matrices via setFromTriplets() throughout.
	///////////////////////////////////////////////////////////////////

	template<typename T, int Options, typename StorageIndex, typename I2>
	inline void eigen_set_identity_row(Eigen::SparseMatrix<T,Options,StorageIndex>& mat, std::span<const I2> rows) {
		GUTIL_ASSERT(mat.isCompressed());
		csr_set_identity_row(mat.outerIndexPtr(), mat.innerIndexPtr(), mat.valuePtr(), rows);
	}

	template<bool InnerSorted=false, typename T, int Options, typename StorageIndex, typename I2>
	inline void eigen_set_symmetric_identity_row_col(Eigen::SparseMatrix<T,Options,StorageIndex>& mat, std::span<const I2> rows) {
		GUTIL_ASSERT(mat.isCompressed());
		csr_set_symmetric_identity_row_col<InnerSorted>(mat.outerIndexPtr(), mat.innerIndexPtr(), mat.valuePtr(), rows);
	}

	template<bool InnerSorted=false, typename T, int Options, typename StorageIndex, typename I2>
	inline void eigen_row_reduce_symmetric(Eigen::SparseMatrix<T,Options,StorageIndex>& mat,
	                                        std::span<const T> x, std::span<T> b, std::span<const I2> rows) {
		GUTIL_ASSERT(mat.isCompressed());
		csr_row_reduce_symmetric<InnerSorted>(mat.outerIndexPtr(), mat.innerIndexPtr(), mat.valuePtr(), x, b, rows);
	}

	// direct convenience overload taking a BcHandler, so the caller doesn't
	// need to separately extract dof_numbers/vals
	template<bool InnerSorted=false, typename T, int Options, typename StorageIndex>
	inline void eigen_row_reduce_symmetric(Eigen::SparseMatrix<T,Options,StorageIndex>& mat,
	                                        std::span<const T> x, std::span<T> b, const BcHandler<T>& bc) {
		eigen_row_reduce_symmetric<InnerSorted>(mat, x, b, std::span<const size_t>(bc.dof_numbers));
	}
#endif
}