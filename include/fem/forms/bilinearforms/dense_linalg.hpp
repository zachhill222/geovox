#pragma once

#include "gutil.hpp"

#include <span>
#include <type_traits>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace GV {
	////////////////////////////////////////////////////
	// Naming conventions (BLAS inspired)
	// Note that result vectors are always accumulated into.
	//
	//	Field 1: matrix type + storage format
	//	ge - general matrix (dense, any size)
	//	sq - square matrix (dense)
	//	sy - symmetric matrix
	//	tr - triangular matrix
	//
	//	rm - row major format
	//	cm - col major format
	//
	//	Field 2: operation type
	//	mv  - matrix-vector product
	//  vmv - vector-matrix-vector product (bilinear form)  
	//
	//	Field 3: modifiers for implicit matrix operations
	//		note for a square matrix A, we set A = L + D + U (strict lower/diagonal/strict upper portions)
	//	l  - strict lower part
	//  d  - diagonal part (square matrix only)
	//	u  - strict upper part
	//  t  - transpose
	//
	//  Ex: germ_mv 	: general matrix (row-major) matrix-vector multply (y+=Ax)
	//  Ex: sqrm_mv_d	: square matrix (row-major) matrix-vector multiply with diagonal only (y+=D*x where A=L+D+U);

	template<typename T>
	inline constexpr void germ_mv(T* y, const size_t M, const T* x, const size_t N, const T* mat) {
		GUTIL_ASSERT(y && x && mat && M>0 && N>0);
		// Compute y += A*x
		// row major: A(i,j) = mat[j + i*N]
		// M - number of rows
		// N - number of columns
		// y - result to accumulate to (size M)
		// x - vector to multiply (size N)
		size_t offset = 0;
		for (size_t i=0; i<M; ++i, offset+=N) {
			T acc{0};
			GUTIL_SIMD(reduction(+:acc))
			for (size_t j=0; j<N; ++j) {
				acc += mat[j+offset]*x[j];
			}
			y[i] += acc;
		}
	}

	template<typename T>
	inline constexpr void gecm_mv(T* y, const size_t M, const T* x, const size_t N, const T* mat) {
		GUTIL_ASSERT(y && x && mat && M>0 && N>0);
		// Compute y += A*x
		// col major: A(i,j) = mat[i + j*M]
		// M - number of rows
		// N - number of columns
		// y - result to accumulate to (size M)
		// x - vector to multiply (size N)
		size_t offset = 0;
		for (size_t j=0; j<N; ++j, offset+=M) {
			GUTIL_SIMD()
			for (size_t i=0; i<M; ++i) {
				y[i] += mat[i+offset]*x[j];
			}
		}
	}

	template<typename T>
	inline constexpr T gecm_vmv(const T* y, const size_t M, const T* x, const size_t N, const T* mat) {
		GUTIL_ASSERT(y && x && mat && M>0 && N>0);
		// Compute y^t*A*x
		// col major: A(i,j) = mat[i + j*M]
		// M - number of rows
		// N - number of columns
		// y - result to accumulate to (size M)
		// x - vector to multiply (size N)
		T val{0};
		GUTIL_SIMD(reduction(+:val) collapse(2))
		for (size_t j=0; j<N; ++j) {
			for (size_t i=0; i<M; ++i) {
				val += y[i] * mat[i + j*M] * x[j];
			}
		}
		return val;
	}










	// template<typename T>
	// void local_multiply_transpose(std::span<T> y, 
	// 					std::span<std::type_identity_t<const T>> x, 
	// 					std::span<std::type_identity_t<const T>> loc_mat_row_major)
	// {
	// 	//Compute y = M^t*x
	// 	const uint64_t n = y.size();
	// 	const uint64_t m = x.size();

	// 	for (uint64_t i=0; i<n; ++i) {
	// 		const uint64_t offset = i*m;
	// 		#pragma omp simd
	// 		for (uint64_t j=0; j<m; ++j) {
	// 			y[j] += loc_mat_row_major[j+offset]*x[i];
	// 		}
	// 	}
	// }

	// template<typename T>
	// void local_multiply_lower(std::span<T> y, 
	// 					std::span<std::type_identity_t<const T>> x, 
	// 					std::span<std::type_identity_t<const T>> loc_mat_row_major)
	// {
	// 	//Compute y = L*x where L is the (strict) lower part of the matrix
	// 	//the matrix must be square
	// 	const uint64_t n = y.size();
	// 	assert(n==x.size());

	// 	for (uint64_t i=0; i<n; ++i) {
	// 		const uint64_t offset = i*n;
	// 		#pragma omp simd
	// 		for (uint64_t j=0; j<i; ++j) {
	// 			y[j] += loc_mat_row_major[j+offset]*x[i];
	// 		}
	// 	}
	// }

	// template<typename T>
	// void local_multiply_upper(std::span<T> y, 
	// 					std::span<std::type_identity_t<const T>> x, 
	// 					std::span<std::type_identity_t<const T>> loc_mat_row_major)
	// {
	// 	//Compute y = U*x where U is the (strict) upper part of the matrix
	// 	//the matrix must be square
	// 	const uint64_t n = y.size();
	// 	assert(n==x.size());

	// 	for (uint64_t i=0; i<n; ++i) {
	// 		const uint64_t offset = i*n;
	// 		#pragma omp simd
	// 		for (uint64_t j=i+1; j<n; ++j) {
	// 			y[j] += loc_mat_row_major[j+offset]*x[i];
	// 		}
	// 	}
	// }

	// template<typename T>
	// void local_multiply_upper_lower(std::span<T> y, 
	// 					std::span<std::type_identity_t<const T>> x, 
	// 					std::span<std::type_identity_t<const T>> loc_mat_row_major)
	// {
	// 	//Compute y = (L+U)*x where U is the (strict) upper part of the matrix and L is the lower part.
	// 	//the matrix must be square
	// 	const uint64_t n = y.size();
	// 	assert(n==x.size());

	// 	for (uint64_t i=0; i<n; ++i) {
	// 		const uint64_t offset = i*n;
	// 		#pragma omp simd
	// 		for (uint64_t j=0; j<i; ++j) {
	// 			y[j] += loc_mat_row_major[j+offset]*x[i];
	// 		}
			
	// 		#pragma omp simd
	// 		for (uint64_t j=i+1; j<n; ++j) {
	// 			y[j] += loc_mat_row_major[j+offset]*x[i];
	// 		}
	// 	}
	// }

	// template<typename T>
	// void local_jacobi(	std::span<T> y, 
	// 					std::span<std::type_identity_t<const T>> x, 
	// 					std::span<std::type_identity_t<const T>> loc_mat_row_major)
	// {
	// 	//Compute y = D^-1 * (x - (L+U)*y) as one iteration of the jacobi method to approximate y=M_inv*x
	// 	const uint64_t n = y.size();
	// 	assert(n==x.size());

	// 	//compute M*y
	// 	std::vector<T> M_y(n,0.0);
	// 	local_multiply(std::span<T>(M_y),y,loc_mat_row_major);

	// 	#pragma omp simd
	// 	for (uint64_t i=0; i<n; ++i) {
	// 		y[i] = (x[i] - M_y[i] + y[i]*loc_mat_row_major[i+i*n]) / loc_mat_row_major[i+i*n];
	// 	}
	// }


	// //TODO: vecorize Gauss Seidel
	// template<typename T>
	// void local_gauss_seidel(std::span<T> y, 
	// 						std::span<std::type_identity_t<const T>> x, 
	// 						std::span<std::type_identity_t<const T>> loc_mat_row_major)
	// {
	// 	//Compute y = L^-1 * (x - U*y) as one iteration of the gauss-seidel method to approximate y=M_inv*x
	// 	const uint64_t n = y.size();
	// 	assert(n==x.size());

	// 	for (uint64_t i=0; i<n; ++i) {
	// 		const uint64_t offset = i*n;
	// 		y[i] = x[i];
	// 		for (uint64_t j=0; j<n; ++j) {
	// 			y[i] -= (i!=j) ? y[j]*loc_mat_row_major[j+offset] : 0.0;
	// 		}
	// 		y[i] /= loc_mat_row_major[i+offset];
	// 	}
	// }

	// template<typename T>
	// void local_gauss_seidel_backward(	std::span<T> y, 
	// 									std::span<std::type_identity_t<const T>> x, 
	// 									std::span<std::type_identity_t<const T>> loc_mat_row_major)
	// {
	// 	//Compute y = U^-1 * (x - L*y) as one iteration of the gauss-seidel method to approximate y=M_inv*x
	// 	const uint64_t n = y.size();
	// 	assert(n==x.size());

	// 	for (uint64_t ii=0; ii<n; ++ii) {
	// 		const uint64_t i = n-1-ii; //loop backwards and guard against integer underflow
	// 		const uint64_t offset = i*n;
	// 		y[i] = x[i];
	// 		for (uint64_t j=0; j<n; ++j) {
	// 			y[i] -= (i!=j) ? y[j]*loc_mat_row_major[j+offset] : 0.0;
	// 		}
	// 		y[i] /= loc_mat_row_major[i+offset];
	// 	}
	// }
}