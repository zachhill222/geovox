#pragma once

#include <span>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{	
	template<typename T>
	void local_multiply(std::span<T> y, 
						std::span<std::type_identity_t<const T>> x, 
						std::span<std::type_identity_t<const T>> loc_mat_row_major)
	{
		//Compute y = M*x

		const uint64_t n = y.size();
		const uint64_t m = x.size();

		for (uint64_t i=0; i<n; ++i) {
			const uint64_t offset = i*m;
			#pragma omp simd
			for (uint64_t j=0; j<m; ++j) {
				y[i] += loc_mat_row_major[j+offset]*x[j];
			}
		}
	}

	template<typename T>
	void local_multiply_transpose(std::span<T> y, 
						std::span<std::type_identity_t<const T>> x, 
						std::span<std::type_identity_t<const T>> loc_mat_row_major)
	{
		//Compute y = M^t*x
		const uint64_t n = y.size();
		const uint64_t m = x.size();

		for (uint64_t i=0; i<n; ++i) {
			const uint64_t offset = i*m;
			#pragma omp simd
			for (uint64_t j=0; j<m; ++j) {
				y[j] += loc_mat_row_major[j+offset]*x[i];
			}
		}
	}

	template<typename T>
	void local_jacobi(	std::span<T> y, 
						std::span<std::type_identity_t<const T>> x, 
						std::span<std::type_identity_t<const T>> loc_mat_row_major)
	{
		//Compute y = D^-1 * (x - (L+U)*y) as one iteration of the jacobi method to approximate y=M_inv*x
		const uint64_t n = y.size();
		assert(n==x.size());

		//compute M*y
		std::vector<T> M_y(n,0.0);
		local_multiply(std::span<T>(M_y),y,loc_mat_row_major);

		#pragma omp simd
		for (uint64_t i=0; i<n; ++i) {
			y[i] = (x[i] - M_y[i] + y[i]*loc_mat_row_major[i+i*n]) / loc_mat_row_major[i+i*n];
		}
	}


	//TODO: vecorize Gauss Seidel
	template<typename T>
	void local_gauss_seidel(std::span<T> y, 
							std::span<std::type_identity_t<const T>> x, 
							std::span<std::type_identity_t<const T>> loc_mat_row_major)
	{
		//Compute y = L^-1 * (x - U*y) as one iteration of the gauss-seidel method to approximate y=M_inv*x
		const uint64_t n = y.size();
		assert(n==x.size());

		for (uint64_t i=0; i<n; ++i) {
			const uint64_t offset = i*n;
			y[i] = x[i];
			for (uint64_t j=0; j<n; ++j) {
				y[i] -= (i!=j) ? y[j]*loc_mat_row_major[j+offset] : 0.0;
			}
			y[i] /= loc_mat_row_major[i+offset];
		}
	}

	template<typename T>
	void local_gauss_seidel_backward(	std::span<T> y, 
										std::span<std::type_identity_t<const T>> x, 
										std::span<std::type_identity_t<const T>> loc_mat_row_major)
	{
		//Compute y = U^-1 * (x - L*y) as one iteration of the gauss-seidel method to approximate y=M_inv*x
		const uint64_t n = y.size();
		assert(n==x.size());

		for (uint64_t ii=0; ii<n; ++ii) {
			const uint64_t i = n-1-ii; //loop backwards and guard against integer underflow
			const uint64_t offset = i*n;
			y[i] = x[i];
			for (uint64_t j=0; j<n; ++j) {
				y[i] -= (i!=j) ? y[j]*loc_mat_row_major[j+offset] : 0.0;
			}
			y[i] /= loc_mat_row_major[i+offset];
		}
	}
}