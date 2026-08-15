#pragma once

#include "gutil.hpp"

#include "fem/forms/util.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	///////////////////////////////////////////////////////////////////
	/// Concept to ensure kernel consistency
	///////////////////////////////////////////////////////////////////
	template<typename K>
	concept IsBilinearKernel = requires (const K k) {
		{ K::NEEDS_WEIGHT     }		-> std::convertible_to<bool>;
		{ K::IS_SYMMETRIC     } 	-> std::convertible_to<bool>;
		{ K::NEEDS_DOF_VALS   }		-> std::convertible_to<bool>;
		{ K::NEEDS_DOF_GRAD   }		-> std::convertible_to<bool>;

		//there should also be a cached_eval method with the signature
		// Scalar(TrialCachedVals*, TrialCachedGrad*, TestCachedVals*, TestCachedGrad*, const QuadRule&)
		// with CachedVals allowed to be nullptr if NEEDS_DOF_VALS is false and
		// CachedGrad* allowed to be nullptr if NEEDS_DOF_GRAD is false.

		//the operator() is templated, but needs 3 arguments
		// { k(std::declval<int>(), std::declval<int>(), std::declval<int>()) };
	};


	///////////////////////////////////////////////////////////////////
	/// A few standard bilinear kernels
	///////////////////////////////////////////////////////////////////
	struct ZeroBilinearKernel {
		//kernel for the bilinear form B(phi,psi) = 0
		static constexpr bool NEEDS_WEIGHT	   = false;
		static constexpr bool IS_SYMMETRIC     = false;
		static constexpr bool NEEDS_DOF_VALS   = false;
		static constexpr bool NEEDS_DOF_GRAD   = false;

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>*, 
			const DofGradCache<QuadRule_t>*, 
			const DofValueCache<QuadRule_t>*,
			const DofGradCache<QuadRule_t>*, 
			const ScalarValueCache<QuadRule_t>*,
			const QuadRule_t&) const noexcept {
			return typename QuadRule_t::Scalar_t{0};
		}

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TrialDof_t, TestDof_t,const QuadRule_t&) const noexcept {
			return typename QuadRule_t::Scalar_t{0};
		}
	};

	struct IdentityBilinearKernel {
		//kernel for the identity matrix
		//TODO: the bilinear form needs a specialization to use this kernel correctly.
		static constexpr bool NEEDS_WEIGHT     = false;
		static constexpr bool IS_SYMMETRIC     = true;
		static constexpr bool NEEDS_DOF_VALS   = false;
		static constexpr bool NEEDS_DOF_GRAD   = false;

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>* u_vals, 
			const DofGradCache<QuadRule_t>*, 
			const DofValueCache<QuadRule_t>* v_vals,
			const DofGradCache<QuadRule_t>*, 
			const ScalarValueCache<QuadRule_t>*,
			const QuadRule_t&) noexcept {
			GUTIL_ABORT("The IdentityBilinearKernel does not have a meaningful cached_eval method");
			return typename QuadRule_t::Scalar_t{0};
		}

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t> requires(std::same_as<TrialDof_t,TestDof_t>)
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TrialDof_t u, TestDof_t v,const QuadRule_t&) const noexcept {
			return static_cast<typename QuadRule_t::Scalar_t>(u==v);
		}
	};

	template<bool IsSymmetric=false, bool NeedsWeight=false>
	struct L2BilinearKernel {
		//kernel for the linear form B(u,v) = int_D(u*v)
		//optionally B(u,v) = int_D(u*v*f) where f is supplied by CRTP in the Derived class
		//and is a function of x,y,z.
		static constexpr bool NEEDS_WEIGHT     = NeedsWeight;
		static constexpr bool IS_SYMMETRIC     = IsSymmetric;
		static constexpr bool NEEDS_DOF_VALS   = true;
		static constexpr bool NEEDS_DOF_GRAD   = false;

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>* 	u_vals, 
			const DofGradCache<QuadRule_t>*, 
			const DofValueCache<QuadRule_t>* 	v_vals,
			const DofGradCache<QuadRule_t>*, 
			const ScalarValueCache<QuadRule_t>* wt,
			const QuadRule_t& qr) const noexcept {

			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			
			GUTIL_ASSERT(u_vals && v_vals);
			if constexpr (NEEDS_WEIGHT) {GUTIL_ASSERT(wt);}

			//accumulation
			auto qw = qr.quad_w();
			Scalar_t result{0};
			if constexpr (NEEDS_WEIGHT) {
				GUTIL_SIMD(reduction(+:result))
				for (int i=0; i<N; ++i) {
					result += (*u_vals)[i] * (*v_vals)[i] * (*wt)[i] * qw[i];
				}
			}
			else {
				GUTIL_SIMD(reduction(+:result))
				for (int i=0; i<N; ++i) {
					result += (*u_vals)[i] * (*v_vals)[i] * qw[i];
				}
			}

			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};

	template<bool IsSymmetric=false, bool NeedsWeight=false>
	struct H1BilinearKernel {
		//kernel for the linear form B(u,v) = int_D(grad(u)*grad(v))
		//optionally B(u,v) = int_D(grad(u)*grad(v)*f) where f is supplied by CRTP in the Derived class
		//f must be a function of x,y,z.
		static constexpr bool NEEDS_WEIGHT     = NeedsWeight;
		static constexpr bool IS_SYMMETRIC     = IsSymmetric;
		static constexpr bool NEEDS_DOF_VALS   = false;
		static constexpr bool NEEDS_DOF_GRAD   = true;

		
		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>*, 
			const DofGradCache<QuadRule_t>* 	u_grad, 
			const DofValueCache<QuadRule_t>*,
			const DofGradCache<QuadRule_t>* 	v_grad, 
			const ScalarValueCache<QuadRule_t>* wt,
			const QuadRule_t& qr) const noexcept {

			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			GUTIL_ASSERT(u_grad && v_grad);
			if constexpr (NEEDS_WEIGHT) {GUTIL_ASSERT(wt);}

			//accumulation
			auto qw = qr.quad_w();
			const Scalar_t j_i_xx = qr.jacobian_diag_inv[v_grad->depth][0]*qr.jacobian_diag_inv[u_grad->depth][0];
			const Scalar_t j_i_yy = qr.jacobian_diag_inv[v_grad->depth][1]*qr.jacobian_diag_inv[u_grad->depth][1];
			const Scalar_t j_i_zz = qr.jacobian_diag_inv[v_grad->depth][2]*qr.jacobian_diag_inv[u_grad->depth][2];

			Scalar_t result{0};
			if constexpr (NEEDS_WEIGHT) {
				GUTIL_SIMD(reduction(+:result))
				for (int i=0; i<N; ++i) {
					result += ( (*v_grad)[0][i] * (*u_grad)[0][i] * j_i_xx +
								(*v_grad)[1][i] * (*u_grad)[1][i] * j_i_yy +
								(*v_grad)[2][i] * (*u_grad)[2][i] * j_i_zz ) * (*wt)[i] * qw[i];
				}
			}
			else {
				GUTIL_SIMD(reduction(+:result))
				for (int i=0; i<N; ++i) {
					result += ( (*v_grad)[0][i] * (*u_grad)[0][i] * j_i_xx +
								(*v_grad)[1][i] * (*u_grad)[1][i] * j_i_yy +
								(*v_grad)[2][i] * (*u_grad)[2][i] * j_i_zz ) * qw[i];
				}
			}

			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};

	
	///////////////////////////////////////////////////////////////
	/// Allow addition and scalar multiplication of forms for more efficient
	/// and convenient kernels.
	///////////////////////////////////////////////////////////////
	template<typename T, IsBilinearKernel K>
	struct ScaledBilinearKernel {
		static constexpr bool NEEDS_WEIGHT     = K::NEEDS_WEIGHT;
		static constexpr bool IS_SYMMETRIC 	   = K::IS_SYMMETRIC;
		static constexpr bool NEEDS_DOF_VALS   = K::NEEDS_DOF_VALS;
		static constexpr bool NEEDS_DOF_GRAD   = K::NEEDS_DOF_GRAD;

		T scale;
		K kernel;

		constexpr ScaledBilinearKernel(T scl, K krnl) : scale(scl), kernel(std::move(krnl)) {}
 
		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>* u_vals, 
			const DofGradCache<QuadRule_t>*  u_grad, 
			const DofValueCache<QuadRule_t>* v_vals,
			const DofGradCache<QuadRule_t>*  v_grad, 
			const ScalarValueCache<QuadRule_t>*  wt,
			const QuadRule_t& qr) const noexcept {
			return scale * kernel.template cached_eval<QuadRule_t>(u_vals, u_grad, v_vals, v_grad, wt, qr);
		}

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TrialDof_t u, TestDof_t v, const QuadRule_t& qr) const noexcept {
			return static_cast<typename QuadRule_t::Scalar_t>(scale) * kernel(u,v,qr);
		}
	};

	template<IsBilinearKernel K1, IsBilinearKernel K2>
	struct SumBilinearKernel {
		//note that the weight must be the same for both kernels
		static constexpr bool NEEDS_WEIGHT     = K1::NEEDS_WEIGHT     || K2::NEEDS_WEIGHT;
		static constexpr bool IS_SYMMETRIC 	   = K1::IS_SYMMETRIC     && K2::IS_SYMMETRIC;
		static constexpr bool NEEDS_DOF_VALS   = K1::NEEDS_DOF_VALS   || K2::NEEDS_DOF_VALS;
		static constexpr bool NEEDS_DOF_GRAD   = K1::NEEDS_DOF_GRAD   || K2::NEEDS_DOF_GRAD;

		K1 left;
		K2 right;

		constexpr SumBilinearKernel(K1 L, K2 R) : left(std::move(L)), right(std::move(R)) {}

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>* u_vals, 
			const DofGradCache<QuadRule_t>*  u_grad, 
			const DofValueCache<QuadRule_t>* v_vals,
			const DofGradCache<QuadRule_t>*  v_grad,
			const ScalarValueCache<QuadRule_t>*  wt,
			const QuadRule_t& qr) const noexcept {
			return left.template cached_eval<QuadRule_t>(u_vals, u_grad, v_vals, v_grad, wt, qr) 
				 + right.template cached_eval<QuadRule_t>(u_vals, u_grad, v_vals, v_grad, wt, qr);
		}
	};

	template<typename T, IsBilinearKernel K>
	[[nodiscard]] inline constexpr auto operator*(T scale, K kernel) noexcept {
		return ScaledBilinearKernel<T,K>{scale, std::move(kernel)};
	}

	template<IsBilinearKernel K1, IsBilinearKernel K2>
	[[nodiscard]] inline constexpr auto operator+(K1 left, K2 right) noexcept {
		return SumBilinearKernel<K1,K2>{std::move(left), std::move(right)};
	}
}