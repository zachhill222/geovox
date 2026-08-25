#pragma once

#include "gutil.hpp"

#include "fem/forms/util.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	///////////////////////////////////////////////////////////////////
	/// A concept for linear kernels
	///////////////////////////////////////////////////////////////////
	template<typename K>
	concept IsLinearKernel = requires (const K k) {
		{ K::NEEDS_WEIGHT   } -> std::convertible_to<bool>;
		{ K::TEST_DOF_VALS  } -> std::convertible_to<bool>;
		{ K::TEST_DOF_GRAD  } -> std::convertible_to<bool>;
		{ K::TRIAL_DOF_VALS } -> std::convertible_to<bool>;	//needed for compatibility with dof caching
		{ K::TRIAL_DOF_GRAD } -> std::convertible_to<bool>;	//needed for compatibility with dof caching
	};



	///////////////////////////////////////////////////////////////////
	/// A few evaluation methods for standard linear kernels
	///////////////////////////////////////////////////////////////////
	struct ZeroLinearKernel {
		//kernel for the linear form L(phi) = 0
		static constexpr bool NEEDS_WEIGHT   = false;
		static constexpr bool TEST_DOF_VALS  = false;
		static constexpr bool TEST_DOF_GRAD  = false;
		static constexpr bool TRIAL_DOF_VALS = false;	//for compatibility with caching dof values
		static constexpr bool TRIAL_DOF_GRAD = false;	//for compatibility with caching dof values

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>*,
			const DofGradCache<QuadRule_t>*,
			const ScalarValueCache<QuadRule_t>*,
			const QuadRule_t&) const noexcept {
			return typename QuadRule_t::Scalar_t{0};
		}
	};


	template<bool NeedsWeight=false>
	struct L2LinearKernel {
		//kernel for the linear form L(phi) = int_D(phi*wt)
		static constexpr bool NEEDS_WEIGHT   = NeedsWeight;
		static constexpr bool TEST_DOF_VALS  = true;
		static constexpr bool TEST_DOF_GRAD  = false;
		static constexpr bool TRIAL_DOF_VALS = false;	//for compatibility with caching dof values
		static constexpr bool TRIAL_DOF_GRAD = false;	//for compatibility with caching dof values
		
		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval (
			const DofValueCache<QuadRule_t>* 	v_vals,
			const DofGradCache<QuadRule_t>*,
			const ScalarValueCache<QuadRule_t>*,
			const QuadRule_t& qr) const noexcept requires(!NEEDS_WEIGHT) {
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			
			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += (*v_vals)[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval (
			const DofValueCache<QuadRule_t>* v_vals,
			const DofGradCache<QuadRule_t>*,
			const ScalarValueCache<QuadRule_t>*  wt,
			const QuadRule_t& qr) const noexcept requires(NEEDS_WEIGHT) {
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			
			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += (*v_vals)[i] * (*wt)[i] * qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};
}