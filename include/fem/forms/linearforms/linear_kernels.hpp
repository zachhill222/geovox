#pragma once

#include "gutil.hpp"

#include "fem/forms/util.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	///////////////////////////////////////////////////////////////////
	/// A few evaluation methods for standard linear kernels
	///////////////////////////////////////////////////////////////////
	struct ZeroLinearKernel {
		//kernel for the linear form L(phi) = 0
		static constexpr bool NEEDS_GEO_POINTS = false;

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval(
			const DofValueCache<QuadRule_t>*,
			const DofGradCache<QuadRule_t>*,
			const QuadRule_t&) const noexcept {
			return typename QuadRule_t::Scalar_t{0};
		}

		template<typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TestDof_t,const QuadRule_t&) const noexcept {
			return typename QuadRule_t::Scalar_t{0};
		}
	};


	template<typename Derived=void>
	struct L2LinearKernel {
		//kernel for the linear form L(phi) = int_D(phi)
		static constexpr bool NEEDS_GEO_POINTS = std::same_as<Derived,void> ? false :Derived::NEEDS_GEO_POINTS;
		
		
		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval (
			const DofValueCache<QuadRule_t>* v_vals,
			const DofGradCache<QuadRule_t>*  v_grad,
			const QuadRule_t&				 qr) const noexcept requires(!NEEDS_GEO_POINTS) {
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			
			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += v_vals[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}

		template<typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t cached_eval (
			const DofValueCache<QuadRule_t>* v_vals,
			const DofGradCache<QuadRule_t>*  v_grad,
			const QuadRule_t&				 qr) const noexcept requires(NEEDS_GEO_POINTS) {
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			
			Scalar_t f_vals[N];
			Derived::eval_weight(std::span<Scalar_t,N>{f_vals},
						qr.geo_x(),qr.geo_y(),qr.geo_z());

			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += v_vals[i]*f_vals[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}

		template<typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TestDof_t v, const QuadRule_t& qr) const noexcept {
			DofValueCache<QuadRule_t> v_vals(v,qr);
			return cached_eval<QuadRule_t>(&v_vals, nullptr, qr);
		}
	};
}