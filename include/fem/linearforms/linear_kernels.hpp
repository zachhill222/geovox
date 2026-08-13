#pragma once

#include "gutil.hpp"

#include "fem/mesh_quadrature.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	///////////////////////////////////////////////////////////////////
	/// A few evaluation methods for standard linear kernels
	///////////////////////////////////////////////////////////////////
	struct ZeroLinearKernel {
		//kernel for the linear form L(phi) = 0
		static constexpr bool NEEDS_GEO_POINTS = false;

		template<typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TestDof_t,const QuadRule_t&) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			return Scalar_t{0};
		}
	};


	struct IdentityLinearKernel {
		//kernel for the linear form L(phi) = int_D(phi)
		static constexpr bool NEEDS_GEO_POINTS = false;

		template<typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TestDof_t v, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			const uint8_t dd  = v.depth();
			const uint8_t loc = v.local_dof_number(qr.support_element[dd]);

			Scalar_t v_vals[QuadRule_t::TOTAL_QUAD_POINTS];
			auto qx=qr.quad_x(dd), qy=qr.quad_y(dd), qz=qr.quad_z(dd);
			v.evaluate_simd(loc, v_vals, qx.data(), qy.data(), qz.data(), N);

			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += v_vals[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};

	template<typename Derived>
	struct WeightedLinearKernel {
		//kernel for the linear form L(phi) = int_D(phi * f)
		//use a simple CRTP class
		// MyWeightedLinearKernel : public WeightedLinearKernel<MyWeightedLinearKernel>
		// with an evaluation method
		// template<typename T, int N>
		// static void eval_weight(std::span<T,N> result, std::span<const T,N> x, ...)
		static constexpr bool NEEDS_GEO_POINTS = true;

		template<typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TestDof_t v, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			const uint8_t dd  = v.depth();
			const uint8_t loc = v.local_dof_number(qr.support_element[dd]);

			Scalar_t v_vals[N];
			auto qx=qr.quad_x(dd), qy=qr.quad_y(dd), qz=qr.quad_z(dd);
			v.evaluate_simd(loc, v_vals, qx.data(), qy.data(), qz.data(), N);

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
	};
}