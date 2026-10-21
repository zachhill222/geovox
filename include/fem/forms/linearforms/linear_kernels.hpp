#pragma once

#include "gutil.hpp"

#include "fem/forms/util.hpp"
#include "fem/forms/base_k_linear_kernel.hpp"
#include "fem/forms/weights.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	///////////////////////////////////////////////////////////////////
	/// A concept for linear kernels
	///////////////////////////////////////////////////////////////////
	template<typename Kernel>
	concept IsLinearKernel = IsKLinearKernel<Kernel> && (Kernel::K==1);


	///////////////////////////////////////////////////////////////////
	/// A few standard linear kernels
	///////////////////////////////////////////////////////////////////
	using ZeroLinearKernel = ZeroKernel<1>;
	static_assert(IsLinearKernel<ZeroLinearKernel>);

	template<IsKernelWeight Weight=void>
	using UnitLinearKernel = UnitKernel<1,Weight>;
	static_assert(IsLinearKernel<UnitLinearKernel<void>>);


	///////////////////////////////////////////////////////////////////
	/// The L2 linear kernel (L2-dual pairing between a scalar field and a given weight)
	/// For the problem -laplace(u) = f(x), the linear rhs should be L(v)=int_D(v*f)
	/// so the weight shoudld be a method to evaluate the function f at each quadrature point.
	/// The quadrature rule will supply the geometric coordinates of x for easier evaluation.
	/// Use GV::FunctionWeight<fun> where fun is a lambda to do the final evaluations.
	/// 	L(v) = int_D(v*wt)
	///////////////////////////////////////////////////////////////////
	template<IsKernelWeight Weight=void>
	struct L2LinearKernel : public KLinearKernel<1,Weight,0,L2LinearKernel<Weight>> {
		using BASE = KLinearKernel<1,Weight,0,L2LinearKernel<Weight>>;
		template<typename QR>
		using ValueArgType = BASE::template ValueArgType<QR>;
		template<typename QR>
		using GradArgType = BASE::template GradArgType<QR>;
		template<typename QR>
		using WeightArgType = BASE::template WeightArgType<QR>;
		template<typename QR>
		using ValueArg = typename BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = typename BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = typename BASE::template WeightArg<QR>;
		using WeightType = Weight;
		
		using BASE::BASE;

		static constexpr std::array<bool,1> NEED_VALS{true};
		static constexpr std::array<bool,1> NEED_GRAD{false};
		
		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEvalImpl(ValueArg<QuadRule> vals,	
				GradArg<QuadRule>, WeightArg<QuadRule> wt_ptr, const QuadRule& qr) noexcept {
			//sanity check
			GUTIL_ASSERT(BASE::IsValArgValid(vals));
			if constexpr (BASE::NEEDS_WEIGHT) {GUTIL_ASSERT(wt_ptr);}

			//types and compile constants
			using Scalar_t = typename QuadRule::Scalar_t;
			static constexpr int N = QuadRule::TOTAL_QUAD_POINTS;
			static constexpr auto qw = QuadRule::quad_w();

			const auto& v_vals = *(vals[0]); //convert pointer to reference

			Scalar_t val{0};
			GUTIL_SIMD(reduction(+:val))
			for (int i=0; i<N; ++i) {
				if constexpr (BASE::NEEDS_WEIGHT) {
					val += v_vals[i] * (*wt_ptr)[i] * qw[i];
				}
				else {
					val += v_vals[i] * qw[i];
				}
			}
			return val * qr.jac_det();
		}
	};
	static_assert(IsLinearKernel<L2LinearKernel<void>>);
}