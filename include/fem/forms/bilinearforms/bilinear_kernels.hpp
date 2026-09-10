#pragma once

#include "gutil.hpp"

#include "fem/forms/util.hpp"
#include "fem/forms/base_k_linear_kernel.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	///////////////////////////////////////////////////////////////////
	/// Concept to ensure kernel consistency
	///////////////////////////////////////////////////////////////////
	template<typename Kernel>
	concept IsBilinearKernel = IsKLinearKernel<Kernel> && (Kernel::K==2);

	template<typename Kernel>
	concept IsSymmetricBilinearKernel = IsBilinearKernel<Kernel> && Kernel::IS_SYMMETRIC;

	///////////////////////////////////////////////////////////////////
	/// A few standard bilinear kernels
	///////////////////////////////////////////////////////////////////
	using ZeroBilinearKernel = ZeroKernel<2>;
	static_assert(IsSymmetricBilinearKernel<ZeroBilinearKernel>);

	template<bool IsWeighted=false>
	using IdentityBilinearKernel = IdentityKernel<2,IsWeighted>;
	static_assert(IsSymmetricBilinearKernel<IdentityBilinearKernel<true>>);
	static_assert(IsSymmetricBilinearKernel<IdentityBilinearKernel<false>>);

	///////////////////////////////////////////////////////////////////
	/// The L2 bilinear kernel for the bilinear form
	///   B(u,v) = int_D(u*v) or B(u,v) = int_D(u*v*wt)
	///////////////////////////////////////////////////////////////////
	template<bool IsWeighted=false>
	struct L2BilinearKernel : public KLinearKernel<2,IsWeighted,2,L2BilinearKernel<IsWeighted>> {
		using BASE = KLinearKernel<2,IsWeighted,2,L2BilinearKernel<IsWeighted>>;
		static_assert(BASE::IS_SYMMETRIC);
		template<typename QR>
		using ValueArg = typename BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = typename BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = typename BASE::template WeightArg<QR>;

		using BASE::BASE;

		static constexpr std::array<bool,2> NEED_VALS{true,true};
		static constexpr std::array<bool,2> NEED_GRAD{false,false};

		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar CachedEvalImpl(const ValueArg<QuadRule>& vals,	
				const GradArg<QuadRule>&, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept {
			//sanity check
			GUTIL_ASSERT(BASE::IsValArgValid(vals));
			if constexpr (IsWeighted) {GUTIL_ASSERT(wt_ptr);}

			//types and compile constants
			using Scalar = typename QuadRule::Scalar;
			static constexpr int N = QuadRule::TOTAL_QUAD_POINTS;
			static constexpr auto qw = QuadRule::quad_w();

			const auto& u_vals = *(vals[0]);
			const auto& v_vals = *(vals[1]);

			//accumulation
			Scalar val{0};
			GUTIL_SIMD(reduction(+:val))
			for (int i=0; i<N; ++i) {
				if constexpr (IsWeighted) {
					val += u_vals[i] * v_vals[i] * (*wt_ptr)[i] * qw[i];
				}
				else {
					val += u_vals[i] * v_vals[i] * qw[i];
				}
			}

			return val * qr.jac_det();
		}
	};
	static_assert(IsSymmetricBilinearKernel<L2BilinearKernel<true>>);
	static_assert(IsSymmetricBilinearKernel<L2BilinearKernel<false>>);


	///////////////////////////////////////////////////////////////////
	/// The (grad portion of the) H1 bilinear kernel for the bilinear form
	///   B(u,v) = int_D(grad(u)*grad(v)) or B(u,v) = int_D(grad(u)*grad(v)*wt)
	///////////////////////////////////////////////////////////////////
	template<bool IsWeighted=false>
	struct H1BilinearKernel : public KLinearKernel<2,IsWeighted,2,H1BilinearKernel<IsWeighted>> {
		using BASE = KLinearKernel<2,IsWeighted,2,H1BilinearKernel<IsWeighted>>;
		static_assert(BASE::IS_SYMMETRIC);
		template<typename QR>
		using ValueArg = typename BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = typename BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = typename BASE::template WeightArg<QR>;

		using BASE::BASE;

		static constexpr std::array<bool,2> NEED_VALS{false,false};
		static constexpr std::array<bool,2> NEED_GRAD{true,true};

		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar CachedEvalImpl(const ValueArg<QuadRule>&,	
				const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept {
			//sanity check
			GUTIL_ASSERT(BASE::IsGradArgValid(grad));
			if constexpr (IsWeighted) {GUTIL_ASSERT(wt_ptr);}

			//types and compile constants
			using Scalar = typename QuadRule::Scalar;
			static constexpr int N = QuadRule::TOTAL_QUAD_POINTS;
			static constexpr auto qw = QuadRule::quad_w();

			const auto& u_grad = *(grad[0]);
			const auto& v_grad = *(grad[1]);

			//chain rule. note we need the jacobian inverse at the dof support depth
			const Scalar j_i_xx = qr.jacobian_diag_inv[u_grad.depth][0]*qr.jacobian_diag_inv[v_grad.depth][0];
			const Scalar j_i_yy = qr.jacobian_diag_inv[u_grad.depth][1]*qr.jacobian_diag_inv[v_grad.depth][1];
			const Scalar j_i_zz = qr.jacobian_diag_inv[u_grad.depth][2]*qr.jacobian_diag_inv[v_grad.depth][2];
			
			//accumulation
			Scalar val{0};
			GUTIL_SIMD(reduction(+:val))
			for (int i=0; i<N; ++i) {
				if constexpr (IsWeighted) {
					//note the access pattern is dof->component->value at quad point
					val += (u_grad[0][i] * v_grad[0][i] * j_i_xx +
							u_grad[1][i] * v_grad[1][i] * j_i_yy +
							u_grad[2][i] * v_grad[2][i] * j_i_zz ) * (*wt_ptr)[i] * qw[i];
				}
				else {
					//note the access pattern is dof->component->value at quad point
					val += (u_grad[0][i] * v_grad[0][i] * j_i_xx +
							u_grad[1][i] * v_grad[1][i] * j_i_yy +
							u_grad[2][i] * v_grad[2][i] * j_i_zz ) * qw[i];
				}
			}

			return val * qr.jac_det();
		}
	};
	static_assert(IsSymmetricBilinearKernel<H1BilinearKernel<true>>);
	static_assert(IsSymmetricBilinearKernel<H1BilinearKernel<false>>);


	///////////////////////////////////////////////////////////////////
	/// The mixed bilinear kernel for the bilinear form (part of Hdiv)
	///   B(u,v) = int_D(u*partial_axis(v)) or B(u,v) = int_D(u*partial_axis(v)*wt)
	///////////////////////////////////////////////////////////////////
	template<int Axis, bool IsWeighted=false> requires (0<=Axis && Axis<3)
	struct ValPartialBilinearForm : KLinearKernel<2,IsWeighted,0,ValPartialBilinearForm<Axis,IsWeighted>> {
		using BASE = KLinearKernel<2,IsWeighted,0,ValPartialBilinearForm<Axis,IsWeighted>>;
		static_assert(!BASE::IS_SYMMETRIC);
		template<typename QR>
		using ValueArg = typename BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = typename BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = typename BASE::template WeightArg<QR>;

		using BASE::BASE;

		static constexpr std::array<bool,2> NEED_VALS{true,false};
		static constexpr std::array<bool,2> NEED_GRAD{false,true};

		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar CachedEvalImpl(const ValueArg<QuadRule>& vals,	
				const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept {
			//sanity check
			GUTIL_ASSERT(BASE::IsValArgValid(vals));
			GUTIL_ASSERT(BASE::IsGradArgValid(grad));
			if constexpr (IsWeighted) {GUTIL_ASSERT(wt_ptr);}

			//types and compile constants
			using Scalar = typename QuadRule::Scalar;
			static constexpr int N = QuadRule::TOTAL_QUAD_POINTS;
			static constexpr auto qw = QuadRule::quad_w();

			const auto& u_vals = *(vals[0]);
			const auto& v_grad = *(grad[1]);

			//chain rule. note we need the jacobian inverse at the dof support depth
			//and that this value can be factored out of the sum.
			const Scalar j_inv = qr.jacobian_diag_inv[v_grad.depth][Axis];
			
			//accumulation
			Scalar val{0};
			GUTIL_SIMD(reduction(+:val))
			for (int i=0; i<N; ++i) {
				if constexpr (IsWeighted) {
					//note the access pattern is dof->component->value at quad point
					val += u_vals[i] * v_grad[Axis][i] * (*wt_ptr)[i] * qw[i];
				}
				else {
					//note the access pattern is dof->component->value at quad point
					val += u_vals[i] * v_grad[Axis][i] * qw[i];
				}
			}

			return val * j_inv * qr.jac_det();
		}
	};
	static_assert(IsBilinearKernel<ValPartialBilinearForm<0,true>>);
	static_assert(IsBilinearKernel<ValPartialBilinearForm<1,true>>);
	static_assert(IsBilinearKernel<ValPartialBilinearForm<2,true>>);
	static_assert(IsBilinearKernel<ValPartialBilinearForm<0,false>>);
	static_assert(IsBilinearKernel<ValPartialBilinearForm<1,false>>);
	static_assert(IsBilinearKernel<ValPartialBilinearForm<2,false>>);


	///////////////////////////////////////////////////////////////////
	/// The mixed bilinear kernel for the bilinear form (part of Hdiv adjoint)
	///   B(u,v) = int_D(partial_axis(u)*v) or B(u,v) = int_D(partial_axis(u)*v*wt)
	///////////////////////////////////////////////////////////////////
	template<int Axis, bool IsWeighted=false> requires (0<=Axis && Axis<3)
	struct PartialValBilinearForm : KLinearKernel<2,IsWeighted,0,PartialValBilinearForm<Axis,IsWeighted>> {
		using BASE = KLinearKernel<2,IsWeighted,0,PartialValBilinearForm<Axis,IsWeighted>>;
		static_assert(!BASE::IS_SYMMETRIC);
		template<typename QR>
		using ValueArg = typename BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = typename BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = typename BASE::template WeightArg<QR>;

		using BASE::BASE;

		static constexpr std::array<bool,2> NEED_VALS{false,true};
		static constexpr std::array<bool,2> NEED_GRAD{true,false};

		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar CachedEvalImpl(const ValueArg<QuadRule>& vals,	
				const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept {
			//sanity check
			GUTIL_ASSERT(BASE::IsValArgValid(vals));
			GUTIL_ASSERT(BASE::IsGradArgValid(grad));
			if constexpr (IsWeighted) {GUTIL_ASSERT(wt_ptr);}

			//types and compile constants
			using Scalar = typename QuadRule::Scalar;
			static constexpr int N = QuadRule::TOTAL_QUAD_POINTS;
			static constexpr auto qw = QuadRule::quad_w();

			const auto& u_grad = *(grad[0]);
			const auto& v_vals = *(vals[1]);

			//chain rule. note we need the jacobian inverse at the dof support depth
			//and that this value can be factored out of the sum.
			const Scalar j_inv = qr.jacobian_diag_inv[u_grad.depth][Axis];
			
			//accumulation
			Scalar val{0};
			GUTIL_SIMD(reduction(+:val))
			for (int i=0; i<N; ++i) {
				if constexpr (IsWeighted) {
					//note the access pattern is dof->component->value at quad point
					val += u_grad[Axis][i] * v_vals[i] * (*wt_ptr)[i] * qw[i];
				}
				else {
					//note the access pattern is dof->component->value at quad point
					val += u_grad[Axis][i] * v_vals[i] * qw[i];
				}
			}

			return val * j_inv * qr.jac_det();
		}
	};
	static_assert(IsBilinearKernel<PartialValBilinearForm<0,true>>);
	static_assert(IsBilinearKernel<PartialValBilinearForm<1,true>>);
	static_assert(IsBilinearKernel<PartialValBilinearForm<2,true>>);
	static_assert(IsBilinearKernel<PartialValBilinearForm<0,false>>);
	static_assert(IsBilinearKernel<PartialValBilinearForm<1,false>>);
	static_assert(IsBilinearKernel<PartialValBilinearForm<2,false>>);
}