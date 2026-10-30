#pragma once

#include "gutil.hpp"
#include "fem/forms/syntax_tree/argument_tags.hpp"

#include <concepts>

namespace GV {
namespace FemKernel {

	template<typename Expr>
	concept IsKernelExpression = requires (Expr e) {
		{Expr::TAG} -> std::convertible_to<KernelTag>;
		{e.at.template<1>(size_t{0}, size_t{0})};
	};


	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	struct KernelExpressionSum {
		static constexpr KernelTag TAG = [](){
			KernelTag t = join(Left::TAG,Right::TAG);
			t.set_rank(Left::TAG.rank());
			return t;
		}();

		Left left;
		Right right;

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at_impl(size_t component, size_t quad_point) const noexcept {
			return left.template at<NQ>(component,quad_point) + right.template at<NQ>(component,quad_point);
		}
	};
	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	[[nodiscard]] inline constexpr KernelExpressionSum<Left,Right> operator+(Left l, Right r) noexcept {
		return KernelExpressionSum<Left,Right>{l,r};
	}


	template<gutil::IsScalar Scalar, IsKernelExpression Node>
	struct KernelExpressionScalarProduct {
		static constexpr KernelTag TAG = Node::TAG;

		Scalar scalar;
		Node node;

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at_impl(size_t component, size_t quad_point) const noexcept {
			return scalar * node.template at<NQ>(component, quad_point);
		}
	};
	template<gutil::IsScalar Scalar, IsKernelExpression Node>
	[[nodiscard]] inline constexpr KernelExpressionScalarProduct<Scalar,Node> operator*(Scalar s, Node n) noexcept {
		return KernelExpressionScalarProduct<Scalar,Node>{s,n};
	}


	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	struct KernelExpressionContraction {
		static constexpr KernelTag TAG = [](){
			KernelTag t = join(Left::TAG,Right::TAG);
			t.set_rank(0);
			return t;
		}();

		Left left;
		Right right;

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at_impl(size_t component, size_t quad_point) const noexcept {
			GUTIL_ASSERT(component==0);

			return [&]<size_t... Cs>(std::index_sequence<Cs...>) {
				return (... + (left.template at<NQ>(Cs,quad_point) * right.template at<NQ>(Cs,quad_point)));
			}(std::make_index_sequence<Left::TAG.n_components()>{});
		}
	};
	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	[[nodiscard]] inline constexpr KernelExpressionContraction<Left,Right> contract(Left l, Right r) noexcept {
		return KernelExpressionContraction<Left,Right>{l,r};
	}
	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == 1 && Right::TAG.rank() == 1 && Left::TAG.dim() == Right::TAG.dim())
	[[nodiscard]] inline constexpr KernelExpressionContraction<Left,Right> dot(Left l, Right r) noexcept {
		return KernelExpressionContraction<Left,Right>{l,r};
	}


}}