#pragma once

#include "gutil.hpp"
#include "fem/forms/syntax_tree/argument_tags.hpp"

#include <concepts>

namespace GV {
namespace FemKernel {

	template<typename Expr>
	concept IsKernelExpression = requires (Expr e) {
		{Expr::TAG} -> std::convertible_to<KernelTag>;
		{e.template at<1>(size_t{0}, size_t{0})};
	};


	///////////////////////////////////////////////////////////
	/// A few base classes to help with passing data arguments
	///////////////////////////////////////////////////////////
	template<IsKernelExpression Left, IsKernelExpression Right>
	struct BinaryKernelExpression {
		static constexpr KernelTag TAG = [](){
			KernelTag t = join(Left::TAG,Right::TAG);
			t.set_rank(Left::TAG.rank());
			return t;
		}();

		Left left;
		Right right;


		template<typename Scalar>
		constexpr void set_data(const Scalar* ptr, const KernelArgumentTag tag) noexcept {
			//set the argument number and use of the tag (value/gradient/jacobian/jacobian_inverse)
			if ( (Left::TAG.arg_tag(tag.arg_no())&tag) == tag ) {left.set_data(ptr,tag);}
			if ( (Right::TAG.arg_tag(tag.arg_no())&tag) == tag ) {right.set_data(ptr,tag);}
		}


		template<typename Scalar>
		constexpr void set_geo_data(const Scalar* ptr) noexcept {
			if (Left::TAG.needs_geo_pts()) {left.set_geo_data(ptr);}
			if (Right::TAG.needs_geo_pts()) {right.set_geo_data(ptr);}
		}
	};


	template<IsKernelExpression Node>
	struct UnaryKernelExpression {
		static constexpr KernelTag TAG = Node::TAG;

		Node node;

		template<typename Scalar>
		constexpr void set_data(const Scalar* ptr, const KernelArgumentTag tag) noexcept {
			//set the argument number and use of the tag (value/gradient/jacobian/jacobian_inverse)
			GUTIL_ASSERT( (Node::TAG.arg_tag(tag.arg_no())&tag) == tag );
			node.set_data(ptr,tag);
		}

		template<typename Scalar>
		constexpr void set_geo_data(const Scalar* ptr) noexcept {
			GUTIL_ASSERT(Node::TAG.needs_geo_pts());
			node.set_geo_data(ptr);
		}
	};


	///////////////////////////////////////////////////////////
	/// Binary arithmetic expressions
	///////////////////////////////////////////////////////////
	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	struct KernelExpressionSum : public BinaryKernelExpression<Left,Right> {
		using BASE = BinaryKernelExpression<Left,Right>;
		using BASE::TAG;

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			return this->left.template at<NQ>(component,quad_point) + this->right.template at<NQ>(component,quad_point);
		}
	};
	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	[[nodiscard]] inline constexpr KernelExpressionSum<Left,Right> operator+(Left l, Right r) noexcept {
		return KernelExpressionSum<Left,Right>{l,r};
	}


	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	struct KernelExpressionDifference : public BinaryKernelExpression<Left,Right> {
		using BASE = BinaryKernelExpression<Left,Right>;
		using BASE::TAG;

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			return this->left.template at<NQ>(component,quad_point) - this->right.template at<NQ>(component,quad_point);
		}
	};
	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	[[nodiscard]] inline constexpr KernelExpressionDifference<Left,Right> operator-(Left l, Right r) noexcept {
		return KernelExpressionDifference<Left,Right>{l,r};
	}


	template<IsKernelExpression Left, IsKernelExpression Right> requires(Left::TAG.rank() == Right::TAG.rank() && Left::TAG.dim() == Right::TAG.dim())
	struct KernelExpressionContraction : BinaryKernelExpression<Left,Right> {
		using BASE = BinaryKernelExpression<Left,Right>;
		using BASE::TAG;

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			GUTIL_ASSERT(component==0);

			return [&]<size_t... Cs>(std::index_sequence<Cs...>) {
				return (... + (this->left.template at<NQ>(Cs,quad_point) * this->right.template at<NQ>(Cs,quad_point)));
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


	///////////////////////////////////////////////////////////
	/// Unary arithmetic expressions
	///////////////////////////////////////////////////////////
	template<gutil::IsScalar Scalar, IsKernelExpression Node>
	struct KernelExpressionScalarProduct : UnaryKernelExpression<Node> {
		using BASE = UnaryKernelExpression<Left,Right>;
		using BASE::TAG;

		Scalar scalar;

		KernelExpressionScalarProduct(Scalar s, Node n) : BASE{n}, scalar{s} {}

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			return scalar * this->node.template at<NQ>(component, quad_point);
		}
	};
	template<gutil::IsScalar Scalar, IsKernelExpression Node>
	[[nodiscard]] inline constexpr KernelExpressionScalarProduct<Scalar,Node> operator*(Scalar s, Node n) noexcept {
		return KernelExpressionScalarProduct<Scalar,Node>{s,n};
	}


	template<typename Lambda, IsKernelExpression Node> requires(std::is_empty_v<Lambda>)
	struct KernelExpressionCompileTimeLambda : public UnaryKernelExpression<Node> {
		using BASE = UnaryKernelExpression<Left,Right>;
		using BASE::TAG;

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			return Lambda{}(this->node.template at<NQ>(component, quad_point));
		}
	};
	template<typename Lambda, IsKernelExpression Node> requires(std::is_empty_v<Lambda>)
	[[nodiscard]] inline constexpr KernelExpressionCompileTimeLambda<Lambda,Node> apply(Lambda, Node n) noexcept {
		return KernelExpressionCompileTimeLambda<Lambda,Node>{n};
	}


	template<typename Lambda, IsKernelExpression Node> requires(!std::is_empty_v<Lambda>)
	struct KernelExpressionRuntimeLambda {
		using BASE = UnaryKernelExpression<Left,Right>;
		using BASE::TAG;

		KernelExpressionRuntimeLambda(Lambda l, Node n) : BASE{n}, lambda{std::move(l)} {}

		Lambda lambda;
		
		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			return lambda(this->node.template at<NQ>(component, quad_point));
		}
	};
	template<typename Lambda, IsKernelExpression Node> requires(!std::is_empty_v<Lambda>)
	[[nodiscard]] inline constexpr auto apply(Lambda&& lambda, Node n) noexcept {
		return KernelExpressionRuntimeLambda<std::decay_t<Lambda>,Node>{std::forward<Lambda>(lambda), n};
	}




}}