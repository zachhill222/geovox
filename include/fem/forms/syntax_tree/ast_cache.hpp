#pragma once

#include "gutil.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	//////////////////////////////////////////////////////
	/// An abstract syntax tree to efficiently do arithmetic
	/// on cached per-quad point values. This allows the user
	/// to easily define kernels for bilinear forms and similar
	/// while maintaining efficient evaluations.
	///
	/// This is accomplishes similar goals as using UFL in FENICs.
	//////////////////////////////////////////////////////
	template<typename Node>
	concept IsTensorAstNode = requires (const Node n) {
		typename Node::Scalar_t;
		{Node::N_QUAD}          -> std::convertible_to<size_t>;
		{Node::RANK}            -> std::convertible_to<size_t>;
		{Node::DIM}             -> std::convertible_to<size_t>;
		{Node::COMPONENT_COUNT} -> std::convertible_to<size_t>;
		{Node::IS_CONSTANT}     -> std::convertible_to<bool>;
		{n.at(size_t{0}, size_t{0})} -> std::convertible_to<typename Node::Scalar_t>;
	};


	//////////////////////////////////////////////////////
	/// The base node class. Tracks what information is stored at
	/// each node of the AST. The node represents a rank-K tensor
	/// with a method on how to compute it from its leaves.
	///
	/// If Rank=2, the matrix must be in column-major format at each quad point
	//////////////////////////////////////////////////////
	template<typename Scalar, size_t N, size_t Dim, size_t Rank, bool IsConstant>
	struct TensorAstNode {
		static_assert(Rank<3);
		using Scalar_t = Scalar;
		static constexpr size_t N_QUAD = N;                                //number of quadrature points
		static constexpr size_t RANK = Rank;                               //0,1,2 (scalar/vector/matrix) needed at each quad point
		static constexpr size_t DIM = Dim;                                 //ambient dimension (usually 3)
		static constexpr size_t COMPONENT_COUNT = gutil::pow(Dim, Rank);   //matrices are always square, vectors are always length DIM
		static constexpr bool IS_CONSTANT = IsConstant;                    //track if the value at different quadrature points are different
		static constexpr size_t PER_COMPONENT_STRIDE = IsConstant ? 1 : N; //stride in the data to get from tensor component at quad point i to quad point i+1

		static constexpr size_t N_DATA = PER_COMPONENT_STRIDE*COMPONENT_COUNT;	//size of data needed
		[[maybe_unused]] const Scalar* const data_{nullptr}; //must point to N_DATA data, only used for leaf nodes

		constexpr TensorAstNode(const Scalar* ptr=nullptr) : data_{ptr} {}

		const Scalar* data() const noexcept {return data_;}

		//leaf access is the default
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar at(size_t component, size_t i) const noexcept {
			if constexpr (IS_CONSTANT) {
				return data_[component];
			}
			else {
				return data_[PER_COMPONENT_STRIDE*component + i];
			}
		}
	};
	static_assert(IsTensorAstNode<TensorAstNode<float,4,3,0,false>>); //test the concept

	//////////////////////////////////////////////////////
	/// "Componentwise" node types for operations on a single node
	//////////////////////////////////////////////////////
	template<IsTensorAstNode Node>
	struct UnaryAstNode : public TensorAstNode<typename Node::Scalar_t,Node::N_QUAD,Node::DIM,Node::RANK,Node::IS_CONSTANT> {
		using BASE = TensorAstNode<typename Node::Scalar_t,Node::N_QUAD,Node::DIM,Node::RANK,Node::IS_CONSTANT>;
		using Scalar_t = typename Node::Scalar_t;
		constexpr UnaryAstNode(Node n) noexcept : node{n} {}

		Node node{};
	};


	template<IsTensorAstNode Node>
	struct NegateAstNode : public UnaryAstNode<Node> {
		using BASE = UnaryAstNode<Node>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			return -BASE::node.at(component,i);
		}
	};
	template<IsTensorAstNode Node>
	[[nodiscard]] inline constexpr NegateAstNode<Node> operator-(Node node) noexcept {
		return NegateAstNode<Node>{node};
	}


	template<IsTensorAstNode Node>
	struct ScaledAstNode : public UnaryAstNode<Node> {
		using BASE = UnaryAstNode<Node>;
		using Scalar_t = typename BASE::Scalar_t;
		
		constexpr ScaledAstNode(Scalar_t s, Node n) noexcept: BASE(n), scale{s} {}

		const Scalar_t scale;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			return scale * BASE::node.at(component,i);
		}
	};
	template<IsTensorAstNode Node>
	[[nodiscard]] inline constexpr ScaledAstNode<Node> operator*(typename Node::Scalar_t s, Node node) noexcept {
		return ScaledAstNode<Node>{s,node};
	}


	template<auto lambda, IsTensorAstNode Node>
	struct UnaryLambdaAstNode : public UnaryAstNode<Node> {
		//for lambdas that don't need captures (e.g., [](Scalar_t a){return a*a;})
		using BASE = UnaryAstNode<Node>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			return lambda(BASE::node.at(component,i));
		}
	};
	template<auto lambda, IsTensorAstNode Node>
	[[nodiscard]] inline constexpr UnaryLambdaAstNode<lambda,Node> apply_lambda(Node node) noexcept {
		return UnaryLambdaAstNode<lambda,Node>{node};
	}


	template<typename Lambda, IsTensorAstNode Node>
	struct UnaryRuntimeLambdaAstNode : public UnaryAstNode<Node> {
		//for lambdas that don't need captures (e.g., [](Scalar_t a){return a*a;})
		using BASE = UnaryAstNode<Node>;
		using Scalar_t = typename BASE::Scalar_t;
		
		UnaryRuntimeLambdaAstNode(Lambda lam, Node n) : BASE(n), lambda{std::move(lam)} {}

		Lambda lambda;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			return lambda(BASE::node.at(component,i));
		}
	};
	template<typename Lambda, IsTensorAstNode Node>
	[[nodiscard]] inline constexpr UnaryRuntimeLambdaAstNode<std::decay_t<Lambda>,Node> apply_lambda(Lambda&& lam, Node node) noexcept {
		return UnaryRuntimeLambdaAstNode<std::decay_t<Lambda>,Node>{std::forward<Lambda>(lam), node};
	}


	//////////////////////////////////////////////////////
	/// "Arithmetic" node types for two TensorAstNodes of the same type
	//////////////////////////////////////////////////////
	template<size_t Rank, IsTensorAstNode Left, IsTensorAstNode Right> 
			requires(std::same_as<typename Left::Scalar_t,typename Right::Scalar_t> && 
				Left::N_QUAD==Right::N_QUAD && Left::DIM==Right::DIM)
	struct BinaryAstNode : public TensorAstNode<typename Left::Scalar_t,Left::N_QUAD,Left::DIM,Rank, (Left::IS_CONSTANT && Right::IS_CONSTANT)> {
		using BASE = TensorAstNode<typename Left::Scalar_t,Left::N_QUAD,Left::DIM,Rank, (Left::IS_CONSTANT && Right::IS_CONSTANT)>;
		using Scalar_t = typename Left::Scalar_t;
		constexpr BinaryAstNode(Left l, Right r) noexcept : left{l}, right{r} {}

		Left left;
		Right right;
	};

	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	struct SumAstNode : public BinaryAstNode<Left::RANK,Left,Right> {
		using BASE = BinaryAstNode<Left::RANK,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			return BASE::left.at(component,i) + BASE::right.at(component,i);
		}
	};
	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	[[nodiscard]] inline constexpr SumAstNode<Left,Right> operator+(Left l, Right r) noexcept {
		return SumAstNode<Left,Right>{l,r};
	}


	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	struct SubtractAstNode : public BinaryAstNode<Left::RANK,Left,Right> {
		using BASE = BinaryAstNode<Left::RANK,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			return BASE::left.at(component,i) - BASE::right.at(component,i);
		}
	};
	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	[[nodiscard]] inline constexpr SubtractAstNode<Left,Right> operator-(Left l, Right r) noexcept {
		return SubtractAstNode<Left,Right>{l,r};
	}


	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	struct ComponentProductAstNode : public BinaryAstNode<Left::RANK,Left,Right> {
		using BASE = BinaryAstNode<Left::RANK,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			return BASE::left.at(component,i) * BASE::right.at(component,i);
		}
	};
	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	[[nodiscard]] inline constexpr ComponentProductAstNode<Left,Right> operator*(Left l, Right r) noexcept {
		return ComponentProductAstNode<Left,Right>{l,r};
	}


	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	struct ContractionAstNode : public BinaryAstNode<0,Left,Right> {
		using BASE = BinaryAstNode<0,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr Scalar_t at(size_t component, size_t i) const noexcept {
			Scalar_t val{0};
			//TODO: use an index_sequence to guarantee compile-time unroll if needed
			//Note COMPONENT_COUNT should be at most 9 and is known at compile-time.
			for (size_t c=0; c<Left::COMPONENT_COUNT; ++c) {val += BASE::left.at(c,i)*BASE::right.at(c,i);}
			return val;
		}
	};
	
	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==Right::RANK)
	[[nodiscard]] inline constexpr ContractionAstNode<Left,Right> contract(Left l, Right r) noexcept {
		return ContractionAstNode<Left,Right>{l,r};
	}

	template<IsTensorAstNode Left, IsTensorAstNode Right> requires(Left::RANK==1 && Right::RANK==1)
	[[nodiscard]] inline constexpr ContractionAstNode<Left,Right> dot(Left l, Right r) noexcept {
		return ContractionAstNode<Left,Right>{l,r};
	}


	//////////////////////////////////////////////////////
	/// Root operations
	/// Return a scalar or an array in column-major format
	//////////////////////////////////////////////////////
	template<typename QR>
	[[nodiscard]] inline constexpr typename QR::Scalar_t element_volume(const QR& qr) noexcept {
		//helper method
		using Scalar_t = typename QR::Scalar_t;
		static constexpr Scalar_t wt_sum = [](){
			Scalar_t val{0};
			for (int i=0; i<QR::TOTAL_QUAD_POINTS; ++i) {val += QR::quad_w()[i];}
			return val;
		}();

		return wt_sum * qr.jac_det();
	}


	template<IsTensorAstNode Root, typename QR> requires(Root::N_QUAD == (size_t)QR::TOTAL_QUAD_POINTS)
	[[nodiscard]] constexpr auto integrate(const Root& root, const QR& qr) noexcept {
		using Scalar_t = typename QR::Scalar_t;
		constexpr int N = QR::TOTAL_QUAD_POINTS;
		static constexpr auto qw = QR::quad_w();

		if constexpr (Root::RANK==0) {
			Scalar_t val{0};
			GUTIL_SIMD(reduction(+:val))
			for (int i=0; i<N; ++i) {val += root.at(0,i) * qw[i];}
			return val * qr.jac_det();
		}
		else {
			std::array<Scalar_t,Root::COMPONENT_COUNT> result;
			for (size_t c=0; c<Root::COMPONENT_COUNT; ++c) {
				Scalar_t val{0};
				GUTIL_SIMD(reduction(+:val))
				for (int i=0; i<N; ++i) { val += root.at(c,i) * qw[i];}
				result[c] = val;
			}
			return result;
		}
	}
}