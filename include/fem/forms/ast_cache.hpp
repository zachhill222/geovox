#pragma once

#include "gutil.hpp"

namespace GV {


	//////////////////////////////////////////////////////
	/// An abstract syntax tree to efficiently do arithmetic
	/// on cached per-quad point values. This allows the user
	/// to easily define kernels for bilinear forms and similar
	/// while maintaining efficient evaluations.
	///
	/// This is accomplishes similar goals as using UFL in FENICs.
	//////////////////////////////////////////////////////


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
		[[maybe_unused]] Scalar* data_{nullptr}; //must point to N_DATA data, only used for leaf nodes

		[[nodiscard]] static constexpr size_t ComponentStart(size_t component) noexcept {
			GUTIL_ASSERT(component<COMPONENT_COUNT);
			if constexpr (IsConstant) {return 0;}
			else {return PER_COMPONENT_STRIDE * component;}
		}

		constexpr TensorAstNode(Scalar* ptr=nullptr) : data_{ptr} {}

		//leaf access is the default
		[[nodiscard]] constexpr Scalar at(size_t component, size_t i) const noexcept {
			if constexpr (IS_CONSTANT) {
				return data_[component];
			}
			else {
				return data_[PER_COMPONENT_STRIDE*component + i];
			}
		}
	};


	//////////////////////////////////////////////////////
	/// "Arithmetic" node types for two TensorAstNodes of the same type
	//////////////////////////////////////////////////////
	template<size_t Rank, typename Left, typename Right> requires(std::same_as<typename Left::Scalar_t,typename Right::Scalar_t> && 
														Left::N_QUAD==Right::N_QUAD && Left::DIM==Right::DIM)
	struct BinaryAstNode : public TensorAstNode<typename Left::Scalar_t,Left::N_QUAD,Left::DIM,Rank, (Left::IS_CONSTANT && Right::IS_CONSTANT)> {
		using BASE = TensorAstNode<typename Left::Scalar_t,Left::N_QUAD,Left::DIM,Rank, (Left::IS_CONSTANT && Right::IS_CONSTANT)>;
		using Scalar_t = typename Left::Scalar_t;
		BinaryAstNode(Left l, Right r) : left{l}, right{r} {}

		Left left{};
		Right right{};
	};

	template<typename Left, typename Right> requires(Left::RANK==Right::RANK)
	struct SumAstNode : public BinaryAstNode<Left::RANK,Left,Right> {
		using BASE = BinaryAstNode<Left::RANK,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		[[nodiscard]] Scalar_t at(size_t component, size_t i) const noexcept {
			return left.at(component,i) + right.at(component,i);
		}
	};

	template<typename Left, typename Right> requires(Left::RANK==Right::RANK)
	struct SubtractAstNode : public BinaryAstNode<Left::RANK,Left,Right> {
		using BASE = BinaryAstNode<Left::RANK,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		[[nodiscard]] Scalar_t at(size_t component, size_t i) const noexcept {
			return left.at(component,i) - right.at(component,i);
		}
	};

	template<typename Left, typename Right> requires(Left::RANK==Right::RANK)
	struct ComponentProductAstNode : public BinaryAstNode<Left::RANK,Left,Right> {
		using BASE = BinaryAstNode<Left::RANK,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		[[nodiscard]] Scalar_t at(size_t component, size_t i) const noexcept {
			return left.at(component,i) * right.at(component,i);
		}
	};

	template<typename Left, typename Right> requires(Left::RANK==Right::RANK)
	struct ContractionAstNode : public BinaryAstNode<0,Left,Right> {
		using BASE = BinaryAstNode<0,Left,Right>;
		using Scalar_t = typename BASE::Scalar_t;
		using BASE::BASE;

		[[nodiscard]] Scalar_t at(size_t component, size_t i) const noexcept {
			Scalar_t val{0};
			//TODO: use an index_sequence to guarantee compile-time unroll if needed
			//Note COMPONENT_COUNT should be at most 9 and is known at compile-time.
			for (size_t c=0; c<Left::COMPONENT_COUNT; ++c) {val += left.at(c,i)*right.at(c,i);}
			return val;
		}
	};
	





}