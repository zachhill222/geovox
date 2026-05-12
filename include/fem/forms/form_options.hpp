#pragma once

#include <concepts>

namespace GV
{
	//options for bilinear forms
	//tells the form at compile-time what actions it will be taking
	struct BilinearFormOptions
	{
		//assemble sparse matrix, do a matrix-vector multiply, or solve
		enum class Action {Assemble, MatVec};

		//specific opertion should be called by the kernel (somewhat redundant)
		enum class Operation {Assemble, Multiply, Jacobi, GaussSeidel_F, GaussSeidel_B};

		//do we accumulate into the global vector or set the global vector
		enum class ScatterMode {Accumulate, Set};

		//is the bilinear form symmetric?
		bool is_symmetric;
		Action action;
		Operation operation;
		ScatterMode scatter_mode;

		//factory for assembling the global matrix
		static constexpr BilinearFormOptions assemble(const bool is_sym) {
			return {is_sym, Action::Assemble, Operation::Assemble, ScatterMode::Accumulate};
		}

		//factory for computing a matrix-vector product
		//note that accumulating is better for iterative solvers
		static constexpr BilinearFormOptions multiply(const bool is_sym) {
			return {is_sym, Action::MatVec, Operation::Multiply, ScatterMode::Accumulate};
		}

		//factories for iterative methods
		static constexpr BilinearFormOptions gauss_seidel_fwd(const bool is_sym) {
			return {is_sym, Action::MatVec, Operation::GaussSeidel_F, ScatterMode::Set};
		}

		static constexpr BilinearFormOptions gauss_seidel_bwd(const bool is_sym) {
			return {is_sym, Action::MatVec, Operation::GaussSeidel_B, ScatterMode::Set};
		}
	};

	//options for linear forms
	//tells the form at compile-time what actions it will be taking
	struct LinearFormOptions
	{
		//assemble a global list or just compute the action on a given vector
		enum class Action {Assemble, Dot};

		//do we accumulate into the global vector or set the global vector
		enum class ScatterMode {Accumulate, Set};

		const Action action;
		const ScatterMode scatter_mode;

		//factory for assembling a global vector
		static constexpr LinearFormOptions assemble() {
			return {Action::Assemble, ScatterMode::Accumulate};
		}

		//factory for computing the action on a given vector
		static constexpr LinearFormOptions dot() {
			return {Action::Dot, ScatterMode::Accumulate};
		}
	};
}