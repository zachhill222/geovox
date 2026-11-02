#pragma once

#include "fem/forms/syntax_tree/argument_tags.hpp"
#include "fem/forms/syntax_tree/fem_kernel_tree.hpp"	
#include "fem/forms/syntax_tree/fem_kernel_leaves.hpp"


namespace GV {
namespace FemKernel {


	//////////////////////////////////////////////////////////////////////////
	/// Class to manage passing arguments to the kernel tree and final integration
	/// over the element
	//////////////////////////////////////////////////////////////////////////
	// template<IsKernelExpression Expr, typename QuadRule>
	// struct FormKernel {
	// 	using Scalar_t = typename QuadRule::Scalar_t;

	// 	std::array<const Scalar_t*,GEOVOX_MAX_KERNEL_ARGS> val_ptrs{};
	// 	std::array<const Scalar_t*,GEOVOX_MAX_KERNEL_ARGS> grad_ptrs{};
		
	// 	std::vector<const Scalar_t*> other_ptrs(0);

	// 	//store an instance of the final expression
	// 	Expr expr;

	// private:

	// };







}}