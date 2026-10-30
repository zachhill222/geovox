#pragma once

#include "gutil.hpp"
#include <cstdint>
#include <cassert>

#ifndef GEOVOX_MAX_KERNEL_ARGS
	#define GEOVOX_MAX_KERNEL_ARGS 3
#endif


namespace GV {
namespace FemKernel {

	//a few constants to help maintain consistent test/trial argument conventions
	inline constexpr uint8_t TEST_ARG{0};
	inline constexpr uint8_t TRIAL_ARG{1};

	//tag system to determine if the final expression requires values, gradients, geometric coordinates, etc.
	//this tag is for a single argument (the argument represents a dof_handler. e.g., trial/test functions)
	struct KernelArgumentTag {
		uint8_t tag = 0b00000000;
		static_assert(GEOVOX_MAX_KERNEL_ARGS < 8);

		static constexpr uint8_t VAL_MASK     = 0b00000001;
		static constexpr uint8_t GRAD_MASK    = 0b00000010;
		static constexpr uint8_t JAC_MASK     = 0b00000100;
		static constexpr uint8_t JAC_INV_MASK = 0b00001000;
		static constexpr uint8_t ARG_NO_MASK  = 0b01110000; //which argument this is a tag for
		static constexpr uint8_t USED_MASK    = 0b10000000; //the expressions default to a max of 3 arguments, but not all are always used

		constexpr bool needs_val()     const noexcept {return  tag & VAL_MASK;}
		constexpr bool needs_grad()    const noexcept {return  tag & GRAD_MASK;}
		constexpr bool needs_jac()     const noexcept {return  tag & JAC_MASK;}
		constexpr bool needs_jac_inv() const noexcept {return  tag & JAC_INV_MASK;}
		constexpr bool is_used()       const noexcept {return  tag & USED_MASK;}
		constexpr uint8_t arg_no()     const noexcept {return (tag & ARG_NO_MASK) >> 4;}

		constexpr void set_needs_val(bool a)     noexcept {a ? tag |= VAL_MASK     : tag &= ~VAL_MASK    ;}
		constexpr void set_needs_grad(bool a)    noexcept {a ? tag |= GRAD_MASK    : tag &= ~GRAD_MASK   ;}
		constexpr void set_needs_jac(bool a)     noexcept {a ? tag |= JAC_MASK     : tag &= ~JAC_MASK    ;}
		constexpr void set_needs_jac_inv(bool a) noexcept {a ? tag |= JAC_INV_MASK : tag &= ~JAC_INV_MASK;}
		constexpr void set_used(bool a)          noexcept {a ? tag |= USED_MASK    : tag &= ~USED_MASK;}
		constexpr void set_arg_no(uint8_t a)     noexcept {tag &= ~ARG_NO_MASK; tag |= (ARG_NO_MASK & (a<<4));}

		//join the requirements of two arguments
		friend constexpr KernelArgumentTag operator|(KernelArgumentTag a, KernelArgumentTag b) noexcept {
			if consteval {static_assert(a.arg_no() == b.arg_no());}
			else {assert(a.arg_no() == b.arg_no());}

			return KernelArgumentTag{static_cast<uint8_t>(a.tag | b.tag)};
		}
		friend constexpr KernelArgumentTag operator&(KernelArgumentTag a, KernelArgumentTag b) noexcept {
			if consteval {static_assert(a.arg_no() == b.arg_no());}
			else {assert(a.arg_no() == b.arg_no());}

			return KernelArgumentTag{static_cast<uint8_t>(a.tag & b.tag)};
		}
	};

	//full tag system for tracking all arguments and a couple extras
	//we track information for three arguments, but not all are required.
	struct KernelTag {
		KernelArgumentTag arg_tags[GEOVOX_MAX_KERNEL_ARGS] = {};
		uint8_t node_tag = 0b00000000;

		static constexpr uint8_t GEO_MASK     = 0b00000001;
		static constexpr uint8_t RANK_MASK    = 0b00000110; //scalar/vector/matrix/3rd order tensor
		// static constexpr uint8_t ARITY_MASK   = 0b00011000; //leaf(none)/unary/binary/ternary
		static constexpr uint8_t CONST_MASK   = 0b00100000;	//is the value of this node constant over every quadrature point
		static constexpr uint8_t DIM_MASK     = 0b11000000; //ambient dimension

		constexpr bool needs_geo_pts() const noexcept {return  node_tag & GEO_MASK;}
		constexpr uint8_t rank()       const noexcept {return (node_tag & RANK_MASK) >> 1;}
		// constexpr uint8_t arity()      const noexcept {return (node_tag & ARITY_MASK) >> 3;}
		constexpr bool is_const()      const noexcept {return  node_tag & CONST_MASK;}
		constexpr uint8_t dim()        const noexcept {return (node_tag & DIM_MASK) >> 6;}

		constexpr void set_needs_geo(bool a) noexcept {a ? node_tag |= GEO_MASK : node_tag &= ~GEO_MASK    ;}
		constexpr void set_rank(uint8_t a)   noexcept {node_tag &= ~RANK_MASK;  node_tag |= (RANK_MASK & (a<<1));}
		// constexpr void set_arity(uint8_t a)  noexcept {node_tag &= ~ARITY_MASK; node_tag |= (ARITY_MASK & (a<<3));}
		constexpr void set_const(bool a)     noexcept {a ? node_tag |= CONST_MASK : node_tag &= ~CONST_MASK    ;}
		constexpr void set_dim(uint8_t a)    noexcept {node_tag &= ~DIM_MASK;  node_tag |= (DIM_MASK & (a<<6));}

		constexpr size_t n_components() const noexcept {return gutil::pow((size_t) dim(), (size_t) rank());}

		//get the tag for a specific argument
		constexpr KernelArgumentTag  arg_tag(uint8_t arg) const noexcept {return arg_tags[arg];}
		constexpr KernelArgumentTag& arg_tag(uint8_t arg) noexcept {return arg_tags[arg];}
	};

	//join the argument requirements of two nodes
	inline constexpr KernelTag join(KernelTag a, KernelTag b) noexcept {
		if consteval {static_assert(a.dim()==b.dim());}
		else {assert(a.dim()==b.dim());}

		KernelTag c{};
		for (int n=0; n<GEOVOX_MAX_KERNEL_ARGS; ++n) {c.arg_tags[n] = a.arg_tags[n] | b.arg_tags[n];}
		c.set_needs_geo( a.needs_geo_pts() | b.needs_geo_pts() );
		c.set_const( a.is_const() & b.is_const() );

		//rank and arity must be explicitly set
		return c;
	}

}}