#pragma once

#include "gutil.hpp"
#include "simd_keys/keyed_object.hpp"
#include "simd_keys/mesh/mesh_key_implementation.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"

#include <cstdint>

namespace GV {
namespace Keys {
namespace DOFS {


	////////////////////////////////////////////////////////////
	/// Utility methods. Evaluating a dof requires passing the
	/// support element. Sometimes quadrature elements are more
	/// refined than the support element so that the quadrature
	/// element and points must be "projected" up to the support
	/// element.
	///
	/// All quadrature (mesh) elements are assumed to have period 0.
	////////////////////////////////////////////////////////////
	template<typename T=double>
	[[maybe_unused]] constexpr VoxelElement<0> ProjectQuadratureElementToSupportElement(uint64_t spt_depth, VoxelElement<0> quad, T* X, T* Y, T* Z, uint32_t N) noexcept {
		GUTIL_ASSERT(X && Y && Z && "X,Y,Z must be valid pointers at the start of N contiguous memory locations");
		GUTIL_ASSERT(quad.depth() >= spt_depth && "the quadrature element must be at least as fine as the support element");
		GUTIL_ASSERT(!quad.is_encoded() && "the quadrature element must be in cartesian form. use quad.decode()");

		#ifndef NDEBUG
			T min_x{2}, min_y{2}, min_z{2}, max_x{-2}, max_y{-2}, max_z{-2};
			GUTIL_SIMD(reduction(min:min_x,min_y,min_z) reduction(max:max_x,max_y,max_z))
			for (uint32_t i=0; i<N; ++i) {
				min_x = (X[i] < min_x) ? X[i] : min_x;
				min_y = (Y[i] < min_y) ? Y[i] : min_y;
				min_z = (Z[i] < min_z) ? Z[i] : min_z;
				max_x = (X[i] > max_x) ? X[i] : max_x;
				max_y = (Y[i] > max_y) ? Y[i] : max_y;
				max_z = (Z[i] > max_z) ? Z[i] : max_z;
			}
			GUTIL_ASSERT(min_x>=T{-1} && "the quadrature points must be in [-1,1] for each axis");
			GUTIL_ASSERT(min_y>=T{-1} && "the quadrature points must be in [-1,1] for each axis");
			GUTIL_ASSERT(min_z>=T{-1} && "the quadrature points must be in [-1,1] for each axis");
			GUTIL_ASSERT(max_x<=T{1}  && "the quadrature points must be in [-1,1] for each axis");
			GUTIL_ASSERT(max_y<=T{1}  && "the quadrature points must be in [-1,1] for each axis");
			GUTIL_ASSERT(max_z<=T{1}  && "the quadrature points must be in [-1,1] for each axis");
		#endif

		while (quad.depth() != spt_depth) {
			//determine which octant to project up as
			const T dx = static_cast<T>(quad.i_simd()&1) - T{0.5};
			const T dy = static_cast<T>(quad.j_simd()&1) - T{0.5};
			const T dz = static_cast<T>(quad.k_simd()&1) - T{0.5};

			GUTIL_SIMD()
			for (uint32_t i=0; i<N; ++i) {
				X[i] = T{0.5}*X[i] + dx;
				Y[i] = T{0.5}*Y[i] + dy;
				Z[i] = T{0.5}*Z[i] + dz;
			}

			quad = quad.parent();
		}

		//return the support element
		return quad;
	}

}}}
