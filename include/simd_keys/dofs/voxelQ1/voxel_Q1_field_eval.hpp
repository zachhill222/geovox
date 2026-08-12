#pragma once

#include "gutil.hpp"

#include "simd_keys/mesh/mesh_keys.hpp"
#include "simd_keys/dofs/voxelQ1/voxel_Q1_interface.hpp"
#include "util/util.hpp"

#include <span>
#include <cstdint>

namespace GV {
namespace Keys {
namespace DOFS {



	///////////////////////////////////////////////////////////////////////
	/// Given a list of VoxelQ1 dofs all at a single depth, coefficients
	/// for each dof, and a list of locations, evaluate the function
	/// f(x) = sum_i coef[i]*dof[i](x) at each location. Here, the locations
	/// are restricted to vertex locations, but may be at any depth.
	///
	/// We assume that the dofs are sorted so that std::lower_bound may be used.
	///////////////////////////////////////////////////////////////////////


	template<typename T, uint8_t Period, typename DOF_t> requires(Period<8)
	[[nodiscard]] inline constexpr T EvaluateVoxelQ1Field_SingleDepth(
		VoxelVertex<Period> loc,
		std::span<const T> coef, 
		std::span<const VoxelQ1<Period>> dofs,
		uint64_t depth) {

		GUTIL_ASSERT(coef.size()==dofs.size());

		using Vert_t = VoxelVertex<Period>;
		using Elem_t = VoxelElement<Period>;

		if (loc.depth() <= depth) {
			// the location correpsonds to a dof or is not in the support of any dof
			while (loc.depth() < depth) {loc = loc.child();}

			DOF_t target_dof{loc.key};
			auto it = std::lower_bound(dofs.begin(), dofs.end(), target_dof);
			if (it==dofs.end() || *it!=target_dof) {return T{0};}
			else {return coef[std::distance(dofs.begin(),it)];}
		}
		else {
			// the the location is in the interior of some support element for up to 8 dofs
			// the support element can be recovered by the index arithmetic.
			// note that if the vertex is on an edge or face, any of the adjacent support elements
			// will work.
			Elem_t spt{Mesh3D::VertexSubgridElement<Period>(loc.key, depth)};
			Vert_t rel_loc{Mesh3D::RelativeVertexInSubgrid<Period>(loc.key, spt.key)};
			gutil::Point<3,T> coord = T{2} * rel_loc.template normalized_coordinate<T>() - gutil::Point<3,T>::Filled(1);
			
			DOF_t target_dof[DOF_t::N_DOF_PER_ELEM];
			DOF_t::dofs_on_elem_simd(spt.key, target_dof);
			T result{0};
			for (int i=0; i<DOF_t::N_DOF_PER_ELEM; ++i) {
				auto it = std::lower_bound(dofs.begin(), dofs.end(), target_dof[i]);
				if (it!=dofs.end() && *it==target_dof[i]) {
					size_t idx = std::distance(dofs.begin(), it);
					result += coef[idx] * it->evaluate(spt, coord);
				}
			}
			return result;
		}
	}
}}}








