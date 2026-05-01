#pragma once

#include "mesh/voxel_mesh.hpp"
#include "fem/handlers/dofhandler_base.hpp"

#include <type_traits>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <span>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	template<VoxelMeshType Mesh_type, typename DOF_type>
	class DofHandlerCharms : public DofHandlerBase<Mesh_type, DOF_type>
	{
	public:
		using BASE       = DofHandlerBase<Mesh_type, DOF_type>;

		using DOF_t      = DOF_type;
		using QuadElem_t = typename DOF_t::QuadElem_t;
		using DOFKey_t   = typename DOF_t::Key_t;
		using MeshKey_t  = typename DOF_t::Key_t::NonPeriodicType;
		using Mesh_t     = Mesh_type;
		using Elem_t     = typename Mesh_t::VoxelElement;
		using Vert_t     = typename Mesh_t::VoxelVertex;
		using Face_t     = typename Mesh_t::VoxelFace;

		//note that the DOF feature type (including QuadElem) may be periodic
		//while the mesh Elem_t is not periodic. Once constructed, the DOF support
		//and children keys and so on can be safely cast to the mesh version with
		//static_cast.
		
		//inherit constructor
		using BASE::BASE;

		//prolong or contract a vector from the old grid to the new grid
		//the old grid dofs must be in active_dof_list_prev and the new grid
		//dofs must be in active_dof_list_curr. This will happen when
		//compress_dof_numbers() is called on the current grid.
		//the grid does not have to be uniformly refined/coarsened
		void transfer_grid(std::span<double> new, std::span<const double> old) const {
			const auto& old_dofs = this->prev_compressed_dofs();
			const auto& new_dofs = this->curr_compressed_dofs();

			assert(old_coefs.size() == old_dofs.size());
			assert(new_coefs.size() == new_dofs.size());
			assert(new_coefs.size() == this->n_dofs());

			
		}



	};
}