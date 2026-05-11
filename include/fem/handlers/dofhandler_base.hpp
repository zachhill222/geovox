#pragma once

#include "mesh/voxel_mesh.hpp"

#include <type_traits>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	template<VoxelMeshType Mesh_type, typename DOF_type>
	class DofHandlerBase
	{
	public:
		using DOF_t      = DOF_type;
		using QuadElem_t = typename DOF_t::QuadElem_t;
		using DOFKey_t   = typename DOF_t::Key_t;
		using MeshKey_t  = typename DOF_t::Key_t::NonPeriodicVariant;
		using Mesh_t     = Mesh_type;
		using Elem_t     = typename Mesh_t::VoxelElement;
		using Vert_t     = typename Mesh_t::VoxelVertex;
		using Face_t     = typename Mesh_t::VoxelFace;

		//note that the DOF feature type (including QuadElem) may be periodic
		//while the mesh Elem_t is not periodic. Once constructed, the DOF support
		//and children keys and so on can be safely cast to the mesh version with
		//static_cast.

		static_assert(
			VoxelEquivFeature<MeshKey_t, typename Mesh_t::VoxelElement> ||
			VoxelEquivFeature<MeshKey_t, typename Mesh_t::VoxelFace>    ||
			VoxelEquivFeature<MeshKey_t, typename Mesh_t::VoxelVertex>,
			"DofHandlerBase - the feature key for the DOF must match the corresponding feature key of the mesh.");
		static_assert(VoxelEquivFeature<Elem_t, typename DOF_type::QuadElem_t::NonPeriodicVariant>,
			"DofHandlerBase - the DOF quadrature element and the mesh element must be of an equivalent type");

		//we can iterate over the mesh and REQUEST mesh refinement
		//however, mesh refinement must be done outside of this class
		//because multiple DOF handlers can reference the same mesh
		static constexpr uint64_t MAX_DEPTH = Mesh_t::MAX_DEPTH;
		const Mesh_t& mesh;

		static constexpr uint64_t TOTAL_POSSIBLE_DOFS = total_possible<MeshKey_t>(MAX_DEPTH);
		static constexpr bool OPENMP = Mesh_t::OPENMP;

		//constructor and destructor
		explicit DofHandlerBase(const Mesh_t& mesh) : mesh(mesh) {}
		virtual ~DofHandlerBase() {
			delete active_dofs;
		}

		//non-copyable
		DofHandlerBase(const DofHandlerBase&) = delete;
		DofHandlerBase& operator=(const DofHandlerBase&) = delete;

		//movable by construction
		DofHandlerBase(DofHandlerBase&& other) :
			mesh(other.mesh), 
			active_dofs(other.active_dofs)
		{
			other.active_dofs = nullptr;
		}

		//can't move by assignment
		DofHandlerBase& operator=(DofHandlerBase&&) = delete;

		protected:
		//track which dofs are active based on a global numbering system of the voxel features.
		//instances of the DOF_t act as an iterator into this set.
		std::bitset<TOTAL_POSSIBLE_DOFS>* active_dofs = new std::bitset<TOTAL_POSSIBLE_DOFS>(0);

		//track a compressed list of active dofs
		//the bitfield is the "source of truth"
		//but this is used to convert the global dof number
		//into the dof number that the matrix solver will use
		//maintain a vector of active dofs and a map for O(1) lookups
		//note that eigen uses ints for its index, but this could be changed.
		std::vector<DOF_t> active_dof_list_prev;
		std::vector<DOF_t> active_dof_list_curr;
		std::unordered_map<DOF_t, uint64_t, typename DOF_t::Hash> dof_to_idx_map;

		public:
		//simple queries
		inline constexpr uint64_t n_dofs() const {return active_dofs->count();}

		inline constexpr bool is_active(const DOF_t dof) const {assert(dof.is_valid()); return active_dofs->test(dof.key.linear_index());}
		inline constexpr void set_active(const DOF_t dof, const bool b) {assert(dof.is_valid()); active_dofs->set(dof.key.linear_index(), b);}

		inline constexpr bool is_active(const uint64_t idx) const {assert(idx<TOTAL_POSSIBLE_DOFS); return active_dofs->test(idx);}
		inline constexpr void set_active(const uint64_t idx, const bool b) {assert(idx<TOTAL_POSSIBLE_DOFS); active_dofs->set(idx, b);}

		inline const auto& prev_compressed_dofs() const {return active_dof_list_prev;}
		inline const auto& curr_compressed_dofs() const {return active_dof_list_curr;}
		
		inline const auto& dof_to_idx() 		  	  const {return dof_to_idx_map;}
		inline uint64_t compressed_index(const DOF_t dof) const {
			const auto it = dof_to_idx_map.find(dof);
			return it    != dof_to_idx_map.end() ? it->second : uint64_t(-1);
		}
		inline DOF_t get_dof(const uint64_t idx) const {assert(idx<active_dof_list_curr.size()); return active_dof_list_curr[idx];}

		//simple management operations
		inline void set_all_inactive() 	{active_dofs->reset();}

		//activate all dofs at a certain depth if they have an active support element all other dofs are inactive.
		//that this method will only be called when the user intends to "set/reset" the problem.
		void set_depth(const uint64_t dd) {
			active_dofs->reset();

			auto action = [this](MeshKey_t key) {
				const DOFKey_t dof_key = static_cast<DOFKey_t>(key);

				//check if the dof can be placed on this feature
				//this might not happen near periodic boundaries where
				//the lower index is used
				if (key != static_cast<MeshKey_t>(dof_key)) {return;}

				const DOF_t dof{dof_key};
				if (has_active_support(dof)) {
					active_dofs->set(dof.linear_index());
				}
			};

			#ifdef _OPENMP
			mesh.template for_each_depth_omp<MeshKey_t>(dd,action);
			#else
			mesh.template for_each_depth<MeshKey_t>(dd,action);
			#endif

			//TODO: make this better if needed
			compress_dof_numbers();
		}

		//check if a dof has an active support element
		bool has_active_support(const DOF_t dof) const {
			for (auto el : dof.support()) {
				if (el.exists() and mesh.is_active(static_cast<Elem_t>(el))) {
					return true;
				}
			}
			return false;
		}

		bool has_active_basis(const Elem_t el) const {
			for (DOF_t dof : DOF_t::dofs_on_elem(el)) {
				if (dof.exists() and is_active(dof)) {
					return true;
				}
			}
			return false;
		}

		//transfer computations between mesh refinements
		void compress_dof_numbers() {
			const uint64_t ndofs = n_dofs();
			active_dof_list_prev = std::move(active_dof_list_curr);
			active_dof_list_curr.clear();
			active_dof_list_curr.reserve(ndofs);

			#ifdef _OPENMP
			std::vector<std::vector<DOF_t>> local_lists(omp_get_max_threads());
			#else
			std::vector<std::vector<DOF_t>> local_lists(1);
			#endif

			auto action = [this, &local_lists](const MeshKey_t key, const int tid=0) {
				if (active_dofs->test(key.linear_index())) {
					local_lists[tid].emplace_back(key);
				}
			};

			for (uint64_t dd=0; dd<MAX_DEPTH+1; ++dd) {
				if constexpr (OPENMP) {mesh.template for_each_depth_omp<MeshKey_t>(dd,action);}
				else {mesh.template for_each_depth<MeshKey_t>(dd,action);}

				for (auto& list : local_lists) {
					active_dof_list_curr.insert(active_dof_list_curr.end(),
						std::make_move_iterator(list.begin()),
						std::make_move_iterator(list.end()));

					list.clear();
				}

				if (active_dof_list_curr.size() == ndofs) {break;}
			}

			//TODO: allow a custom sorting comparator
			//dofs are already sorted within each thread vector by their global linear index
			// std::sort(active_dof_list_curr.begin(), active_dof_list_curr.end());

			//build the map
			dof_to_idx_map.clear();
			for (uint64_t i=0; i<active_dof_list_curr.size(); ++i) {
				dof_to_idx_map[active_dof_list_curr[i]] = i;
			}
		}

		template<typename CoefContainer_t, typename EvalMethod>
		void init_coefs_by_dof(CoefContainer_t& coefs, EvalMethod&& eval) const {
			assert(static_cast<size_t>(coefs.size()) == active_dof_list_curr.size());
			for (uint64_t i=0; i<static_cast<uint64_t>(coefs.size()); ++i) {
				const DOF_t dof = active_dof_list_curr[i];
				assert(dof.exists());
				assert(dof.is_valid());
				assert(is_active(dof));
				coefs[i] = eval(dof);
			}
		}
	};
}