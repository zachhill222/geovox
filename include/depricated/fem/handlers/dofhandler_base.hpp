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

namespace GV {


	////////////////////////////////////////////////////////////////////////////////////////
	/// A base class for storing dofs at mesh features.
	////////////////////////////////////////////////////////////////////////////////////////
	template<VoxelMeshType Mesh_type, typename DOF_type>
	struct DofHandlerBase {


		////////////////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		////////////////////////////////////////////////////////////////////////////////////
		using DOF_t      = DOF_type;
		using QuadElem_t = typename DOF_t::QuadElem_t;
		using DOFKey_t   = typename DOF_t::Key_t;
		using MeshKey_t  = typename DOF_t::Key_t::NonPeriodicVariant;
		using Mesh_t     = Mesh_type;
		using Elem_t     = typename Mesh_t::VoxelElement;
		using Vert_t     = typename Mesh_t::VoxelVertex;
		using Face_t     = typename Mesh_t::VoxelFace;

		static constexpr bool IS_VERTEX = std::same_as<MeshKey_t,Vert_t>;
		static constexpr bool IS_FACE   = std::same_as<MeshKey_t,Face_t>;
		static constexpr bool IS_ELEM   = std::same_as<MeshKey_t,Elem_t>;

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
		
		#ifdef _OPENMP
		static constexpr bool OPENMP = true;
		#else
		static constexpr bool OPENMP = false;
		#endif
		
		////////////////////////////////////////////////////////////////////////////////////
		/// Constructors and memory management
		////////////////////////////////////////////////////////////////////////////////////
		explicit DofHandlerBase(const Mesh_t& mesh) : mesh(mesh) {}
		~DofHandlerBase() { delete active_dofs; }

		//not copyable or movable
		DofHandlerBase(const DofHandlerBase&) = delete;
		DofHandlerBase& operator=(const DofHandlerBase&) = delete;
		DofHandlerBase(DofHandlerBase&& other) = delete;
		DofHandlerBase& operator=(DofHandlerBase&&) = delete;

	protected:
		////////////////////////////////////////////////////////////////////////////////////
		/// Storage
		////////////////////////////////////////////////////////////////////////////////////
		//track which dofs are active based on a global numbering system of the voxel features.
		//instances of the DOF_t act as an iterator into this set.
		std::bitset<TOTAL_POSSIBLE_DOFS>* active_dofs = new std::bitset<TOTAL_POSSIBLE_DOFS>{};

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
		////////////////////////////////////////////////////////////////////////////////////
		/// Interface
		////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr uint64_t n_dofs() const noexcept {return active_dof_list_curr.size();}
		[[nodiscard]] constexpr uint64_t count_dofs() const noexcept {return active_dofs->count();}

		[[nodiscard]] constexpr bool is_active(const DOF_t dof) const noexcept {
			assert(dof.is_valid());
			return active_dofs->test(dof.key.linear_index());
		}

		[[nodiscard]] constexpr bool is_active(const uint64_t idx) const noexcept {
			assert(idx<TOTAL_POSSIBLE_DOFS);
			return active_dofs->test(idx);
		}
		
		constexpr void set_active(const DOF_t dof, const bool b) noexcept {
			assert(dof.is_valid());
			active_dofs->set(dof.key.linear_index(), b);
		}

		constexpr void set_active(const uint64_t idx, const bool b) noexcept {
			assert(idx<TOTAL_POSSIBLE_DOFS);
			active_dofs->set(idx, b);
		}

		[[nodiscard]] const auto& prev_compressed_dofs() const noexcept {return active_dof_list_prev;}
		[[nodiscard]] const auto& curr_compressed_dofs() const noexcept {return active_dof_list_curr;}
		
		[[nodiscard]] const auto& dof_to_idx() const noexcept {return dof_to_idx_map;}
		
		[[nodiscard]] uint64_t compressed_index(DOF_t dof) const noexcept {
			const auto it = dof_to_idx_map.find(dof);
			return (it != dof_to_idx_map.end()) ? it->second : uint64_t(-1);
		}
		
		[[nodiscard]] DOF_t get_dof(uint64_t idx) const noexcept {
			assert(idx<active_dof_list_curr.size());
			return active_dof_list_curr[idx];
		}

		//determine if the specified feature is 'cononical'. Returns true when
		//the feature is the feature 
		// [[nodiscard]] bool is_cononical(MeshKey_t key) const noexcept {
			
		// }

		//get begin/end iterators to the mesh feature
		auto feature_begin() const noexcept {
			if constexpr (IS_ELEM) { return mesh.element_begin(); }
			else if constexpr (IS_VERTEX) { return mesh.vertex_begin(); }
			else { return mesh.face_begin(); }
		}

		auto feature_end() const noexcept {
			if constexpr (IS_ELEM) { return mesh.element_end(); }
			else if constexpr (IS_VERTEX) { return mesh.vertex_end(); }
			else { return mesh.face_end(); }
		}

		//convert a mesh feature to a dof
		[[nodiscard]] static constexpr DOF_t feature_to_dof(MeshKey_t key) noexcept {
			return DOF_t{static_cast<DOFKey_t>(key)};
		}

		//convert a mesh feature to a (possibly periodic) feature for the dof
		[[nodiscard]] static constexpr DOFKey_t mesh_key_to_dof_key(MeshKey_t key) noexcept {
			return static_cast<DOFKey_t>(key);
		}


		//simple management operations
		void set_all_inactive() noexcept {active_dofs->reset();}

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

		//build dofs at every conformal feature of the mesh
		//this requires the mesh to have an iterator for the feature and
		//for there to be an is_conformal() check
		void init_conformal() noexcept {
			set_all_inactive();
			for (auto it=feature_begin(); it!=feature_end(); ++it) {
				if (mesh.is_conformal(*it)) {
					//note that a few periodic dofs may be activated twice,
					//but this redundant work is minimal.
					set_active(feature_to_dof(*it), true);
				}
			}
		}

		//transfer computations between mesh refinements
		void compress_dof_numbers() {
			const uint64_t ndofs = count_dofs();
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