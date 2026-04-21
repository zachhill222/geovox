#pragma once

#include "mesh/voxel_mesh.hpp"

#include <type_traits>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace GV
{
	template<VoxelMeshType Mesh_type, typename DOF_type>
	class DofHandler
	{
	public:
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

		static_assert(
			VoxelEquivFeature<MeshKey_t, typename Mesh_t::VoxelElement> ||
			VoxelEquivFeature<MeshKey_t, typename Mesh_t::VoxelFace>    ||
			VoxelEquivFeature<MeshKey_t, typename Mesh_t::VoxelVertex>,
			"DofHandler - the feature key for the DOF must match the corresponding feature key of the mesh.");
		static_assert(VoxelEquivFeature<Elem_t, typename DOF_type::QuadElem_t::NonPeriodicType>,
			"DofHandler - the DOF quadrature element and the mesh element must be of an equivalent type");

		//we can iterate over the mesh and REQUEST mesh refinement
		//however, mesh refinement must be done outside of this class
		//because multiple DOF handlers can reference the same mesh
		static constexpr uint64_t MAX_DEPTH = Mesh_t::MAX_DEPTH;
		const Mesh_t& mesh;

		static constexpr uint64_t TOTAL_POSSIBLE_DOFS = total_possible<MeshKey_t>(MAX_DEPTH);
		static constexpr bool OPENMP = Mesh_t::OPENMP;

		//constructor and destructor
		DofHandler(const Mesh_t& mesh) : mesh(mesh) {}
		virtual ~DofHandler() {
			delete active_dofs;
			delete stale_dofs;
		}

		//non-copyable
		DofHandler(const DofHandler&) = delete;
		DofHandler& operator=(const DofHandler&) = delete;

		//movable by construction
		DofHandler(DofHandler&& other) :
			mesh(other.mesh), 
			active_dofs(other.active_dofs), 
			stale_dofs(other.stale_dofs)
		{
			other.active_dofs = nullptr;
			other.stale_dofs = nullptr;
		}

		//can't move by assignment
		DofHandler& operator=(DofHandler&&) = delete;

		//maintain lists of dofs that have been newly activated or deactivated
		//these are public so that the problem class can incrementally update any matrices
		//any time a dof is activated or deactivated, it's corresponding stale_dof bit is set to true.
		//another class may do something with this information and should then set the bit to false.
		std::bitset<TOTAL_POSSIBLE_DOFS>* stale_dofs = new std::bitset<TOTAL_POSSIBLE_DOFS>(0);

		protected:
		//track which dofs are active based on a global numbering system of the voxel features.
		//instances of the DOF_t act as an iterator into this set.
		std::bitset<TOTAL_POSSIBLE_DOFS>* active_dofs = new std::bitset<TOTAL_POSSIBLE_DOFS>(0);

		//track a compressed list of active dofs
		//the bitfield is the "source of truth"
		//but this is used to convert the global dof number
		//into the dof number that the matrix solver will use
		//to help incrementally update scalar fields and integration matrices,
		std::vector<DOF_t> active_dof_list_prev;
		std::vector<DOF_t> active_dof_list_curr;

		public:
		//simple querries
		inline constexpr uint64_t n_dofs() const {return active_dofs->count();}

		inline constexpr bool is_active(const DOF_t dof) const {assert(dof.is_valid()); return active_dofs->test(dof.key.linear_index());}
		inline constexpr void set_active(const DOF_t dof, const bool b) {assert(dof.is_valid()); active_dofs->set(dof.key.linear_index(), b);}

		inline constexpr bool is_stale(const DOF_t dof) const {assert(dof.is_valid()); return stale_dofs->test(dof.key.linear_index());}
		inline constexpr void set_stale(const DOF_t dof, const bool b) {assert(dof.is_valid()); stale_dofs->set(dof.key.linear_index(), b);}

		inline const std::vector<DOF_t>& last_compressed_dofs() const {return active_dof_list_prev;}
		inline const std::vector<DOF_t>& compressed_dofs() const {return active_dof_list_curr;}

		//simple management operations
		inline void reset_active() {active_dofs->reset();}
		inline void reset_stale() {stale_dofs->reset();}
		void save_dof_list() {
			active_dof_list_prev = active_dof_list_curr;
			// active_dof_list_curr.clear();
		}

		//activate all dofs at a certain depth if they have an active support element
		//all other dofs are inactive. the stale bits are reset as it is assumed
		//that this method will only be called when the user intends to "set/reset" the problem.
		void set_depth(const uint64_t dd) {
			active_dofs->reset();
			stale_dofs->reset();

			auto action = [this](MeshKey_t key) {
				const DOF_t dof{static_cast<DOFKey_t>(key)};
				if (has_active_support(dof)) {
					active_dofs->set(dof.linear_index());
				}
			};

			//TODO: call in parallel if needed
			mesh.template for_each_depth_omp<MeshKey_t>(dd,action);

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

		//gather active basis sets
		template<typename Elem_type> requires (VoxelEquivFeature<Elem_t,Elem_type>)
		std::vector<DOF_t> basis_s(const Elem_type el) const {
			assert(el.is_valid());
			std::vector<DOF_t> bs;
			
			bs.reserve(DOF_t::N_DOF_PER_ELEM);
			for (DOF_t dof : DOF_t::dofs_on_elem(el)) {
				if (dof.exists() && is_active(dof)) {
					bs.push_back(dof);
				}
			}
			return bs;
		}


		template<typename Elem_type> requires (VoxelEquivFeature<Elem_t,Elem_type>)
		std::vector<DOF_t> basis_a(Elem_type el) const {
			assert(el.is_valid());
			std::vector<DOF_t> ba;
			if (el.depth()==0) {return ba;}
			
			ba.reserve(DOF_t::N_DOF_PER_ELEM * (el.depth()-1));
			for (uint64_t dd=el.depth(); dd>0; --dd) {
				auto bs = basis_s(el.parent());
				ba.insert(ba.end(), 
					std::make_move_iterator(bs.begin()),
					std::make_move_iterator(bs.end()));
				el = el.parent();
			}
			return ba;
		}

		template<typename Elem_type> requires (VoxelEquivFeature<Elem_t,Elem_type>)
		std::vector<DOF_t> basis_active(Elem_type el) const {
			std::vector<DOF_t> b_a = basis_a(el);
			std::vector<DOF_t> b_s = basis_s(el);
			b_a.insert(b_a.end(),
				std::make_move_iterator(b_s.begin()),
				std::make_move_iterator(b_s.end()));
			return b_a;
		}

		//atomic operations
		void activate(const DOF_t dof) {
			const uint64_t idx = dof.linear_index();
			assert(dof.key.is_valid());

			//request the mesh to activate the support elements
			for (auto el : dof.support()) {
				if(el.exists()) {mesh.activate(static_cast<Elem_t>(el));}
			}
			active_dofs->set(idx,true);
			stale_dofs->set(idx,true);
		}

		void deactivate(const DOF_t dof) {
			const uint64_t idx = dof.linear_index();
			
			active_dofs->set(idx,false);
			stale_dofs->set(idx,true);
			for (auto el : dof.support()) {
				if (el.exists() and basis_s(el).empty()) {mesh.deactivate(static_cast<Elem_t>(el));}
			}
		}

		template<bool HIERARCHICAL=false>
		void refine(const std::vector<DOF_t>& dofs) {
			for (const DOF_t dof : dofs) {refine<HIERARCHICAL>(dof);}
		}

		template<bool HIERARCHICAL=false>
		void refine(const DOF_t dof) {
			if(dof.key.depth() >= MAX_DEPTH) {return;};
			assert(is_active(dof));

			if constexpr (HIERARCHICAL) {
				for (DOF_t c : dof.children()) {
					if (c.exists() and !c.key.parent().exists()) {
						activate(c);
					}
				}
			}
			else {
				deactivate(dof);
				for (DOF_t c : dof.children()) {
					if (c.exists()) {
						activate(c);
					}
				}
			}
		}

		template<bool HIERARCHICAL=false>
		void coarsen(const std::vector<DOF_t>& dofs) {
			for (const DOF_t dof : dofs) {coarsen<HIERARCHICAL>(dof);}
		}

		template<bool HIERARCHICAL=false>
		void coarsen(const DOF_t dof) {
			if constexpr (HIERARCHICAL) {
				assert(is_active(dof));

				for (DOF_t c : dof.children()) {
					if (c.exists() and !c.key.parent().exists()) {
						deactivate(c);
					}
				}
			}
			else {
				assert(!is_active(dof));
				for (DOF_t c : dof.children()) {
					if (c.exists()) {
						deactivate(c);
					}
				}
				activate(dof);
			}
		}

		template<bool HIERARCHICAL=false, typename Predicate=std::nullptr_t>
		void refine_depth(const uint64_t depth, Predicate&& pred=nullptr) {
			assert(depth < Mesh_t::MAX_DEPTH);

			auto action = [this](MeshKey_t key) {
				const DOF_t dof{key};
				if (is_active(dof))	{refine<HIERARCHICAL>(DOF_t{key});}
			};

			mesh.template for_each_depth_omp<MeshKey_t>(depth, action, pred);
		}

		//transfer computations between mesh refinements
		void compress_dof_numbers() {
			const uint64_t ndofs = n_dofs();
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
			std::sort(active_dof_list_curr.begin(), active_dof_list_curr.end());
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

		template<typename CoefContainer_t>
		void update_coefs(CoefContainer_t& old_coefs, CoefContainer_t& new_coefs) {
			//transfer each coefficient of old into new
			//or split its contribution into its children in new
			//or compress its contribution into its parent in new
			//the coef lists must always be sorted so that the lookup is fast
			assert(static_cast<size_t>(old_coefs.size()) == active_dof_list_prev.size());
			assert(static_cast<size_t>(new_coefs.size()) == active_dof_list_curr.size());
			assert(static_cast<size_t>(new_coefs.size()) == n_dofs());

			//lambda to directly transfer a coefficient
			auto transfer = [&new_coefs, this](const double val, const DOF_t dof) {
				assert(dof.is_valid());
				auto it = std::lower_bound(active_dof_list_curr.begin(), active_dof_list_curr.end(), dof);
				assert(it != active_dof_list_curr.end());
				uint64_t idx = std::distance(active_dof_list_curr.begin(), it);
				new_coefs[idx] = val;
				return true;
			};

			//lambda to increment a child or parent dof
			auto increment = [&new_coefs, this](const double val, const DOF_t dof) {
				assert(dof.is_valid());
				auto it = std::lower_bound(active_dof_list_curr.begin(), active_dof_list_curr.end(), dof);
				assert(it != active_dof_list_curr.end());
				uint64_t idx = std::distance(active_dof_list_curr.begin(), it);
				new_coefs[idx] += val;
			};

			for (uint64_t i=0; i<static_cast<uint64_t>(old_coefs.size()); ++i) {
				const DOF_t old_dof = active_dof_list_prev[i];
				const double val    = old_coefs[i];

				//transfer, split, or compress
				if (is_active(old_dof)) {transfer(val, old_dof);}
				else {
					bool has_active_child = false;
					const auto child_coefs = old_dof.children_coef();
					const auto child_dofs  = old_dof.children();
					for (uint64_t c=0; c<DOF_t::N_CHILDREN; ++c) {
						DOF_t child = child_dofs[c];
						if (child.exists() and is_active(child)) {
							has_active_child = true;
							increment(val*child_coefs[c], child);
						}
					}

					if (!has_active_child) {
						const auto parent_coefs = old_dof.parent_coefs();
						const auto parent_dofs = old_dof.parents();
						for (uint64_t p=0; p<DOF_t::N_PARENTS; ++p) {
							DOF_t parent = parent_dofs[p];
							if (!parent.exists()) {break;} //parents are packed to the front, children are not
							if (is_active(parent)) {
								increment(val*parent_coefs[p], parent);
							}
						}
					}
				}
			}
		}

		//append solution to a vtk file
		template<typename CoefContainer_t>
		std::vector<double> interpolate_to_vertices(const CoefContainer_t& coefs, uint64_t n_vertices) const {
			//increment the position value for every active dof
			std::vector<double> result(n_vertices, 0.0);
			assert(static_cast<size_t>(coefs.size()) == active_dof_list_curr.size());
			
			//struct for tracking how to evaluate basis functions
			//and to ensure that a basis function is evaluated only once at a given point
			struct Triple
			{
				DOF_t dof;
				DOF_t::QuadElem_t el;
				DOF_t::RefPoint_t pt;
				bool operator<(const Triple& other) const {return dof.key<other.dof.key;}
				bool operator==(const Triple& other) const {return dof.key==other.dof.key;}
			};


			Vert_t vtx(0,0);
			for (uint64_t i=0; i<n_vertices; ++i, ++vtx) {
				assert(vtx.linear_index() == i);

				//find the deepest active elements containing this vertex
				Vert_t dv{vtx};
				while (dv.depth()<MAX_DEPTH) {dv = dv.child();}
				std::vector<Triple> track_dof_eval;

				//go through all 8 possible elements
				std::vector<DOF_t> basis;
				const auto elems = dv.elements();
				const auto ref_coords = dv.ref_coords();
				for (uint64_t e=0; e<8; ++e) {
					Elem_t el = elems[e];
					if (!el.is_valid()) {continue;}

					auto q_el = static_cast<DOF_t::QuadElem_t>(el);
					const auto pt = static_cast<DOF_t::RefPoint_t>(ref_coords[e]);

					assert(el.is_valid());
					assert(q_el.is_valid());

					basis = basis_s(el);
					for (DOF_t dof : basis) {
						assert(dof.is_valid());
						track_dof_eval.emplace_back(dof,q_el,pt);
					}
					basis = basis_a(el);
					for (DOF_t dof : basis) {
						assert(dof.is_valid());
						track_dof_eval.emplace_back(dof,q_el,pt);
					}
				}

				//sort the evaluation and make it unique to ensure that each dof is evaluated once
				//this is only a problem when evaluating at vertices as they belong to multiple 
				//support elements
				std::sort(track_dof_eval.begin(), track_dof_eval.end());
				auto last = std::unique(track_dof_eval.begin(), track_dof_eval.end());
				track_dof_eval.erase(last, track_dof_eval.end());

				//evaluate the basis functions
				for (Triple& tr : track_dof_eval) {
					tr.dof.proj_to_support(tr.el, tr.pt);
					auto it = std::lower_bound(active_dof_list_curr.begin(), active_dof_list_curr.end(), tr.dof);
					assert (it != active_dof_list_curr.end());
					uint64_t idx = std::distance(active_dof_list_curr.begin(), it);
					result[i] += coefs[idx] * tr.dof.eval(tr.el, tr.pt);
				}
			}

			return result;
		}
	};
}