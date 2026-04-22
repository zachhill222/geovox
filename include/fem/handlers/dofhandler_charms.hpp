#pragma once

#include "mesh/voxel_mesh.hpp"
#include "fem/handlers/dofhandler_base.hpp"

#include <type_traits>
#include <cstdint>
#include <vector>
#include <algorithm>

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

		//gather active basis sets
		template<typename Elem_type> requires (VoxelEquivFeature<Elem_t,Elem_type>)
		std::vector<DOF_t> basis_s(const Elem_type el) const {
			assert(el.is_valid());
			std::vector<DOF_t> bs;
			
			bs.reserve(DOF_t::N_DOF_PER_ELEM);
			for (DOF_t dof : DOF_t::dofs_on_elem(el)) {
				if (dof.exists() && this->is_active(dof)) {
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
			std::vector<DOF_t> b   = basis_a(el);
			std::vector<DOF_t> b_s = basis_s(el);
			b.insert(b.end(),
				std::make_move_iterator(b_s.begin()),
				std::make_move_iterator(b_s.end()));
			return b;
		}

		//atomic operations
		void activate(const DOF_t dof) {
			assert(dof.key.is_valid());

			//request the mesh to activate the support elements
			for (auto el : dof.support()) {
				if(el.exists()) {this->mesh.activate(static_cast<Elem_t>(el));}
			}
			this->set_active(dof,true);
		}

		void deactivate(const DOF_t dof) {
			this->set_active(dof,false);
			for (auto el : dof.support()) {
				if (el.exists() and basis_s(el).empty()) {this->mesh.deactivate(static_cast<Elem_t>(el));}
			}
		}


		//check if a basis function can be refined or coarsened
		bool can_refine(const DOF_t dof) const {
			assert(dof.is_valid());
			if (dof.depth() >= BASE::MAX_DEPTH) {return false;}

			for (const DOF_t p : dof.parents()) {
				if (p.exists() && this->is_active(p)) {return false;}
			}
			return true;
		}

		bool can_coarsen(const DOF_t dof) const {
			assert(dof.is_valid());
			if (dof.depth()==0) {return false;}
			if (!this->is_active(dof)) {return false;}

			for (const DOF_t c : dof.children()) {
				if (c.exists() && this->is_active(c)) {return false;}
			}
			return true;
		}

		void refine(const std::vector<DOF_t>& dofs) {
			for (const DOF_t dof : dofs) {refine(dof);}
		}

		void refine(const DOF_t dof) {
			assert(this->is_active(dof));
			if (!can_refine(dof)) {return;}
	
			deactivate(dof);
			for (DOF_t c : dof.children()) {
				if (c.exists()) {
					activate(c);
				}
			}		
		}

		void coarsen(const std::vector<DOF_t>& dofs) {
			for (const DOF_t dof : dofs) {coarsen(dof);}
		}

		void coarsen(const DOF_t dof) {
			if (!can_coarsen(dof)) {return;}

			//TODO: in the quasi-hierarchcial case, do we deactivate all siblings?
			for (DOF_t c : dof.children()) {
				if (c.exists()) {
					deactivate(c);
				}
			}
			activate(dof);
		}

		
		template<typename CoefContainer_t>
		void update_coefs(CoefContainer_t& old_coefs, CoefContainer_t& new_coefs) {
			//transfer each coefficient of old into new
			//or split its contribution into its children in new
			//or compress its contribution into its parent in new
			//the coef lists must always be sorted so that the lookup is fast
			const auto& prev_dofs = this->prev_compressed_dofs();
			const auto& curr_dofs = this->curr_compressed_dofs();

			assert(static_cast<size_t>(old_coefs.size()) == prev_dofs.size());
			assert(static_cast<size_t>(new_coefs.size()) == curr_dofs.size());
			assert(static_cast<size_t>(new_coefs.size()) == this->n_dofs());

			//lambda to directly transfer a coefficient
			auto transfer = [&](const double val, const DOF_t dof) {
				assert(dof.is_valid());
				auto it = std::lower_bound(curr_dofs.begin(), curr_dofs.end(), dof);
				assert(it != curr_dofs.end());
				uint64_t idx = std::distance(curr_dofs.begin(), it);
				new_coefs[idx] = val;
				return true;
			};

			//lambda to increment a child or parent dof
			auto increment = [&](const double val, const DOF_t dof) {
				assert(dof.is_valid());
				auto it = std::lower_bound(curr_dofs.begin(), curr_dofs.end(), dof);
				assert(it != curr_dofs.end());
				uint64_t idx = std::distance(curr_dofs.begin(), it);
				new_coefs[idx] += val;
			};

			for (uint64_t i=0; i<static_cast<uint64_t>(old_coefs.size()); ++i) {
				const DOF_t old_dof = prev_dofs[i];
				const double val    = old_coefs[i];

				//transfer, split, or compress
				if (this->is_active(old_dof)) {transfer(val, old_dof);}
				else {
					bool has_active_child = false;
					const auto child_coefs = old_dof.children_coef();
					const auto child_dofs  = old_dof.children();
					for (uint64_t c=0; c<DOF_t::N_CHILDREN; ++c) {
						DOF_t child = child_dofs[c];
						if (child.exists() and this->is_active(child)) {
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
							if (this->is_active(parent)) {
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
			const auto& curr_dofs = this->curr_compressed_dofs();
			std::vector<double> result(n_vertices, 0.0);
			assert(static_cast<size_t>(coefs.size()) == curr_dofs.size());
			
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
				while (dv.depth()<BASE::MAX_DEPTH) {dv = dv.child();}
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
					auto it = std::lower_bound(curr_dofs.begin(), curr_dofs.end(), tr.dof);
					assert (it != curr_dofs.end());
					uint64_t idx = std::distance(curr_dofs.begin(), it);
					result[i] += coefs[idx] * tr.dof.eval(tr.el, tr.pt);
				}
			}

			return result;
		}
	};
}