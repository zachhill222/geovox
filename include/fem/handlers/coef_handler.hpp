#pragma once

#include "gutil.hpp"

#include "simd_keys/mesh/mesh_key_implementation.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"
#include "simd_keys/dofs/utility.hpp"

#include <cstdint>
#include <vector>
#include <array>
#include <span>
#include <algorithm>

namespace GV {

	//////////////////////////////////////////////////////////////////////
	/// A helper class to track the current dofs and update coefficients
	/// after refinement.
	//////////////////////////////////////////////////////////////////////
	template<typename DofHandler_type, typename T=double, uint8_t N=1>
	struct  CoefHandler {
		

		//////////////////////////////////////////////////////////////////
		/// Aliases and constants
		//////////////////////////////////////////////////////////////////
		//DOF features may be periodic
		using DofHandler_t = DofHandler_type;
		using DOF_t        = typename DofHandler_t::DOF_t;
		using DofVert_t    = typename DofHandler_t::DofVert_t;
		using DofElem_t    = typename DofHandler_t::DofElem_t;
		using DofFeature_t = typename DofHandler_t::DofFeature_t;

		//Mesh features are never periodic
		using Mesh_t        = typename DofHandler_t::Mesh_t;
		using MeshVert_t    = typename DofHandler_t::MeshVert_t;
		using MeshElem_t    = typename DofHandler_t::MeshElem_t;
		using MeshFeature_t = typename DofHandler_t::MeshFeature_t;

		using Scalar_t      = T;

		//////////////////////////////////////////////////////////////////
		// Hold a snapshot of the last dofs (current coefficients)
		// and a reference to the latest dofs (what we need to transform to)
		//////////////////////////////////////////////////////////////////
		mutable gutil::ThreadPool 		threads{N};
		const Mesh_t& 					mesh;
		const DofHandler_t& 			dofhandler;
		std::span<const DOF_t> 			dh_curr_dofs;
		std::vector<DOF_t> 				dofs{};
		std::array<std::vector<T>,N> 	coefs{};
		static constexpr uint8_t 		COEF_MARKED_BIT = DofHandler_type::COEF_MARKED_BIT;

		//////////////////////////////////////////////////////////////////
		/// Constructor and movement
		//////////////////////////////////////////////////////////////////
		CoefHandler() = delete;
		CoefHandler(const DofHandler_t& dofhandler) noexcept : 
			mesh{dofhandler.mesh},
			dofhandler{dofhandler},
			dh_curr_dofs{dofhandler.active_dofs} {}
		CoefHandler(CoefHandler&& other) noexcept = default;
		CoefHandler(const CoefHandler&) = default;
		CoefHandler& operator=(CoefHandler&& other) noexcept = default;
		~CoefHandler() {threads.wait_idle();}

		///////////////////////////////////////////////////////////////////
		/// Initialize a coefficient field by a scalar function.
		/// The mesh/dofs must be in a conformal state. The 'scalar function'
		/// is actually a linear operator on the shape functions (dofs).
		/// In lagrange dofs, this is the point evaluation of a function
		/// at the corresponding vertex
		///////////////////////////////////////////////////////////////////
		template<typename DofEval>
		void init_coefs(uint8_t i, DofEval&& eval) noexcept {
			GUTIL_ASSERT(dofhandler.is_current());
			auto lock = dofhandler.begin_active_keys_stable();

			dh_curr_dofs = dofhandler.active_dofs;
			dofs.clear();
			dofs.insert(dofs.end(), dh_curr_dofs.begin(),dh_curr_dofs.end());
			GUTIL_ASSERT(i<N);

			GUTIL_ASSERT(dofs.size()>0);
			coefs[i].resize(dofs.size());
			GUTIL_OMP(parallel for)
			for (uint64_t idx=0; idx<dofs.size(); ++idx) {
				#ifndef NDEBUG
					bool flag = mesh.is_conformal(static_cast<MeshFeature_t>(dofhandler.feature(dofs[idx])));
					GUTIL_ASSERT(flag);
				#endif
				coefs[i][idx] = eval(dofs[idx]);
			}

			dofhandler.end_active_keys_stable();
		}


		///////////////////////////////////////////////////////////////////
		/// Transform dofs due to refinement/unrefinement
		///
		/// Note that we can only do one layer of refinement at a time
		///////////////////////////////////////////////////////////////////
		void update_coefs() noexcept {
			GUTIL_ASSERT(dofhandler.is_current());
			GUTIL_TIMER("updating coefficients (", dofs.size(), " -> ", dh_curr_dofs.size(), ")");
			{
				auto lock = dofhandler.begin_active_keys_stable();
				dh_curr_dofs = dofhandler.active_dofs;
			}
			
			{
				//mark all dof masks with which ones are in the current batch
				auto lock = dofhandler.begin_key_mask_unstable();
				dofhandler.unconditional_bitwise_and_all_masks(~COEF_MARKED_BIT);
				GUTIL_OMP(parallel for schedule(static, 1024))
				for (uint64_t idx=0; idx<dofs.size(); ++idx) {
					dofhandler.set_coef_marked(dofs[idx], false);
				}
				dofhandler.end_key_mask_unstable();
			}

			{
				//compute new coefficients
				auto lock1 = dofhandler.begin_key_mask_stable();
				auto lock2 = dofhandler.begin_active_keys_stable();
				std::array<std::vector<Scalar_t>,N> new_coefs;
				auto job = [&](uint8_t i) { update_coefs(i,new_coefs[i]); };
				for (uint8_t i=0; i<N; ++i) {
					threads.submit(job, i);
				}
				threads.wait_idle();
				coefs = std::move(new_coefs);
				dofhandler.end_active_keys_stable();
				dofhandler.end_key_mask_stable();
			}

			//finalize
			dofs.clear();
			dofs.insert(dofs.end(), dh_curr_dofs.begin(), dh_curr_dofs.end());

			dofhandler.end_active_keys_stable();
		}

		void update_coefs(uint8_t i, std::vector<Scalar_t>& new_coefs) noexcept {
			GUTIL_ASSERT(dofhandler.is_active_keys_stable());
			GUTIL_ASSERT(dofhandler.is_key_mask_stable());
			GUTIL_ASSERT(i<N);

			const size_t old_size = dofs.size();
			const size_t new_size = dh_curr_dofs.size();
			new_coefs.resize(new_size, Scalar_t{0});

			GUTIL_OMP(parallel)
			{
				GUTIL_OMP(for)
				for (size_t idx=0; idx<old_size; ++idx) {
					DOF_t dof = dofs[idx];

					size_t n_idx = dofhandler.global_number(dof);
					if (n_idx < new_size) {
						GUTIL_OMP(atomic)
						new_coefs[n_idx] += coefs[i][idx];
						continue;
					}
					GUTIL_ASSERT(!dofhandler.is_active_stable(dof))

					if (dofhandler.is_refined_stable(dof)) {
					    distribute_refined(dof, coefs[i][idx], new_coefs, new_size);
					}
					else {
						distribute_unrefined(dof, coefs[i][idx], new_coefs, new_size);
					}
				}//for dofs
			}//omp parallel
		}//update coefs




		//////////////////////////////////////////////////////////////////////////
		/// Helper functions to ensure multiple refinements can be processed correctly
		//////////////////////////////////////////////////////////////////////////
	private:
		//recursively distribute a refined old-dof's contribution down through the hierarchy
		//until reaching descendants that are genuinely active (not themselves refined further
		//within the same batch)
		void distribute_refined(DOF_t dof, Scalar_t contribution, std::vector<Scalar_t>& new_coefs, size_t new_size) const noexcept {
			DOF_t c_dofs[DOF_t::N_CHILDREN];
			dof.children_simd(c_dofs);
			for (uint8_t c=0; c<DOF_t::N_CHILDREN; ++c) {
				if (!c_dofs[c].exists()) {continue;}
				Scalar_t child_contribution = contribution * dof.template child_coef<Scalar_t>(c);
				
				if (dofhandler.is_refined_stable(c_dofs[c])) {
					distribute_refined(c_dofs[c], child_contribution, new_coefs, new_size);
				}
				else {
					size_t n_idx = dofhandler.global_number(c_dofs[c]);
					if (n_idx<new_size) {
						GUTIL_ASSERT(dofhandler.is_active_stable(c_dofs[c]));
						GUTIL_OMP(atomic)
						new_coefs[n_idx] += child_contribution;
					}
				}
			}
		}

		//recursively distribute an unrefined-away old-dof's contribution up through the
		//hierarchy until reaching an ancestor that's genuinely active
		void distribute_unrefined(DOF_t dof, Scalar_t contribution, std::vector<Scalar_t>& new_coefs, size_t new_size) const noexcept {
			DOF_t pc_dofs[DOF_t::N_PARENTS];
			dof.parents_simd(pc_dofs);
			for (uint8_t p=0; p<DOF_t::N_PARENTS; ++p) {
				if (!pc_dofs[p].exists()) {continue;}
				Scalar_t parent_contribution = contribution * dof.template parent_coef<Scalar_t>(p);
				if (parent_contribution == Scalar_t{0}) {continue;}

				size_t n_idx = dofhandler.global_number(pc_dofs[p]);
				if (n_idx<new_size) {
					if (!dofhandler.is_coef_marked_stable(pc_dofs[p])) {
						GUTIL_OMP(atomic)
						new_coefs[n_idx] += parent_contribution;
					}
				}
				else if (!dofhandler.is_active_stable(pc_dofs[p])) {
					distribute_unrefined(pc_dofs[p], parent_contribution, new_coefs, new_size);
				}
			}
		}

	public:
		////////////////////////////////////////////////////////////////////////
		/// Evaluate at mesh vertices for visualizations
		/// Pass begin/end iterators to the existing vertices
		////////////////////////////////////////////////////////////////////////
		template<std::random_access_iterator I>
		std::vector<Scalar_t> evaluate(uint8_t i, I v_begin, I v_end) const noexcept {
			GUTIL_ASSERT(i<N);
			GUTIL_ASSERT(coefs[i].size()==dofs.size());
			GUTIL_ASSERT(dofs.size()>0 && "there are no dofs or coefficients. did you forget to initialize them?");
			GUTIL_ASSERT(dofs.size()==dh_curr_dofs.size() && "the coef_handler and dof_handler are out of sync");
			GUTIL_ASSERT(dofs[0]==dh_curr_dofs[0] && "the coef_handler and dof_handler are out of sync");
			GUTIL_ASSERT(dofs[dofs.size()-1]==dh_curr_dofs[dofs.size()-1] && "the coef_handler and dof_handler are out of sync");

			static_assert(std::same_as<std::iter_value_t<I>, MeshVert_t>);
			GUTIL_ASSERT(v_begin!=v_end && "there are no vertices. did you forget to collect them?");

			const size_t n_verts = std::distance(v_begin,v_end);
			GUTIL_ASSERT(v_end == v_begin+n_verts);
			
			//initialize storage
			std::vector<Scalar_t> vals(n_verts, Scalar_t{0});

			GUTIL_OMP(parallel)
			{
				
				const uint64_t n_threads 	= GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
				const uint64_t tid       	= GUTIL_OMP_TERNARY(omp_get_thread_num(),  0);
				const uint64_t n_per_thread = n_verts/n_threads;
				const uint64_t start        = tid*n_per_thread;
				const uint64_t end          = (tid==n_threads-1) ? n_verts : start+n_per_thread;
				

				size_t idx = start; I it_end = v_begin+end;
				auto action = [&](DOF_t dof, uint8_t local_n, Scalar_t x, Scalar_t y, Scalar_t z, uint64_t global_n) {
					Scalar_t val{0};
					dof.evaluate_simd(local_n, &val, &x, &y, &z, 1);
					vals[idx] += coefs[i][global_n] * val;
				};

				GUTIL_ASSERT(tid!=n_threads-1 || it_end==v_end);
				for (I it = v_begin+start; it!=it_end; ++it, ++idx) {
					GUTIL_ASSERT(it!=v_end);
					//check neighbor elements with the same periodicity as the dofs
					//additionally, descend to the bottom of the mesh so the hierarchical
					//dof gather gets all dofs
					DofVert_t dv = static_cast<DofVert_t>(*it);
					
					dofhandler.gather_hierarchical_dofs_at_vertex(action, dv);
				}
			}

			return vals;
		}


		//////////////////////////////////////////////////////////////////////
		/// Debugging
		//////////////////////////////////////////////////////////////////////
		void print_coefs(int i=0) {
			std::cout << "\nCoefs for field " << i << ":\n";
			for (size_t idx=0; idx<dofs.size(); ++idx) {
				std::cout << idx << " : " << dofs[idx] << " -> " << coefs[i][idx] << "\n";
			}
		}

	};





}


