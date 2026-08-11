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
		gutil::BinSort<uint64_t>		sorter{};	//capture the sorter for the dofs snapshot
		std::array<std::vector<T>,N> 	coefs{};
		bool            				restrict_is_average{false};	//when true, use an average during unrefinement

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
			{
				auto lock = dofhandler.begin_active_keys_stable();

				dh_curr_dofs = dofhandler.active_dofs;
				snapshot_dofs();

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
			{
				auto lock = dofhandler.begin_key_mask_unstable();
				dofhandler.clear_coef_marks();
				dofhandler.set_has_coef(dofs);
				dofhandler.end_key_mask_unstable();
			}
		}


		///////////////////////////////////////////////////////////////////
		/// Look up previous global dof numbers and get snapshot of the handler's current dofs
		///////////////////////////////////////////////////////////////////
		[[nodiscard]] size_t global_number(DOF_t dof) const noexcept {
			const int bin_number = dofhandler.dof_key_bin(dof.key);
			std::span<const uint64_t> list = sorter.get_bin(bin_number);
			auto it = std::lower_bound(list.begin(), list.end(), dof.key);
			if (it==list.end() || *it!=dof.key) {return size_t(-1);}
			return sorter.bin_start(bin_number) + std::distance(list.begin(), it);

			// auto it = std::find(dofs.begin(), dofs.end(), dof);
			// return (it==dofs.end()) ? size_t(-1) : std::distance(dofs.begin(), it);
		}

		void snapshot_dofs() noexcept {
			dofs.clear();
			dofs.insert(dofs.end(), dh_curr_dofs.begin(), dh_curr_dofs.end());
			sorter = dofhandler.get_sorter();
			//the sorter uses the raw key values
			sorter.rebind_to_copy(dofhandler.template reinterpret_key_span<uint64_t,DOF_t>(dofs));
		}


		///////////////////////////////////////////////////////////////////
		/// Transform dofs due to refinement/unrefinement
		///////////////////////////////////////////////////////////////////
		void prolong_coefs() noexcept {
			GUTIL_ASSERT(dofhandler.is_current());
			GUTIL_PROFILE("updating coefficients (", dofs.size(), " -> ", dh_curr_dofs.size(), ")");
			{
				auto lock = dofhandler.begin_active_keys_stable();
				dh_curr_dofs = dofhandler.active_dofs;
				dofhandler.end_active_keys_stable();
			}
			{
				//compute new coefficients
				auto lock1 = dofhandler.begin_key_mask_stable();
				auto lock2 = dofhandler.begin_active_keys_stable();
				std::array<std::vector<Scalar_t>,N> new_coefs;
				auto job = [&](uint8_t i) { prolong_coefs(i,new_coefs[i]); };
				for (uint8_t i=0; i<N; ++i) {
					threads.submit(job, i);
				}
				threads.wait_idle();
				coefs = std::move(new_coefs);
				dofhandler.end_active_keys_stable();
				dofhandler.end_key_mask_stable();
			}
			{
				//finalize
				snapshot_dofs();

				auto lock1 = dofhandler.begin_key_mask_unstable();
				dofhandler.clear_coef_marks();
				dofhandler.set_has_coef(dofs);
				dofhandler.end_key_mask_unstable();
			}
		}

		void restrict_coefs() noexcept {
			GUTIL_ASSERT(dofhandler.is_current());
			GUTIL_PROFILE("updating coefficients (", dofs.size(), " -> ", dh_curr_dofs.size(), ")");
			{
				auto lock = dofhandler.begin_active_keys_stable();
				dh_curr_dofs = dofhandler.active_dofs;
				dofhandler.end_active_keys_stable();
			}
			{
				//compute new coefficients
				auto lock1 = dofhandler.begin_key_mask_stable();
				auto lock2 = dofhandler.begin_active_keys_stable();
				std::array<std::vector<Scalar_t>,N> new_coefs;
				
				for (uint8_t i=0; i<N; ++i) {
					threads.submit([&,i](){
						// if (restrict_is_average) {restrict_coefs_average(i,new_coefs[i]);}
						// else {restrict_coefs(i,new_coefs[i]);}
						restrict_coefs(i,new_coefs[i]);
					});
				}
				threads.wait_idle();

				coefs = std::move(new_coefs);
				dofhandler.end_active_keys_stable();
				dofhandler.end_key_mask_stable();
			}
			{
				//finalize
				snapshot_dofs();

				auto lock1 = dofhandler.begin_key_mask_unstable();
				dofhandler.clear_coef_marks();
				dofhandler.set_has_coef(dofs);
				dofhandler.end_key_mask_unstable();
			}
		}

		protected:
		////////////////////////////////////////////////////////////////////////////////
		/// Compute coefficients for dofs that were introduced by refinement.
		///
		/// Start at the coarse node that used to be active and propagate it's coefficient
		/// to all children multiplied by their weight, recursing as needed.
		////////////////////////////////////////////////////////////////////////////////
		void prolong_coefs(uint8_t i, std::vector<Scalar_t>& new_coefs) noexcept {
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
					GUTIL_ASSERT(!dofhandler.is_active_stable(dof));
					GUTIL_ASSERT(dofhandler.is_refined_stable(dof));
					distribute_refined(dof, coefs[i][idx], std::span<Scalar_t>(new_coefs), new_size);

				}//for dofs
			}//omp parallel
		}//update coefs

		void restrict_coefs(uint8_t i, std::vector<Scalar_t>& new_coefs) noexcept {
			GUTIL_ASSERT(dofhandler.is_active_keys_stable());
			GUTIL_ASSERT(dofhandler.is_key_mask_stable());
			GUTIL_ASSERT(i<N);

			const size_t old_size = dofs.size();
			const size_t new_size = dh_curr_dofs.size();
			new_coefs.assign(new_size, Scalar_t{0});

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
					GUTIL_ASSERT(!dofhandler.is_active_stable(dof));
					GUTIL_ASSERT(!dofhandler.is_refined_stable(dof));
					distribute_unrefined_restrict(dof, coefs[i][idx], std::span<Scalar_t>(new_coefs), new_size);
				}//for dofs
			}//omp parallel
		}//update coefs

		// void restrict_coefs_average(uint8_t i, std::vector<Scalar_t>& new_coefs) noexcept {
		// 	GUTIL_ASSERT(dofhandler.is_active_keys_stable());
		// 	GUTIL_ASSERT(dofhandler.is_key_mask_stable());
		// 	GUTIL_ASSERT(i<N);

		// 	const size_t new_size = dh_curr_dofs.size();
		// 	new_coefs.assign(new_size, Scalar_t{0});
			
		// 	std::vector<Scalar_t> w_increment(new_size, Scalar_t{0});
		// 	std::vector<Scalar_t> w_sum(new_size, Scalar_t{0});

		// 	std::span<Scalar_t> new_coefs_span(new_coefs);
		// 	std::span<Scalar_t> w_increment_span(w_increment);
		// 	std::span<Scalar_t> w_sum_span(w_sum);

		// 	GUTIL_OMP(parallel for)
		// 	for (size_t idx=0; idx<dofs.size(); ++idx) {
		// 		DOF_t dof = dofs[idx];
		// 		size_t n_idx = dofhandler.global_number(dof);
		// 		if (n_idx < new_size) {
		// 			GUTIL_OMP(atomic) new_coefs[n_idx]	+= coefs[i][idx];
		// 			GUTIL_OMP(atomic) w_sum[n_idx]		+= Scalar_t{1};
		// 		}
		// 		distribute_unrefined_average(dof, coefs[i][idx], new_coefs_span, w_sum_span, new_size);
		// 	}

		// 	//normalize
		// 	GUTIL_SIMD()
		// 	for (size_t idx=0; idx<new_size; ++idx) {
		// 		new_coefs[idx] += w_increment[idx]/w_sum[idx];
		// 		if (w_increment[idx]>0) {
		// 		}
		// 	}


		// 	// GUTIL_OMP(parallel for)
		// 	// for (size_t idx=0; idx<new_size; ++idx) {
		// 	// 	DOF_t dof = dh_curr_dofs[idx];
		// 	// 	GUTIL_ASSERT(dofhandler.has_coef_stable(dof) || dofhandler.is_coef_prop_stable(dof));

		// 	// 	// was this dof ALREADY active with a known value before this batch? just keep it.
		// 	// 	if (dofhandler.has_coef_stable(dof)) {
		// 	// 		size_t old_idx = global_number(dof);
		// 	// 		GUTIL_ASSERT(old_idx<dofs.size());
		// 	// 		new_coefs[idx] = coefs[i][old_idx];
		// 	// 		// continue;
		// 	// 	}

				
		// 	// 	if (dofhandler.is_coef_prop_stable(dof)) {
		// 	// 		// newly-reactivated -- gather a weighted average from its former children
		// 	// 		Scalar_t w_coef = Scalar_t{0};
		// 	// 		Scalar_t w_sum  = Scalar_t{0};   // was incorrectly initialized to 1
		// 	// 		gather_unrefined_average(dof, coefs[i], Scalar_t{1}, w_coef, w_sum);
		// 	// 		GUTIL_ASSERT(w_sum > Scalar_t{0});
		// 	// 		new_coefs[idx] += w_coef / w_sum;   // was missing entirely -- result was never written
		// 	// 	}
		// 	// }
		// }




		//////////////////////////////////////////////////////////////////////////
		/// Helper functions to ensure multiple refinements can be processed correctly
		//////////////////////////////////////////////////////////////////////////
		//recursively distribute a refined old-dof's contribution down through the hierarchy
		//until reaching descendants that are genuinely active (not themselves refined further
		//within the same batch)
		void distribute_refined(DOF_t dof, Scalar_t contribution, std::span<Scalar_t> new_coefs, size_t new_size) const noexcept {
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
		//hierarchy until reaching an active ancestor. using restriction means that only
		//features that 'live' at the parent feature contribute to the parent coefficient
		//
		void distribute_unrefined_restrict(DOF_t dof, Scalar_t contribution, std::span<Scalar_t> new_coefs, size_t new_size) const noexcept {
			DOF_t p_dofs[DOF_t::N_PARENTS];
			dof.parents_simd(p_dofs);
			for (uint8_t p=0; p<DOF_t::N_PARENTS; ++p) {
				if (!p_dofs[p].exists()) {continue;}
				Scalar_t parent_contribution = contribution * dof.template parent_coef_restrict<Scalar_t>(p);
				if (parent_contribution == Scalar_t{0}) {continue;}

				size_t n_idx = dofhandler.global_number(p_dofs[p]);
				if (n_idx<new_size) {
					GUTIL_OMP(atomic)
					new_coefs[n_idx] += parent_contribution;
				}
				else if (!dofhandler.is_active_stable(p_dofs[p])) {
					distribute_unrefined_restrict(p_dofs[p], parent_contribution, new_coefs, new_size);
				}
			}
		}


		void distribute_unrefined_average(DOF_t dof, Scalar_t contribution, std::span<Scalar_t> new_coefs,
							std::span<Scalar_t> w_sum, size_t new_size) const noexcept {
			DOF_t p_dofs[DOF_t::N_PARENTS];
			dof.parents_simd(p_dofs);
			for (uint8_t p=0; p<DOF_t::N_PARENTS; ++p) {
				if (!p_dofs[p].exists()) {continue;}
				Scalar_t weight   = dof.template parent_coef_weight<Scalar_t>(p);
				Scalar_t weighted = contribution * weight;
				size_t n_idx = dofhandler.global_number(p_dofs[p]);
				// size_t o_idx = global_number(p_dofs[p]);
				if (n_idx<new_size){// && o_idx==size_t(-1)) {
					GUTIL_OMP(atomic) new_coefs[n_idx]	+= weighted;
					GUTIL_OMP(atomic) w_sum[n_idx]	 	+= weight;
				} else if (!dofhandler.is_active_stable(p_dofs[p])) {
					distribute_unrefined_average(p_dofs[p], weighted, new_coefs, w_sum, new_size);
				}
			}
		}



		//start at a newly activated dof from unrefinement. Its coefficient needs to be initialized.
		//we do this in a manner that averages the contribution from all descendants that used to be active.
		//we start at the parent and recurse into the children, accumulating a weighted average of their coefficients.
		//
		// - dof: 		the dof being examined (start at active parent and recurse into children)
		// - w:   		the current weight that the dof contributes
		// - w_coef:	the current sum of weights*coefficients
		// - w_sum:		the current sum of weights
		// void gather_unrefined_average(DOF_t dof, std::span<const Scalar_t> old_coefs, Scalar_t w, Scalar_t& w_coef, Scalar_t& w_sum) const noexcept {
		// 	DOF_t c_dofs[DOF_t::N_CHILDREN];
		// 	dof.children_simd(c_dofs);
		// 	for (uint8_t c=0; c<DOF_t::N_CHILDREN; ++c) {
		// 		if (!c_dofs[c].exists()) {continue;}
		// 		Scalar_t w_new = w * dof.template child_coef<Scalar_t>(c);
		// 		if (dofhandler.has_coef_stable(c_dofs[c])) {
		// 			size_t old_idx = global_number(c_dofs[c]);
		// 			GUTIL_ASSERT(old_idx < old_coefs.size());
		// 			w_sum  += w_new;
		// 			w_coef += w_new*old_coefs[old_idx];
		// 		} else if (dofhandler.is_coef_prop_stable(c_dofs[c])) {
		// 			gather_unrefined_average(c_dofs[c], old_coefs, w_new, w_coef, w_sum);   // was missing old_coefs
		// 		}
		// 	}
		// }



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


