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
	template<typename DofHandlerType, typename T=double, uint8_t N=1>
	struct  CoefHandler {
		

		//////////////////////////////////////////////////////////////////
		/// Aliases and constants
		//////////////////////////////////////////////////////////////////
		//DOF features may be periodic
		using DofHandler_t = DofHandlerType;
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
		using Point_t       = gutil::Point<3,T>;

		static constexpr bool IS_CHARMS = DofHandlerType::HFlag | DofHierarchicalVariants::Charms;
		static constexpr bool IS_QH = DofHandlerType::HFlag | DofHierarchicalVariants::QuasiHierarchical;
		static constexpr bool IS_TH = DofHandlerType::HFlag | DofHierarchicalVariants::TrueHierarchical;


		//////////////////////////////////////////////////////////////////
		// Hold a snapshot of the last dofs (current coefficients)
		// and a reference to the latest dofs (what we need to transform to)
		//////////////////////////////////////////////////////////////////
		const Mesh_t& 					mesh;
		const DofHandler_t& 			dofhandler;
		const uint8_t  					max_depth;
		std::span<const DOF_t> 			dh_curr_dofs;
		std::vector<DOF_t> 				dofs{};
		gutil::BinSort<uint64_t>		sorter{};	//capture the sorter for the dofs snapshot
		std::vector<Scalar_t> 			coefs{};	//store coefficients contiguously so that we can map to a column major dense matrix
		

		//////////////////////////////////////////////////////////////////
		/// Methods to access various data fields
		//////////////////////////////////////////////////////////////////
		[[nodiscard]] Scalar_t* data(uint8_t i) noexcept {
			GUTIL_ASSERT(i<N);
			GUTIL_ASSERT(coefs.size() == N*dofs.size());
			return coefs.data() + i*dofs.size();
		}

		[[nodiscard]] const Scalar_t* data(uint8_t i) const noexcept {
			GUTIL_ASSERT(i<N);
			GUTIL_ASSERT(coefs.size() == N*dofs.size());
			return coefs.data() + i*dofs.size();
		}

		[[nodiscard]] Scalar_t* data() noexcept {
			return coefs.data();
		}

		[[nodiscard]] const Scalar_t* data() const noexcept {
			return coefs.data();
		}

		[[nodiscard]] std::span<Scalar_t> get_coefs(uint8_t i) noexcept {
			GUTIL_ASSERT(i<N);
			return {data(i), dofs.size()};
		}

		[[nodiscard]] std::span<const Scalar_t> get_coefs(uint8_t i) const noexcept {
			GUTIL_ASSERT(i<N);
			return {data(i), dofs.size()};
		}

		[[nodiscard]] size_t size() const noexcept {
			return dofs.size();
		}

		//////////////////////////////////////////////////////////////////
		/// Map or copy to an Eigen vector or matrix
		//////////////////////////////////////////////////////////////////
		#ifdef EIGEN_MAJOR_VERSION
			Eigen::Map<Eigen::Matrix<Scalar_t, Eigen::Dynamic, N, Eigen::ColMajor>> eigen_matrix_map() {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return {data(), static_cast<Eigen::Index>(dofs.size())};
			}
			Eigen::Map<const Eigen::Matrix<Scalar_t, Eigen::Dynamic, N, Eigen::ColMajor>> eigen_matrix_map() const {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return {data(), static_cast<Eigen::Index>(dofs.size())};
			}
			Eigen::Map<Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor>> eigen_vector_map(uint8_t i) {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return {data(i), static_cast<Eigen::Index>(dofs.size())};
			}
			Eigen::Map<const Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor>> eigen_vector_map(uint8_t i) const {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return {data(i), static_cast<Eigen::Index>(dofs.size())};
			}
			Eigen::Map<Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor>> eigen_vector_map() {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return {data(), static_cast<Eigen::Index>(dofs.size())};
			}
			Eigen::Map<const Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor>> eigen_vector_map() const {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return {data(), static_cast<Eigen::Index>(dofs.size())};
			}

			//note that Eigen will convert a Map<Matrix> to a Matrix via copy
			Eigen::Matrix<Scalar_t, Eigen::Dynamic, N, Eigen::ColMajor> eigen_matrix() const {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return eigen_matrix_map();
			}
			Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor> eigen_vector(uint8_t i) const {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return eigen_vector_map(i);
			}
			Eigen::Matrix<Scalar_t, Eigen::Dynamic, 1, Eigen::ColMajor> eigen_vector() const {
				GUTIL_ASSERT(coefs.size()==N*dofs.size());
				return eigen_vector_map();
			}
		#endif

		//////////////////////////////////////////////////////////////////
		/// Constructor and movement
		//////////////////////////////////////////////////////////////////
		CoefHandler() = delete;
		CoefHandler(const DofHandler_t& dofhandler) noexcept : 
			mesh{dofhandler.mesh},
			dofhandler{dofhandler},
			max_depth{dofhandler.mesh.max_depth},
			dh_curr_dofs{dofhandler.active_dofs} {}
		CoefHandler(CoefHandler&& other) noexcept = default;
		CoefHandler(const CoefHandler&) = default;
		CoefHandler& operator=(CoefHandler&& other) = delete;
		CoefHandler& operator=(const CoefHandler& other) = delete;


		///////////////////////////////////////////////////////////////////
		/// Initialize a coefficient field by a scalar function.
		/// The mesh/dofs must be in a conformal state. The 'scalar function'
		/// is actually a linear operator on the shape functions (dofs).
		/// In lagrange dofs, this is the point evaluation of a function
		/// at the corresponding vertex. Alternatively, set them all to 0.
		///////////////////////////////////////////////////////////////////
		void init_coefs() noexcept {
			GUTIL_ASSERT(dofhandler.is_current());

			auto lock = dofhandler.begin_active_keys_stable();
			update_dh_dof_link();
			snapshot_dofs();
			coefs.assign(N*dofs.size(), Scalar_t{0});
			dofhandler.end_active_keys_stable();
		}

		template<typename DofEval>
		void assign_coefs(uint8_t i, DofEval&& eval) noexcept {
			GUTIL_ASSERT(dofhandler.is_current());
			GUTIL_ASSERT(i<N);
			GUTIL_ASSERT(dofs.size()>0);
			GUTIL_ASSERT(coefs.size()==N*dofs.size());
			{
				auto lock = dofhandler.begin_active_keys_stable();
				
				GUTIL_OMP(parallel for)
				for (uint64_t idx=0; idx<dofs.size(); ++idx) {
					#ifndef NDEBUG
						bool flag = mesh.is_conformal(static_cast<MeshFeature_t>(dofhandler.feature(dofs[idx])));
						GUTIL_ASSERT(flag);
					#endif
					data(i)[idx] = eval(dofs[idx]);
				}

				dofhandler.end_active_keys_stable();
			}
		}

		///////////////////////////////////////////////////////////////////
		/// Check if the coefs are likely up to date
		///////////////////////////////////////////////////////////////////
		[[nodiscard]] bool is_current() const noexcept {
			if (!dofhandler.is_current()) {return false;}

			if (dh_curr_dofs.size() != dofhandler.active_dofs.size()) {return false;}
			if (dh_curr_dofs.data() != dofhandler.active_dofs.data()) {return false;}

			if (dofs.empty()) {return false;}
			if (dofs.size()  != dh_curr_dofs.size())  {return false;}
			if (dofs.front() != dh_curr_dofs.front()) {return false;}
			if (dofs.back()  != dh_curr_dofs.back())  {return false;}

			if (coefs.size() != N*dofs.size())        {return false;}

			return true;
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
		}

		void update_dh_dof_link() noexcept {
			//only call when the stable dofhandler locks are acquired
			dh_curr_dofs = dofhandler.active_dofs;
		}

		void snapshot_dofs() noexcept {
			//only call when the stable dofhandler locks are acquired and
			//after update_dh_dof_link() has been called.
			dofs.clear();
			dofs.insert(dofs.end(), dh_curr_dofs.begin(), dh_curr_dofs.end());
			sorter = dofhandler.get_sorter();
			//the sorter uses the raw key values
			sorter.rebind_to_copy(gutil::reinterpret_as_span<uint64_t,DOF_t>(dofs));
			GUTIL_ASSERT(sorter.n_bins()== (int) max_depth+1);
		}


		///////////////////////////////////////////////////////////////////
		/// Static methods for interpolating a field between bases.
		///////////////////////////////////////////////////////////////////
		static void ProlongCoefs(std::span<Scalar_t> pro_coefs, std::span<const DOF_t> pro_dofs,
			std::span<const Scalar_t> cur_coefs, std::span<const DOF_t> const DofHandler_t& cur_handler) noexcept {

		}




		///////////////////////////////////////////////////////////////////
		/// Transform dofs due to refinement/unrefinement
		///////////////////////////////////////////////////////////////////
		void prolong_coefs() noexcept {
			GUTIL_ASSERT(dofhandler.is_current());
			update_dh_dof_link();
			
			GUTIL_PROFILE("updating coefficients (", dofs.size(), " -> ", dh_curr_dofs.size(), ")");
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
			}
		}

		void restrict_coefs() noexcept {
			GUTIL_ASSERT(dofhandler.is_current());
			update_dh_dof_link();

			GUTIL_PROFILE("updating coefficients (", dofs.size(), " -> ", dh_curr_dofs.size(), ")");
			{
				//compute new coefficients
				auto lock1 = dofhandler.begin_key_mask_stable();
				auto lock2 = dofhandler.begin_active_keys_stable();
				std::array<std::vector<Scalar_t>,N> new_coefs;
				
				for (uint8_t i=0; i<N; ++i) {
					threads.submit([&,i](){restrict_coefs(i,new_coefs[i]);});
				}
				threads.wait_idle();

				coefs = std::move(new_coefs);
				dofhandler.end_active_keys_stable();
				dofhandler.end_key_mask_stable();
			}
			{
				//finalize
				snapshot_dofs();
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
			new_coefs.assign(new_size, Scalar_t{0});

			GUTIL_OMP(parallel)
			{
				GUTIL_OMP(for)
				for (size_t idx=0; idx<old_size; ++idx) {
					DOF_t dof = dofs[idx];

					size_t n_idx = dofhandler.global_number(dof);
					if (n_idx < new_size) {
						GUTIL_OMP(atomic)
						new_coefs[n_idx] += data(i)[idx];
						continue;
					}
					GUTIL_ASSERT(!dofhandler.is_active_stable(dof));
					distribute_refined(dof, data(i)[idx], std::span<Scalar_t>(new_coefs), new_size);
				}//for dofs
			}//omp parallel
		}//update coefs


		void restrict_coefs(uint8_t i, std::vector<Scalar_t>& new_coefs) noexcept {
			GUTIL_ASSERT(dofhandler.is_active_keys_stable());
			GUTIL_ASSERT(dofhandler.is_key_mask_stable());
			GUTIL_ASSERT(i<N);

			// const size_t old_size = dofs.size();
			const size_t new_size = dh_curr_dofs.size();
			new_coefs.assign(new_size, Scalar_t{0});
			auto cur_sorter = dofhandler.get_sorter();

			GUTIL_OMP(parallel)
			{
				const size_t n_threads = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
				const size_t tid       = GUTIL_OMP_TERNARY(omp_get_thread_num(),  0);

				for (int dd=0; dd<cur_sorter.n_bins(); ++dd) {
					std::span<const DOF_t> list = gutil::reinterpret_as_span<DOF_t>(cur_sorter.get_bin(dd));
						// dofhandler.template reinterpret_key_span<DOF_t,uint64_t>(cur_sorter.get_bin(dd));
				
					//transfer same depth dofs
					//split the dofs at this depth into batches to be evaluated
					const size_t n_dofs 	= list.size();
					const size_t batch_size = n_dofs/n_threads;
					const size_t start      = tid*batch_size;
					const size_t end        = (tid==n_threads-1) ? n_dofs : start+batch_size;

					//collect the dof locations (at this depth) that this thread is responsible for
					std::span<const DofVert_t> thread_dof_locs = gutil::reinterpret_as_span<DofVert_t>(list.begin()+start, list.begin()+end);

					//collect the new coefficients (at this depth) that this thread is responsible for incrementing
					const size_t new_idx_start = cur_sorter.bin_start(dd) + start;
					const size_t new_idx_end   = cur_sorter.bin_start(dd) + end;
					std::span<Scalar_t> thread_new_coefs(new_coefs.begin()+new_idx_start, new_coefs.begin()+new_idx_end);

					//increment the dofs using the old coefs
					batched_evaluate_at(thread_new_coefs, thread_dof_locs, data(i), sorter, max_depth);

					//decrement the dofs using the new coefs at lower depths
					if (dd>0) {
						batched_evaluate_at<false>(thread_new_coefs, thread_dof_locs, new_coefs, cur_sorter, dd-1);
					}
					GUTIL_OMP(barrier)
				}
			}
		}



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
				Scalar_t parent_contribution = contribution * dof.template parent_coef_restrict<Scalar_t>(p_dofs[p]);
				if (parent_contribution == Scalar_t{0}) {continue;}

				size_t n_idx = dofhandler.global_number(p_dofs[p]);
				if (n_idx<new_size) {
					GUTIL_OMP(atomic) new_coefs[n_idx] += parent_contribution;
				}
				else if (!dofhandler.is_active_stable(p_dofs[p])) {
					distribute_unrefined_restrict(p_dofs[p], parent_contribution, new_coefs, new_size);
				}
			}
		}


	public:
		////////////////////////////////////////////////////////////////////////
		/// Evaluate at mesh vertices for visualizations
		/// Pass begin/end iterators to the existing vertices
		////////////////////////////////////////////////////////////////////////
		template<std::random_access_iterator I> requires (std::same_as<std::iter_value_t<I>, MeshVert_t>)
		std::vector<Scalar_t> evaluate(uint8_t i, I v_begin, I v_end) const noexcept {
			GUTIL_ASSERT(i<N);
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(v_begin!=v_end && "there are no vertices. did you forget to collect them?");

			const size_t n_verts = std::distance(v_begin,v_end);
			GUTIL_ASSERT(v_end == v_begin+n_verts);
			
			//initialize storage
			std::vector<Scalar_t> vals(n_verts, Scalar_t{0});
			GUTIL_OMP(parallel)
			{
				const gutil::OmpIteratorRange range(v_begin, v_end);

				size_t idx = std::distance(v_begin, range.begin);
				auto action = [&](DOF_t dof, uint8_t local_n, Scalar_t x, Scalar_t y, Scalar_t z, uint64_t global_n) {
					Scalar_t val{0};
					dof.evaluate_simd(local_n, &val, &x, &y, &z, 1);
					vals[idx] += data(i)[global_n] * val;
				};

				for (I it = range.begin; it!=range.end; ++it, ++idx) {
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

		protected:
		[[nodiscard]] static Scalar_t evaluate_at(DofVert_t loc, std::span<const Scalar_t> coefs, const gutil::BinSort<uint64_t>& dof_depth_sorter, uint8_t target_depth) noexcept {
			//evaluate the scalar field using dofs up to the target depth
			GUTIL_ASSERT(dof_depth_sorter.n_bins() >= target_depth);
			GUTIL_ASSERT(dof_depth_sorter.size() == coefs.size());
			
			Scalar_t result{0};
			for (int dd=0; dd<= (int) target_depth; ++dd) {
				size_t depth_start = dof_depth_sorter.bin_start(dd);
				size_t depth_size  = dof_depth_sorter.bin_size(dd);
				
				std::span<const DOF_t> dofs_at_depth = gutil::reinterpret_as_span<DOF_t>(
					dof_depth_sorter.get_bin(dd));

				result += DOF_t::evaluate_field_at_depth(
						loc,coefs.subspan(depth_start, depth_size), 
							gutil::reinterpret_as_span<DOF_t>(dof_depth_sorter.get_bin(dd)), (uint64_t) dd);
			}
			return result;
		}

		template<bool Increment=true>
		static void batched_evaluate_at(std::span<Scalar_t> vals, std::span<const DofVert_t> loc, 
				std::span<const Scalar_t> coefs, const gutil::BinSort<uint64_t>& dof_depth_sorter, uint8_t target_depth) noexcept {
			//note vals are incremented or decremented, not assigned.
			//evaluate the scalar field using dofs up to the target depth
			//the provided locations must all be at the same depth and be sorted in increasing order
			GUTIL_ASSERT(dof_depth_sorter.n_bins() >= target_depth);
			GUTIL_ASSERT(dof_depth_sorter.size() == coefs.size());
			GUTIL_ASSERT(vals.size()==loc.size());
			if (loc.size()==0) {return;}

			for (int dd=0; dd<= (int) target_depth; ++dd) {
				size_t depth_start = dof_depth_sorter.bin_start(dd);
				size_t depth_size  = dof_depth_sorter.bin_size(dd);
				
				std::span<const DOF_t> dofs_at_depth = gutil::reinterpret_as_span<DOF_t>(
					dof_depth_sorter.get_bin(dd));

				DOF_t::template batched_evaluate_field_at_depth<Increment>(
					vals, loc, coefs.subspan(depth_start, depth_size), 
					dofs_at_depth, (uint64_t) dd);
			}
		}


		//////////////////////////////////////////////////////////////////////
		/// Debugging
		//////////////////////////////////////////////////////////////////////
		public:
		void print_coefs(int i=0) {
			std::cout << "\nCoefs for field " << i << ":\n";
			for (size_t idx=0; idx<dofs.size(); ++idx) {
				std::cout << idx << " : " << dofs[idx] << " -> " << data(i)[idx] << "\n";
			}
		}
	};





}


