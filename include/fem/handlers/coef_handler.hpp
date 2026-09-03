#pragma once

#include "gutil.hpp"

#include "simd_keys/mesh/mesh_key_implementation.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"
#include "simd_keys/dofs/utility.hpp"

#include <iostream>
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

		static constexpr bool IS_CONFORMAL = DofHandlerType::HFlag & DofHierarchicalVariants::Conformal;
		static constexpr bool IS_QH = DofHandlerType::HFlag & DofHierarchicalVariants::QuasiHierarchical;
		static constexpr bool IS_TH = DofHandlerType::HFlag & DofHierarchicalVariants::TrueHierarchical;


		//////////////////////////////////////////////////////////////////
		// Hold a snapshot of the last dofs (current coefficients)
		// and a reference to the latest dofs (what we need to transform to)
		//////////////////////////////////////////////////////////////////
		const Mesh_t& 					mesh;
		const DofHandler_t& 			d_handler;
		const uint8_t  					max_depth;
		gutil::BinSortVector<DOF_t>     dofs{};
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
		CoefHandler(const DofHandler_t& d_handler) noexcept : 
			mesh{d_handler.mesh},
			d_handler{d_handler},
			max_depth{d_handler.mesh.max_depth} {}
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
			GUTIL_ASSERT(d_handler.is_current());
			snapshot_dofs();
			coefs.assign(N*dofs.size(), Scalar_t{0});
		}

		template<typename DofEval>
		void assign_coefs(uint8_t i, DofEval&& eval) noexcept {
			GUTIL_ASSERT(d_handler.is_current());
			GUTIL_ASSERT(i<N);
			GUTIL_ASSERT(dofs.size()>0);
			GUTIL_ASSERT(coefs.size()==N*dofs.size());
			{
				auto lock = d_handler.begin_active_keys_stable();
				
				GUTIL_OMP(parallel for)
				for (uint64_t idx=0; idx<dofs.size(); ++idx) {
					#ifndef NDEBUG
						bool flag = mesh.is_conformal(static_cast<MeshFeature_t>(d_handler.feature(dofs[idx])));
						GUTIL_ASSERT(flag);
					#endif
					data(i)[idx] = eval(dofs[idx]);
				}

				d_handler.end_active_keys_stable();
			}
		}

		///////////////////////////////////////////////////////////////////
		/// Check if the coefs are likely up to date
		///////////////////////////////////////////////////////////////////
		[[nodiscard]] bool is_current() const noexcept {
			if (!d_handler.is_current()) {return false;}

			//just check size for now
			if (dofs.size()  != d_handler.n_dofs())  {return false;}
			if (coefs.size() != N*dofs.size())     {return false;}

			return true;
		}

		///////////////////////////////////////////////////////////////////
		/// Look up previous global dof numbers and get snapshot of the handler's current dofs
		///////////////////////////////////////////////////////////////////
		void snapshot_dofs() noexcept {
			dofs = d_handler.collect_snapshot(0);
			GUTIL_ASSERT(dofs.n_bins() == max_depth+1);
		}

		///////////////////////////////////////////////////////////////////
		/// Transfer coefficients
		///////////////////////////////////////////////////////////////////
		void prolong_coefs(uint8_t fine_number=0, uint8_t coarse_number=0) noexcept {
			GUTIL_ASSERT(d_handler.is_current());
			auto lock = d_handler.begin_active_keys_stable();

			gutil::BinSortVector<DOF_t> new_dofs = d_handler.collect_snapshot(0);
			std::vector<Scalar_t> new_coefs(N*new_dofs.size());
			ProlongCoefs(new_coefs, new_dofs, coefs, dofs, d_handler, fine_number, coarse_number);
			coefs = std::move(new_coefs);
			dofs  = std::move(new_dofs);

			d_handler.end_active_keys_stable();
		}

		void restrict_coefs(uint8_t fine_number=0, uint8_t coarse_number=0) noexcept {
			GUTIL_ASSERT(d_handler.is_current());
			auto lock = d_handler.begin_active_keys_stable();

			gutil::BinSortVector<DOF_t> new_dofs = d_handler.collect_snapshot(0);
			std::vector<Scalar_t> new_coefs(N*new_dofs.size());
			RestrictCoefs(coefs, dofs, new_coefs, new_dofs, d_handler, fine_number, coarse_number);
			coefs = std::move(new_coefs);
			dofs  = std::move(new_dofs);

			d_handler.end_active_keys_stable();
		}

		///////////////////////////////////////////////////////////////////
		/// Static methods for interpolating a field between bases.
		///////////////////////////////////////////////////////////////////
		static void ProlongCoefs(std::span<Scalar_t> fine_coefs, const gutil::BinSortVector<DOF_t>& fine_dofs,
			std::span<const Scalar_t> coarse_coefs, const gutil::BinSortVector<DOF_t>& coarse_dofs,
			const DofHandler_t& handler, uint8_t fine_number, uint8_t coarse_number) noexcept {
			GUTIL_ASSERT(fine_coefs.size()==N*fine_dofs.size());
			GUTIL_ASSERT(coarse_coefs.size()==N*coarse_dofs.size());

			const size_t coarse_size = coarse_dofs.size();
			const size_t fine_size   = fine_dofs.size();
			std::fill(fine_coefs.begin(), fine_coefs.end(), Scalar_t{0});

			GUTIL_OMP(parallel for)
			for (size_t idx=0; idx<coarse_size; ++idx) {
				DOF_t dof = coarse_dofs[idx];
				
				//aggregate the contribution for each component
				Scalar_t contribution[N];
				for (uint8_t f=0; f<N; ++f) {
					contribution[f] = coarse_coefs[f*coarse_size + idx];
				}

				//under a true hierachical regime, the global dof numbers may change,
				//but the coefs must be directly transfered over
				if constexpr (IS_TH) {
					size_t n_idx = fine_dofs.index_sorted(dof);
					if (n_idx<fine_size) {
						for (uint8_t f=0; f<N; ++f) {
							GUTIL_OMP(atomic)
							fine_coefs[f*fine_size + n_idx] += contribution[f];
						}
					}
				}
				else {
					DistributeToChildren(dof, contribution, fine_coefs, fine_dofs, handler, fine_number);
				}
			}
		}

		static void RestrictCoefs(std::span<const Scalar_t> fine_coefs, const gutil::BinSortVector<DOF_t>& fine_dofs,
			std::span<Scalar_t> coarse_coefs, const gutil::BinSortVector<DOF_t>& coarse_dofs,
			const DofHandler_t& handler, uint8_t fine_number, uint8_t coarse_number) noexcept {
			GUTIL_ASSERT(fine_coefs.size()==N*fine_dofs.size());
			GUTIL_ASSERT(coarse_coefs.size()==N*coarse_dofs.size());

			const size_t coarse_size = coarse_dofs.size();
			const size_t fine_size   = fine_dofs.size();
			std::fill(coarse_coefs.begin(), coarse_coefs.end(), Scalar_t{0});
			const uint8_t max_depth = handler.mesh.max_depth;

			for (uint8_t f=0; f<N; ++f) {
				std::span<const Scalar_t> fine_field   = fine_coefs.subspan(f*fine_size, fine_size);
				std::span<Scalar_t>       coarse_field = coarse_coefs.subspan(f*coarse_size, coarse_size);

				GUTIL_OMP(parallel)
				{
					for (int dd=0; dd<coarse_dofs.n_bins(); ++dd) {
						std::span<const DOF_t> list = coarse_dofs.get_bin(dd);
						const gutil::OmpIteratorRange range(list.begin(), list.end());

						//TODO: this must change to be dof agnostic or dispatch by dof type at compile time
						std::span<const DofVert_t> thread_dof_locs = gutil::reinterpret_as_span<DofVert_t>(range.begin, range.end);

						const size_t new_idx_start = coarse_dofs.bin_start(dd) + std::distance(list.begin(), range.begin);
						const size_t new_idx_end   = coarse_dofs.bin_start(dd) + std::distance(list.begin(), range.end);
						std::span<Scalar_t> thread_new_coefs(coarse_field.begin()+new_idx_start, coarse_field.begin()+new_idx_end);

						batched_evaluate_at(thread_new_coefs, thread_dof_locs, fine_field, fine_dofs, max_depth);

						if (dd>0) {
							batched_evaluate_at<false>(thread_new_coefs, thread_dof_locs, coarse_field, coarse_dofs, dd-1);
						}
						GUTIL_OMP(barrier)
					}
				}
			}
		}



		//////////////////////////////////////////////////////////////////////
		/// Debugging
		//////////////////////////////////////////////////////////////////////
		void print_coefs(int i=0) {
			std::cout << "\nCoefs for field " << i << ":\n";
			for (size_t idx=0; idx<dofs.size(); ++idx) {
				std::cout << idx << " : " << dofs[idx] << " -> " << data(i)[idx] << "\n";
			}
		}

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
					
					d_handler.gather_hierarchical_dofs_at_vertex(action, dv);
				}
			}

			return vals;
		}

		protected:
		static void DistributeToChildren(DOF_t dof, const Scalar_t* contribution, std::span<Scalar_t> fine_coefs,
			const gutil::BinSortVector<DOF_t>& fine_dofs, const DofHandler_t& handler, uint8_t fine_number) noexcept requires (IS_QH) {
			//assume that if we are not in TH, a dof is either refined, active, or irrelevant
			if (IsRefined(handler, dof, fine_number)) {
				DOF_t c_dofs[DOF_t::N_CHILDREN];
				dof.children_simd(c_dofs);
				for (uint8_t c=0; c<DOF_t::N_CHILDREN; ++c) {
					if (!c_dofs[c].exists()) {continue;}
					Scalar_t weight = dof.template child_coef<Scalar_t>(c);
					Scalar_t child_contribution[N];
					for (uint8_t f=0; f<N; ++f) { child_contribution[f] = contribution[f] * weight; }
					DistributeToChildren(c_dofs[c], child_contribution, fine_coefs, fine_dofs, handler, fine_number);
				}
			}
			else if (IsActive(handler, dof, fine_number)) {
				const size_t fine_size = fine_dofs.size();
				size_t n_idx = fine_dofs.index_sorted(dof);
				GUTIL_ASSERT(n_idx < fine_size);
				for (uint8_t f=0; f<N; ++f) {
					GUTIL_OMP(atomic)
					fine_coefs[f*fine_size + n_idx] += contribution[f];
				}
			}
		}


		template<bool Increment=true>
		static void batched_evaluate_at(std::span<Scalar_t> vals, std::span<const DofVert_t> loc, 
				std::span<const Scalar_t> coefs, const gutil::BinSortVector<DOF_t>& dof_depth_sorter, uint8_t target_depth) noexcept {
			//note vals are incremented or decremented, not assigned.
			//evaluate the scalar field using dofs up to the target depth
			//the provided locations must all be at the same depth and be sorted in increasing order
			GUTIL_ASSERT(dof_depth_sorter.n_bins() > target_depth);
			GUTIL_ASSERT(dof_depth_sorter.size() == coefs.size());
			GUTIL_ASSERT(vals.size()==loc.size());
			if (loc.size()==0) {return;}

			for (int dd=0; dd<= (int) target_depth; ++dd) {
				size_t depth_start = dof_depth_sorter.bin_start(dd);
				size_t depth_size  = dof_depth_sorter.bin_size(dd);
				
				std::span<const DOF_t> dofs_at_depth = dof_depth_sorter.get_bin(dd);

				DOF_t::template batched_evaluate_field_at_depth<Increment>(
					vals, loc, coefs.subspan(depth_start, depth_size), 
					dofs_at_depth, (uint64_t) dd);
			}
		}


		//////////////////////////////////////////////////////////////////////
		/// Adaptor to get snapshot information from a dof_handler from its snapshot number
		//////////////////////////////////////////////////////////////////////
		[[nodiscard]] static bool IsActive(const DofHandler_t& handler, DOF_t dof, uint8_t snapshot_number) noexcept {
			switch (snapshot_number) {
				case 0:  return handler.is_active_stable(dof);
				case 1:  return handler.is_snapshot_A_active(dof);
				case 2:  return handler.is_snapshot_B_active(dof);
				default: GUTIL_ABORT("invalid snapshot number"); return false;
			}
		}
		[[nodiscard]] static bool IsRefined(const DofHandler_t& handler, DOF_t dof, uint8_t snapshot_number) noexcept {
			switch (snapshot_number) {
				case 0:  return handler.is_refined_stable(dof);
				case 1:  return handler.is_snapshot_A_refined(dof);
				case 2:  return handler.is_snapshot_B_refined(dof);
				default: GUTIL_ABORT("invalid snapshot number"); return false;
			}
		}
	};





}


