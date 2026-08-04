#pragma once

#include "gutil.hpp"

#include "simd_keys/containers.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"
#include "util/byte_print.hpp"

#include <concepts>
#include <cstdint>
#include <vector>
#include <span>
#include <type_traits>
#include <algorithm>

namespace GV {


	/////////////////////////////////////////////////////////////////////////////
	/// Concepts to check compatibility of the mesh and DOF types
	/////////////////////////////////////////////////////////////////////////////
	template<typename Mesh_t>
	concept VoxelMeshType = requires(const Mesh_t& mesh, 
		typename Mesh_t::VoxelElement el, 
		typename Mesh_t::VoxelVertex vtx) {

		//necessary aliases
		typename Mesh_t::VoxelElement;
		typename Mesh_t::VoxelVertex;
		typename Mesh_t::VoxelFace;
		typename Mesh_t::GeoPoint_t;

		//necessary constants
		{ Mesh_t::MAX_DEPTH } 		-> std::convertible_to<uint64_t>;

		//necessary queries
		{ mesh.is_active(el) }		-> std::same_as<bool>;
		{ mesh.is_conformal(vtx) }	-> std::same_as<bool>;
		{ mesh.get_conformal(vtx) }	-> std::same_as<typename Mesh_t::VoxelVertex>;
		{ mesh.n_elements() } 		-> std::same_as<uint64_t>;
		{ mesh.n_vertices() }		-> std::same_as<uint64_t>;

		//necessary iterators (might not be contiguous)
		mesh.element_begin(); mesh.element_end();
		mesh.vertex_begin();  mesh.vertex_end();
	};

	template<typename Mesh_t>
	concept DepthSeparableVoxelMeshType = VoxelMeshType<Mesh_t> &&
		requires(const Mesh_t& mesh) {
			//contiguous iterators for each depth
			mesh.element_begin(0); mesh.element_end(0);
			mesh.vertex_begin(0);  mesh.vertex_end(0);
		};


	/////////////////////////////////////////////////////////////////////////////
	/// DOF handler class.
	///
	/// Note that periodic dofs are assigned to the periodic feature with least indices.
	/// For a 1D example   0 --- 1 --- 2 --- 3 --- 4
	/// where nodes 0 and 4 are identified, they will be different vertices in the mesh but both
	/// vertex keys will produce the same DOF that 'lives' at vertex 0. To check if we can assign
	/// a dof to a feature (due to periodicity only, not active elements), we cast the dof key to 
	/// the mesh key and back. Note that this casting will produce vertex 0 as the output for dof keys at 0 and 4,
	/// so we then check for equality.
	/////////////////////////////////////////////////////////////////////////////
	template<VoxelMeshType MeshType, typename DofType>
	struct DofHandler : public Keys::HybridKeyTracker {
		using BASE = Keys::HybridKeyTracker;

		/////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////////////////////
		static constexpr bool VERTEX_DOF  = DofType::ID == Keys::Mesh3D::VERTEX_FLAG;
		static constexpr bool ELEMENT_DOF = DofType::ID == Keys::Mesh3D::ELEMENT_FLAG;
		static constexpr bool FACE_DOF    = DofType::ID == Keys::Mesh3D::FACE_FLAG;
		
		//DOF features may be periodic
		using DOF_t        = DofType;
		using DofVert_t    = Keys::VoxelVertex<DOF_t::PERIOD>;
		using DofElem_t    = Keys::VoxelElement<DOF_t::PERIOD>;
		using DofFeature_t = std::conditional_t<VERTEX_DOF, DofVert_t, std::conditional_t<ELEMENT_DOF, DofElem_t, void>>;
		static_assert(!std::same_as<DofFeature_t,void>);

		//Mesh features are never periodic
		using Mesh_t        = MeshType;
		using MeshElem_t    = typename Mesh_t::VoxelElement;
		using MeshVert_t    = typename Mesh_t::VoxelVertex;
		using MeshFeature_t = std::conditional_t<VERTEX_DOF, MeshVert_t, std::conditional_t<ELEMENT_DOF, MeshElem_t, void>>;
		static_assert(!std::same_as<MeshFeature_t,void>);

		static constexpr bool     IS_DEPTH_SEPARABLE  = DepthSeparableVoxelMeshType<Mesh_t>;
		static constexpr uint8_t  MAX_DEPTH           = Mesh_t::MAX_DEPTH;
		static constexpr uint64_t TOTAL_POSSIBLE_DOFS = DOF_t::total_possible(MAX_DEPTH);

		using BASE::ACTIVE_BIT; 						//0b00000001;	
		static constexpr uint8_t REFINED_BIT 			= 0b00000010;
		static constexpr uint8_t INITIAL_DOF_BIT        = 0b00000100;	//these dofs don't check their parents and cannot be unrefined
		static constexpr uint8_t COEF_MARKED_BIT        = 0b00001000;	//one coef handler per dof handler (it can handle multiple fields)
		static constexpr uint8_t BATCH_PROCESS_BIT      = 0b00010000;	//mark all active dofs at the start of a batch process for synchronization
		static constexpr uint8_t FREE_BITS 	            = 0b11100000;


		/////////////////////////////////////////////////////////////////////////
		/// Storage. Store a vector<uint8_t> for O(1) active queries.
		/// Additionally, store a compressed list of active dofs for tracking
		/// global DOF numbers. It is essential for fast quadrature that we may
		/// look up all active DOFs whos support OVERLAPS a given active element.
		///
		/// Using uint8_t instead of bool guarantees thread safe access of different elements
		/// and allows one bit to be used for an "is active" flag and another bit for
		/// "has been refined" flag, which is useful for hierarchical methods. Additionally,
		/// it gives us 6 more bits that could be used for other purposes.
		/////////////////////////////////////////////////////////////////////////
		protected:
		using BASE::key_mask;
		using BASE::threads;										//max hardware concurency by default
		public:
		// std::vector<DOF_t> 						active_dofs{};		//a compressed list of the active dofs
		std::span<DOF_t>						active_dofs;
		gutil::BinSort<DOF_t> 					active_dof_sorter;	//responsible for global dof numbers and lookup
		const Mesh_t& 							mesh;				//link to the mesh, we can request refinement through const methods
		mutable bool 							batch_is_set{false};//check if the batch start bit has been set.

		/////////////////////////////////////////////////////////////////////////
		/// Constructors. The dofhandler must be linked to the mesh at construction
		/// and the mesh must outlive the dofhandler.
		/////////////////////////////////////////////////////////////////////////
		DofHandler() = delete;
		DofHandler(const Mesh_t& m) : BASE(TOTAL_POSSIBLE_DOFS), mesh(m) {}
		DofHandler(const Mesh_t& m, int n_threads) : BASE(TOTAL_POSSIBLE_DOFS,n_threads), mesh{m} {}
		DofHandler(const DofHandler& other) : BASE(other), mesh(other.mesh) {}
		DofHandler(DofHandler&& other) : BASE(other), mesh(other.mesh) {}
		
		[[nodiscard]] DofHandler& operator=(const DofHandler& other) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				key_mask = other.key_mask;
				active_dofs = other.active_dofs;
				active_dof_sorter = other.active_dof_sorter;
			}
			return *this;
		}
		[[nodiscard]] DofHandler& operator=(DofHandler&& other) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				key_mask = std::move(other.key_mask);
				active_dofs = std::move(other.active_dofs);
				active_dof_sorter = std::move(other.active_dof_sorter);
			}
			return *this;
		}


		/////////////////////////////////////////////////////////////////////////
		/// Adapt the specific mesh api to the feature in question
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] uint64_t n_features() const noexcept {
			if constexpr (VERTEX_DOF) { return mesh.n_vertices(); }
			else if constexpr (ELEMENT_DOF) { return mesh.n_elements(); }
			else {assert(false); return 0;}
		}
		
		auto feature_begin() const noexcept {
			if constexpr (VERTEX_DOF) { return mesh.vertex_begin(); }
			else if constexpr (ELEMENT_DOF) { return mesh.element_begin(); }
			else {assert(false); return 0;}
		}

		auto feature_end() const noexcept {
			if constexpr (VERTEX_DOF) { return mesh.vertex_end(); }
			else if constexpr (ELEMENT_DOF) { return mesh.element_end(); }
			else {assert(false); return 0;}
		}

		auto feature_begin(uint64_t dd) const noexcept requires DepthSeparableVoxelMeshType<Mesh_t> {
			if constexpr (VERTEX_DOF) { return mesh.vertex_begin(dd); }
			else if constexpr (ELEMENT_DOF) { return mesh.element_begin(dd); }
			else {assert(false); return 0;}
		}

		auto feature_end(uint64_t dd) const noexcept requires DepthSeparableVoxelMeshType<Mesh_t> {
			if constexpr (VERTEX_DOF) { return mesh.vertex_end(dd); }
			else if constexpr (ELEMENT_DOF) { return mesh.element_end(dd); }
			else {assert(false); return 0;}
		}

		template<typename Elem_t> requires (std::same_as<Elem_t,MeshElem_t> || std::same_as<Elem_t,DofElem_t>)
		[[nodiscard]] static constexpr auto features(Elem_t el) noexcept {
			if constexpr (VERTEX_DOF) { return el.vertices(); }
			else if constexpr (ELEMENT_DOF) { return std::array<Elem_t,1>{el}; }
			else { return el.faces(); }
		}

		template<typename Feature_t> requires (std::same_as<Feature_t,MeshFeature_t> || std::same_as<Feature_t,DofElem_t>)
		[[nodiscard]] static constexpr auto elements(Feature_t feat) noexcept {
			if constexpr (ELEMENT_DOF) { return std::array<Feature_t,1>{feat}; }
			else { return feat.elements(); }
		}

		[[nodiscard]] static constexpr DofFeature_t feature(DOF_t dof) noexcept {
			return DofFeature_t{dof.key};
		}


		/////////////////////////////////////////////////////////////////////////
		/// Utility methods. The set_* methods return true if the status changed.
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] uint8_t get_mask(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid() && dof.linear_index()<key_mask.size());
			return key_mask[dof.linear_index()];
		}

		[[nodiscard]] uint8_t& get_mask_ref(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid() && dof.linear_index()<key_mask.size());
			return key_mask[dof.linear_index()];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t get_mask(uint64_t idx) const noexcept {
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t& get_mask_ref(uint64_t idx) const noexcept {
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		[[nodiscard]] bool is_active(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			return ACTIVE_BIT & get_mask(dof);
		}

		[[nodiscard]] bool is_refined(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			return REFINED_BIT & get_mask(dof);
		}

		[[nodiscard]] bool is_coef_marked(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			return COEF_MARKED_BIT & get_mask(dof);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_active(uint64_t d_idx) const noexcept {
			GUTIL_ASSERT(d_idx < TOTAL_POSSIBLE_DOFS);
			return ACTIVE_BIT & get_mask(d_idx);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_refined(uint64_t d_idx) const noexcept {
			GUTIL_ASSERT(d_idx < TOTAL_POSSIBLE_DOFS);
			return REFINED_BIT & get_mask(d_idx);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_coef_marked(uint64_t d_idx) const noexcept {
			GUTIL_ASSERT(d_idx < TOTAL_POSSIBLE_DOFS);
			return COEF_MARKED_BIT & get_mask(d_idx);
		}

		[[maybe_unused]] bool set_active(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid());
			uint8_t& byte  = get_mask_ref(dof);
			const bool old = byte&ACTIVE_BIT;
			if (old==val) {return false;}
			val ? byte|=ACTIVE_BIT : byte&=~ACTIVE_BIT;
			return true;
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] bool set_active(uint64_t d_idx, bool val) noexcept {
			GUTIL_ASSERT(d_idx < TOTAL_POSSIBLE_DOFS);
			uint8_t& byte  = get_mask_ref(d_idx);
			const bool old = byte&ACTIVE_BIT;
			if (old==val) {return false;}
			val ? byte|=ACTIVE_BIT : byte&=~ACTIVE_BIT;
			return true;
		}

		[[maybe_unused]] bool set_refined(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid())
			uint8_t& byte  = get_mask_ref(dof);
			const bool old = byte&REFINED_BIT;
			if (old==val) {return false;}
			val ? byte|=REFINED_BIT : byte&=~REFINED_BIT;
			#ifndef NDEBUG
				if (!val) GUTIL_ASSERT(!(bool)(byte&INITIAL_DOF_BIT)); //the initial dofs should never be unrefined
			#endif
			return true;
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] bool set_refined(uint64_t d_idx, bool val) noexcept {
			GUTIL_ASSERT(d_idx < TOTAL_POSSIBLE_DOFS);
			uint8_t& byte  = get_mask_ref(d_idx);
			const bool old = byte&REFINED_BIT;
			if (old==val) {return false;}
			val ? byte|=REFINED_BIT : byte&=~REFINED_BIT;
			#ifndef NDEBUG
				if (!val) GUTIL_ASSERT(!(bool)(byte&INITIAL_DOF_BIT)); //the initial dofs should never be unrefined
			#endif
			return true;
		}

		[[maybe_unused]] bool set_coef_marked(DOF_t dof, bool val) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			uint8_t& byte  = get_mask_ref(dof);
			const bool old = byte&COEF_MARKED_BIT;
			if (old==val) {return false;}
			val ? byte|=COEF_MARKED_BIT : byte&=~COEF_MARKED_BIT;
			return true;
		}

		[[maybe_unused]] bool set_coef_marked(uint64_t d_idx, bool val) const noexcept {
			GUTIL_ASSERT(d_idx < TOTAL_POSSIBLE_DOFS)
			uint8_t& byte  = get_mask_ref(d_idx);
			const bool old = byte&COEF_MARKED_BIT;
			if (old==val) {return false;}
			val ? byte|=COEF_MARKED_BIT : byte&=~COEF_MARKED_BIT;
			return true;
		}

		[[nodiscard]] static constexpr int dof_bin(DOF_t dof) noexcept {
			int bin = 0;
			const uint64_t dd = dof.depth();
			const uint64_t split = (dd>0) ? (uint64_t{1}<<(dd-1)) : 0;	//split the index space into 8 octants. ties go to lower bin.
			
			if (dof.i() > split) { bin |= 1;}
			if (dof.j() > split) { bin |= 2;}
			if (dof.k() > split) { bin |= 4;}
			return bin;
		}

		//get a single dof global number
		[[nodiscard]] uint64_t global_number(DOF_t dof) const noexcept {
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			const int bin = dof_bin(dof);
			auto list = active_dof_sorter.get_bin(bin);
			auto it = std::lower_bound(list.begin(), list.end(), dof); 
			return (it==list.end() || *it!=dof) ? uint64_t(-1) : active_dof_sorter.bin_start(bin) + std::distance(list.begin(), it);
		}

		//for vertex dofs, sometimes getting the correct depth is awkward.
		//determine if any (there is at most one) dof lives at the same geometric location
		//as the requested vertex
		template<typename Vert_t> requires(std::same_as<Vert_t,MeshVert_t> || std::same_as<Vert_t,DofVert_t>)
		[[nodiscard]] DofFeature_t get_dof_vertex(Vert_t vtx_) const noexcept requires(VERTEX_DOF) {
			DofVert_t vtx = static_cast<DofVert_t>(vtx_);
			if (!DOF_t{vtx}.exists()) {return DofFeature_t::None();} //not a valid feature for periodic dofs
			//traverse to shallowest mesh key, then check going down
			while (vtx.parent().exists()) { vtx = vtx.parent(); }
			for (uint64_t dd=vtx.depth(); dd<=MAX_DEPTH; ++dd) {
				if (is_active(vtx.linear_index())) {return vtx;}
				else if (vtx.child().exists()) {vtx = vtx.child();}
				else {break;}
			}
			return DofFeature_t::None();
		}  

		//sort the dofs by increasing global index and get their global index
		void get_dof_number_local_sort(std::span<DOF_t> dofs, std::span<uint64_t> global_idx) const noexcept {
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			auto local_sorter = sort_dofs_singlethread(dofs);
			uint64_t n = 0; //the local dof number (index into the dofs)
			for (int i=0; i<8; ++i) {
				auto local_list = local_sorter.get_bin(i);
				auto global_list = active_dof_sorter.get_bin(i);
				for (uint64_t j=0; j<local_sorter.bin_size(i); ++j, ++n) {
					DOF_t dof = local_list[j]; //subspan of the sorted original dofs
					GUTIL_ASSERT(dof==dofs[n]);
					auto it = std::lower_bound(global_list.begin(), global_list.end(), dof);
					if ( it==global_list.end() || *it!=dof) { global_idx[n] = uint64_t(-1); }
					else {
						global_idx[n] = active_dof_sorter.bin_start(i) + 
								static_cast<uint64_t>(std::distance(global_list.begin(), it));
					}
				}
			}
		}
		
		





		/////////////////////////////////////////////////////////////////////////
		/// Initialization. The mesh must be in a conformal state.
		/////////////////////////////////////////////////////////////////////////
		void init_dofs() noexcept { dispatch_init_dofs(); wait_idle();}
		void dispatch_init_dofs() noexcept {
			if (mesh.element_begin()==mesh.element_end()) {
				GUTIL_ABORT("ERROR: no elements found.")
			}

			clear();
			
			const uint64_t n_workers  = threads.n_threads();
			const uint64_t el_per_thread = (n_workers==0) ? mesh.n_elements() : mesh.n_elements()/n_workers;

			for (uint64_t tid=0; tid<n_workers; ++tid) {
				const uint64_t start = tid * el_per_thread;
				const uint64_t end   = (tid==n_workers-1) ? mesh.n_elements() : start + el_per_thread;
				auto job = [&](auto it, auto end) {
					while (it != end) {
						for (MeshFeature_t feat : features(*it)) {
							const DOF_t dof{feat}; //constructor handles period transformation if needed
							#ifndef NDEBUG
							DofFeature_t d_feat{feat};
							if (d_feat.exists() && !mesh.is_geometrically_conformal(d_feat)) {
								GUTIL_ASSERT(dof==DOF_t{d_feat});
								GUTIL_ERROR("dof_feature ", d_feat, " is not conformal in the mesh");
								std::terminate();
							}
							#endif
							if (dof.exists()) {
								const uint64_t idx = dof.linear_index();
								set_active(idx, true);

								//mark as explicitly refinable
								if (dof.depth() != MAX_DEPTH) {
									key_mask[idx] |= INITIAL_DOF_BIT;
								}
							}
						}
						++it;
					}
				};
				GUTIL_ASSERT(start<=end);
				threads.submit(job, mesh.element_begin()+start, mesh.element_begin()+end);
			}
		}


		/////////////////////////////////////////////////////////////////////////
		/// Book keeping methods
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] size_t n_dofs() const noexcept { return active_dofs.size(); }

		
		void sort_dofs() noexcept {
			// BASE::sort_keys(8, &DofHandler::dof_bin);
			active_dof_sorter = (active_dofs.size() > 1024) ? 
					sort_dofs_multithread(active_dofs) : sort_dofs_singlethread(active_dofs);
		}

		[[nodiscard]] gutil::BinSort<DOF_t> sort_dofs_multithread(std::span<DOF_t> dofs) const noexcept {
			gutil::BinSort<DOF_t> local_sorter(dofs, 8);
			local_sorter.dispatch_sort([&](DOF_t dof){ return dof_bin(dof); }, &threads);
			threads.wait_idle();

			for (int i=0; i<8; ++i) {
				threads.submit( [](std::span<DOF_t> list){ std::sort(list.begin(), list.end()); }, local_sorter.get_bin(i));
			}
			threads.wait_idle();

			return local_sorter;
		}

		[[nodiscard]] gutil::BinSort<DOF_t> sort_dofs_singlethread(std::span<DOF_t> dofs) const noexcept {
			gutil::BinSort<DOF_t> local_sorter(dofs, 8);
			local_sorter.sort([&](DOF_t dof){ return dof_bin(dof); });
			for (int i=0; i<8; ++i) {
				std::span<DOF_t> list = local_sorter.get_bin(i);
				std::sort(list.begin(), list.end());
			}
			return local_sorter;
		}

		[[nodiscard]] bool is_sorted() const noexcept {
			//TODO: make a bool to track the status of sorting. similar for the mesh being up to date.
			GUTIL_ASSERT(active_dofs.size()>0 && "DofHandler - did you forget dofhandler.collect_dofs()?");
			return active_dof_sorter.bin_end(7) == active_dofs.size();
		}


		void collect_dofs() noexcept {
			BASE::collect_active_keys<DOF_t>();
			active_dofs = std::span<DOF_t>{reinterpret_cast<DOF_t*>(BASE::active_keys.data()), BASE::active_keys.size()};
			GUTIL_LOG("BASE ", BASE::active_keys.size());
			GUTIL_LOG("DOF ", active_dofs.size());
			// const size_t n_threads = threads.n_threads();
			// const size_t dofs_per_thread = n_threads==0 ? TOTAL_POSSIBLE_DOFS : TOTAL_POSSIBLE_DOFS/n_threads;
			
			// std::vector<std::vector<DOF_t>> thread_dofs(threads.n_threads());
			// auto job = [&](size_t tid){
			// 	const size_t start = tid*dofs_per_thread;
			// 	const size_t end = (tid==threads.n_threads()-1) ? TOTAL_POSSIBLE_DOFS : start + dofs_per_thread;

			// 	for (size_t idx=start; idx<end; ++idx) {
			// 		GUTIL_ASSERT(DOF_t::MakeFromIndex(idx).is_valid());
			// 		if (is_active(idx)) { thread_dofs[tid].push_back(DOF_t::MakeFromIndex(idx)); }
			// 	}
			// };

			// for (size_t tid=0; tid<threads.n_threads(); ++tid) {
			// 	threads.submit(job, tid);
			// }
			// threads.wait_idle();

			// active_dofs.clear();
			// for (auto& list : thread_dofs) {
			// 	active_dofs.insert(active_dofs.end(), std::make_move_iterator(list.begin()), std::make_move_iterator(list.end()));
			// }

			sort_dofs();
		}


		/// mark the start of a wide (look at lots of dofs) process
		/// set_batch_start() sets the BATCH_PROCESS_BIT to true on all active dof masks
		void set_batch_start() const noexcept {
			GUTIL_TIMER("set_batch_start");
			BASE::conditional_bitwise_or_all_masks(ACTIVE_BIT,BATCH_PROCESS_BIT);
			batch_is_set = true;
		}

		/// mark the end of a wide (look at lots of dofs) process
		/// set_batch_end() sets the BATCH_PROCESS_BIT to false on all dof masks
		void set_batch_end() const noexcept {
			GUTIL_TIMER("set_batch_end");
			BASE::unconditional_bitwise_and_all_masks(~BATCH_PROCESS_BIT);
			batch_is_set = false;
		}

		// void deactivate_stranded_dofs() noexcept {
		// 	//if the mesh does not honor some element activation request,
		// 	//it is possible that some active dofs have no active element.
		// 	//this will cause any global stiffness matrix to be non-invertible
		// 	//so these dofs should be de-activated
		// 	//this can happen when the mesh only activates elements that fall within
		// 	//some specified geometry.
		// 	GUTIL_OMP(parallel for)
		// 	for (size_t i=0; i<active_dofs.size(); ++i) {
		// 		bool has_active_support = false;
		// 		const uint8_t dd = active_dofs[i].depth_u8();
		// 		for (DofElem_t el : active_dofs[i].support()) {
		// 			//note features being periodic does not change their linear index
		// 			if (mesh.read_depth(el) >= dd) {
		// 				has_active_support=true;
		// 				break;
		// 			}
		// 		}
		// 		if (!has_active_support) {deactivate(i);}
		// 	}
		// }


		/////////////////////////////////////////////////////////////////////////
		/// Queries. Most of these need to be callable from a quadrature loop over
		/// the mesh elements.
		/////////////////////////////////////////////////////////////////////////
		template<typename Elem_t> requires(std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t> )
		void get_active_dofs_conformal(Elem_t el_, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			GUTIL_ASSERT(dofs.size()==global_idx.size());
			//assume that dofs only exist at the features of active elements
			//and that all such features correspond to a dof
			//this is the same as getting the "basis_s" dofs in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			size_t start_size = dofs.size();

			for (DofFeature_t feat : features(el)) {
				const DOF_t dof{feat};
				if (dof.exists() && is_active(dof)) {
					dofs.push_back(dof);
				}
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin()+start_size, dofs.end()}, 
						std::span<uint64_t>{global_idx.begin()+start_size, global_idx.end()});
		}

		template<typename Elem_t> requires(std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t> )
		void get_active_dofs_quasi_hierarchical(Elem_t el_, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			GUTIL_ASSERT(dofs.size()==global_idx.size());
			//assume that dofs only exist at the features of active elements or the feature of a parent of an active element
			//additionally, if a feature at depth d is active, then its parent feature at depth d-1 cannot be active
			//this is the same as getting "basis_s U basis_a" in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			size_t start_size = dofs.size();

			for (int i=0; i<2 && el.exists(); ++i) {
				for (DofFeature_t feat : features(el)) {
					const DOF_t dof{feat};
					if (dof.exists() && is_active(dof)) {
						dofs.push_back(dof);
					}
				}
				el = el.parent();
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin()+start_size, dofs.end()}, 
						std::span<uint64_t>{global_idx.begin()+start_size, global_idx.end()});
		}

		template<typename Elem_t> requires(std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t> )
		void get_active_dofs_full_hierarchical(Elem_t el_, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			GUTIL_ASSERT(dofs.size()==global_idx.size());
			//assume that dofs only exist at the features of active elements or the feature of any ancestor of an active element
			//additionally, if a feature at depth d is active, then its parent feature at depth d-1 cannot be active
			//this is the same as getting "basis_s U basis_a" in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			size_t start_size = dofs.size();

			while (el.exists()) {
				for (DofFeature_t feat : features(el)) {
					const DOF_t dof{feat};
					if (dof.exists() && is_active(dof)) {
						dofs.push_back(dof);
					}
				}
				el = el.parent();
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin()+start_size, dofs.end()}, 
						std::span<uint64_t>{global_idx.begin()+start_size, global_idx.end()});
		}

		template<typename T=double, typename Action>
		void gather_hierarchical_dofs(Action&& action, DofElem_t el, T* x, T* y, T* z, uint32_t N) const noexcept {
			GUTIL_ASSERT(active_dofs.size()>0);
			GUTIL_ASSERT(x && y && z && N>0);
			GUTIL_ASSERT(el.is_valid());
			
			constexpr bool ACTION_NEEDS_NO_INDEX     = std::is_invocable_r_v<void, Action, DOF_t, uint8_t, T, T, T>;
			constexpr bool ACTION_NEEDS_GLOBAL_INDEX = std::is_invocable_r_v<void, Action, DOF_t, uint8_t, T, T, T, uint64_t>;
			
			static_assert(ACTION_NEEDS_NO_INDEX ^ ACTION_NEEDS_GLOBAL_INDEX, 
				"the action must have the signature void(DOF_t,uint8_t,T,T,T,uint64_t) or void(DOF_t,uint8_t,T,T,T)");

			//given an element and N reference coordinates in that element,
			//perform action(dof, local_n, x[i], y[i], z[i]) for i=0:N-1
			//for each active dof whose support overlaps the specified element at a coarser (lower) depth
			//to capture all dofs, el should be active in the mesh (cast to DofElem_t first). local_n is the local dof number
			//on the support element that overlaps the provided el.
			//if x,y,z corresponds to a vertex, use the vertex method below instead.

			//optionally, the global index of the dof can be supplied as the last argument to the action.
			//if the local index (i.e. the i in x[i]) is needed, it should be tracked as a captured variable
			//in the lambda. For example
			// uint32_t k=0; auto action = [&,N](dof, n, x, y, z) {uint32_t i=(k++)%N; ...}

			while (el.exists()) {
				//handle current depth
				std::array<DOF_t, DOF_t::N_DOF_PER_ELEM> basis;
				DOF_t::dofs_on_elem_simd(el.key, reinterpret_cast<uint64_t*>(&basis[0]));
				for (DOF_t dof : basis) {
					if (dof.exists() && is_active(dof)) {
						[[maybe_unused]] uint64_t global_n;
						if constexpr (ACTION_NEEDS_GLOBAL_INDEX) {
							global_n = global_number(dof);
							GUTIL_ASSERT(global_n<active_dofs.size() && active_dofs[global_n]==dof);
						}

						const uint8_t local = dof.local_dof_number(el);
						GUTIL_SIMD()
						for (uint32_t i=0; i<N; ++i) {
							if constexpr (ACTION_NEEDS_GLOBAL_INDEX) {
								action(dof, local, x[i], y[i], z[i], global_n);
							}
							else {
								action(dof, local, x[i], y[i], z[i]);
							}
						}
					}
				}

				//project up
				DofElem_t parent = el.parent();
				if (parent.exists()) {
					//note the last bits of the i/j/k index encode which child
					//this element is. low bits are the low side of the axis and high bits are the high side
					//the pairity bits are arranged as 0bkji.
					const uint64_t bits = el.kji_pairity_simd();	//also the child number
					const T sx = (bits&0b001) ? T{0.5} : T{-0.5};
					const T sy = (bits&0b010) ? T{0.5} : T{-0.5};
					const T sz = (bits&0b100) ? T{0.5} : T{-0.5};
					GUTIL_SIMD()
					for (uint32_t i=0; i<N; ++i) {
						x[i] = T{0.5}*x[i] + sx;
						y[i] = T{0.5}*y[i] + sy;
						z[i] = T{0.5}*z[i] + sz;
					}
				}
				el = parent;
			}
		}


		template<typename T=double, typename Action, typename Vert_t> requires(std::same_as<Vert_t,MeshVert_t> || std::same_as<Vert_t,DofVert_t>)
		void gather_hierarchical_dofs_at_vertex(Action&& action, Vert_t vtx) const noexcept {
			GUTIL_ASSERT(active_dofs.size()>0);

			//this method calls action(dof,local,x,y,z) for each active dof whose
			//support contains vtx as an interior point (i.e., if the shape function is 0 at vtx, that dof is skipped)
			//local is the local dof number of the vtx projected into the support element of the dof and
			//x,y,z are the reference coordinates of vtx in that support element.
			//optionally, the action signature can accept the global dof number as it's last argument.

			constexpr bool ACTION_NEEDS_NO_INDEX     = std::is_invocable_r_v<void, Action, DOF_t, uint8_t, T, T, T>;
			constexpr bool ACTION_NEEDS_GLOBAL_INDEX = std::is_invocable_r_v<void, Action, DOF_t, uint8_t, T, T, T, uint64_t>;
			static_assert(ACTION_NEEDS_NO_INDEX ^ ACTION_NEEDS_GLOBAL_INDEX, 
				"the action must have the signature void(DOF_t,uint8_t,T,T,T,uint64_t) or void(DOF_t,uint8_t,T,T,T)");

			//project down to the deepest possible representation, so .elements() sees
			//every element that could possibly be active nearby, regardless of local refinement
			DofVert_t dv = static_cast<DofVert_t>(vtx);
			while (dv.depth() < Mesh_t::MAX_DEPTH) { dv = dv.child(); }

			std::array<DofElem_t,8> elems;
			dv.elements_simd(&elems[0]);

			
			//traverse back up and search all (up to 8) elements that the vertex
			//is adjacent to for dofs. Deduplicate the dofs at each depth.

			// per-element bookkeeping, computed once up front (unaffected by depth)
			std::array<uint64_t,8> eff_i{}, eff_j{}, eff_k{};
			std::array<bool,8> elem_exists{};
			for (uint8_t e=0; e<8; ++e) {
				elem_exists[e] = elems[e].exists();
				if (!elem_exists[e]) { continue; }
				const uint8_t local0 = dv.local_vertex_number_simd(elems[e].key);
				GUTIL_ASSERT(!elems[e].is_encoded());
				eff_i[e] = elems[e].i_simd() + (local0&1);
				eff_j[e] = elems[e].j_simd() + ((local0>>1)&1);
				eff_k[e] = elems[e].k_simd() + ((local0>>2)&1);
			}

			std::vector<uint64_t> seen;	//running tally of which dofs have been acted on at each depth

			for (uint8_t level=0; level<=Mesh_t::MAX_DEPTH; ++level) {
				seen.clear();
				for (uint8_t e=0; e<8; ++e) {
					if (!elem_exists[e] || !elems[e].exists()) { continue; }

					DofElem_t el = elems[e];
					GUTIL_ASSERT(!el.is_encoded());

					const uint64_t delta = Mesh_t::MAX_DEPTH - el.depth();
					const uint64_t span  = uint64_t{1} << delta;

					GUTIL_ASSERT(eff_i[e]>=span*el.i_simd());
					GUTIL_ASSERT(eff_j[e]>=span*el.j_simd());
					GUTIL_ASSERT(eff_k[e]>=span*el.k_simd());
					const T x = T{2}*T(eff_i[e] - el.i_simd()*span)/T(span) - T{1};
					const T y = T{2}*T(eff_j[e] - el.j_simd()*span)/T(span) - T{1};
					const T z = T{2}*T(eff_k[e] - el.k_simd()*span)/T(span) - T{1};

					std::array<DOF_t, DOF_t::N_DOF_PER_ELEM> basis;
					DOF_t::dofs_on_elem_simd(el.key, reinterpret_cast<uint64_t*>(&basis[0]));
					for (DOF_t dof : basis) {
						if (!dof.exists() || !is_active(dof)) { continue; }
						uint64_t key = static_cast<uint64_t>(dof);
						if (std::find(seen.begin(), seen.end(), key) != seen.end()) { continue; }
						seen.push_back(key);

						[[maybe_unused]] uint64_t global_n;
						if constexpr (ACTION_NEEDS_GLOBAL_INDEX) {
							global_n = global_number(dof);
							if (active_dofs[global_n]!=dof){
								
								GUTIL_ERROR(dof, " is marked as active. global number ", global_n, " was found, but active_dofs[",global_n,"]= ", active_dofs[global_n]);

								auto it = std::find(active_dofs.begin(), active_dofs.end(), dof);
								if (it!=active_dofs.end()) {
									global_n = std::distance(active_dofs.begin(), it);
									GUTIL_ERROR("std::find found index ", global_n);
								}
								else {
									GUTIL_ERROR("std::find did not find the dof");
								}

							}

							GUTIL_ASSERT(active_dofs[global_n]==dof);
						}
						const uint8_t local = dof.local_dof_number(el);
						if constexpr (ACTION_NEEDS_GLOBAL_INDEX) { action(dof, local, x, y, z, global_n); }
						else                                     { action(dof, local, x, y, z); }
					}
					elems[e] = el.parent();
				}
			}
		}




		/////////////////////////////////////////////////////////////////////////
		/// Refinement queries and operations
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool mesh_can_support_any(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());

			//true if ANY support element of the dof has an intersection with an active mesh element
			//the mesh depth field marks the deepest level of refinement at or below the given element
			const uint8_t depth = dof.depth_u8();
			for (DofElem_t d_el : dof.support()) {
				if (!d_el.exists()) {continue;}
				MeshElem_t m_el = static_cast<MeshElem_t>(d_el);
				if (m_el.exists() && mesh.read_depth(m_el)>=depth) {return true;}
			}
			return false;
		}

		[[nodiscard]] bool mesh_can_support_all(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());

			//true if ALL support element of the dof has an intersection with an active mesh element
			//the mesh depth field marks the deepest level of refinement at or below the given element
			const uint8_t depth = dof.depth_u8();
			for (DofElem_t d_el : dof.support()) {
				if (!d_el.exists()) {continue;}
				MeshElem_t m_el = static_cast<MeshElem_t>(d_el);
				if (m_el.exists() && mesh.read_depth(m_el)<depth) {return false;}
			}
			return true;
		}


		[[nodiscard]] bool is_refinable(DOF_t dof) const noexcept {
			//	From 'Natural hierarchical refinement for finite element methods'
			//	in International J. for Numerical Methods in Engineering (2003, DOI 10.1002/nme.601)
			//
			// 	There are 3 rules to guide refinement:
			//
			//	1) The refining/unrefining of a dof at depth dd may activate or deactivate that dof
			//			or any of its children at depth dd+1.
			//	2) A dof on level dd+1>0 may be refined only when all its parents on level dd have
			//			been refined. (We track this by the INITIAL_DOF_BIT. here dd=0 is the root element)
			//	3) A dof on level dd may be unrefined only if a) it was previously refined and
			//			b) all its children on level dd+1 are not refined.


			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(is_active(dof));

			if (dof.depth() >= MAX_DEPTH) {return false;}		//we can't refine past max depth
			
			uint8_t byte = get_mask(dof);
			if (byte&REFINED_BIT) {return false;}				//we can't refine a dof twice
			if (byte&INITIAL_DOF_BIT) {return true;}			//an unrefined initial dof not at max depth can be refined
			
			for (const DOF_t p : dof.parents()) {
				if (p.exists() && mesh_can_support_any(p)) {	//if the mesh does not capture the entire [0,1]^3 space, don't consider irrelevant parents
					if (!is_refined(p)) {return false;}			//a dof can't be refined until ALL of its parents are refined
				}
			}
			return true;
		}

		[[nodiscard]] bool is_unrefinable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			
			if (dof.depth()>=MAX_DEPTH) {return false;}			//we call unrefine on the parent. at max depth it can't be a parent.
			const uint8_t byte = get_mask(dof);
			if (byte&INITIAL_DOF_BIT) {return false;}			//we can't unrefine to be coarser than the initial dofs
			if (!(bool)(byte&REFINED_BIT)) {return false;}		//we can't unrefine a dof that hasn't been previously refined

			for (const DOF_t c : dof.children()) {				//we can't unrefine a dof if it has valid children that have been refined
				if (c.exists() && mesh_can_support_any(c)) {
					if (is_refined(c)) {return false;}
				}
			}

			return true;
		}

		[[nodiscard]] bool can_deactivate(DOF_t dof) const noexcept {
			for (DOF_t p : dof.parents()) {						//rather than making a recursive unrefine, we check if child dofs can be deactivated
				if (p.exists() && is_refined(p)) {return false;}
			}
			return true;
		}

		void activate(DOF_t dof) noexcept {
			if (!set_active(dof,true)) {						//return if the dof was already active
				return;
			}
																//when refining, it is essential to have the mesh be able to resolve the support
			const uint8_t depth = dof.depth_u8();
			if (depth==0) { return; }							//at depth 0, there is nothing to do
			
			//	Request the mesh to resolve the support. Note that
			//	the mesh may deny the request of any support element,
			//	but it should never deny the request for all elements.
			//	If this is the case, there is probably a bad predicate
			//	in the mesh refinement. The mesh depth field marks 
			//	the deepest level of refinement at or below the given element
			//	so that we do not have to traverse the entire mesh hierarchy
			for (DofElem_t spt : dof.support()) {			
				if (!spt.exists()) {continue;}
				MeshElem_t el = static_cast<MeshElem_t>(spt);
				if (el.exists() && mesh.read_depth(el) < depth) {
					GUTIL_ASSERT(mesh.is_active(el.parent()))		//the mesh should be respecting a 2-1 refinement rule
					mesh.refine(el.parent());						//this is a request. pushes the element to a mutable list. it is protected by a mutex.
				}
				
				#ifndef NDEBUG
				else if (!batch_is_set) {							//if no elements were requested to be active, then the dof must already
					DofFeature_t d_feat{dof.key};					//be at a geometrically conformal feature.
					if (!mesh.is_geometrically_conformal(d_feat)) {	//the request takes into account any periodic axes
						GUTIL_ERROR(d_feat, " is not conformal in the mesh. ", DOF_t{d_feat}, " should have requested mesh refinement");
						// std::terminate();
					}
				}
				#endif
			}
		}

		void deactivate(DOF_t dof) noexcept {						//when un-refining, it is not essential to have the mesh un-refine as well.
			set_active(dof,false); 									//mesh unrefinement should be done in some cleanup pass so that
		}															//multiple dofhandlers can be organized

		void refine_quasi_hierarchical(DOF_t dof) noexcept {
			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(is_active(dof));
			
			if(!is_refinable(dof)) {return;};
			deactivate(dof);
			for (DOF_t c : dof.children()) {
				if (c.exists()) { activate(c); }
			}

			uint8_t& byte = get_mask_ref(dof);
			byte|=(ACTIVE_BIT|REFINED_BIT);							//set active and refined
			byte&=~BATCH_PROCESS_BIT;								//remove from current batch
			GUTIL_ASSERT(is_refined(dof));
			GUTIL_ASSERT(is_unrefinable(dof) || (get_mask(dof)&INITIAL_DOF_BIT));
		}

		void refine_hierarchical(DOF_t dof) noexcept requires(VERTEX_DOF) {
			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(is_active(dof));
			if(!is_refinable(dof)) {return;};
			const MeshFeature_t p_feat = static_cast<MeshFeature_t>(dof);
			for (DOF_t c : dof.children()) {
				if (c.exists() && static_cast<MeshFeature_t>(c).parent() != p_feat) {
					activate(c);
				}
			}

			uint8_t& byte = get_mask_ref(dof);
			byte|=REFINED_BIT;										//set refined
			byte&=~BATCH_PROCESS_BIT;								//remove from current batch
			GUTIL_ASSERT(is_refined(dof));
			GUTIL_ASSERT(is_unrefinable(dof) || (get_mask(dof)&INITIAL_DOF_BIT));
		}

		void unrefine_quasi_hierarchical(DOF_t dof) noexcept {
			GUTIL_ASSERT(dof.is_valid());
			uint8_t& byte = get_mask_ref(dof);

			if (byte&ACTIVE_BIT) { return; }						//in a Q-H scheme, the parent dof must not be active
			if(!is_unrefinable(dof)) {return;};
			
			activate(dof);											//processes mesh refinement request
			for (DOF_t c : dof.children()) {
				if (c.exists()) {
					GUTIL_ASSERT(!is_refined(c));
					if (can_deactivate(c)) {
						set_active(c, false);
					}
				}
			}

			byte|=ACTIVE_BIT;										//mark as active
			byte&=~REFINED_BIT;										//mark as not refined
			byte&=~BATCH_PROCESS_BIT;								//remove from current batch

			#ifndef NDEBUG
				bool success  = true;
				if (!is_active(dof)) {
					success = false;
					GUTIL_ERROR(dof, " should have been activated");
				}
				if (is_refined(dof)) {
					success = false;
					GUTIL_ERROR(dof, " should have been unrefined");
				}
				if (!success) {
					GUTIL_ERROR(dof, " has mask ", print_bytes(byte));
				}
				GUTIL_ASSERT(success);
			#endif
		}

		void unrefine_hierarchical(DOF_t dof) noexcept {
			GUTIL_ASSERT(is_active(dof));
			//in a quasi-hierarchical scheme, the parent dof (this) stays active
			if(!is_active(dof) || !is_unrefinable(dof)) {return;};
			for (DOF_t c : dof.children()) {
				if (c.exists()) {set_active(c, false);}
			}

			uint8_t& byte = get_mask_ref(dof);
			byte&=~REFINED_BIT;										//mark as not refined
			byte&=~BATCH_PROCESS_BIT;								//remove from current batch
			GUTIL_ASSERT(is_unrefinable(dof));
		}


		//////////////////////////////////////////////////////////////////////////////////////
		/// Utility refine/unrefine commands
		//////////////////////////////////////////////////////////////////////////////////////
		void refine_hierarchical(DofElem_t el) noexcept {
			GUTIL_ASSERT(batch_is_set && "call set_batch_start() for synchronization");
			std::vector<DOF_t> dofs;
			std::vector<uint64_t> ids;
			get_active_dofs_full_hierarchical(el, dofs, ids);

			for (DOF_t dof : dofs) {
				if(dof.exists() && (get_mask(dof)&BATCH_PROCESS_BIT)) {refine_hierarchical(dof);}
			}
		}

		void refine_quasi_hierarchical(DofElem_t el) noexcept {
			GUTIL_ASSERT(batch_is_set && "call set_batch_start() for synchronization");
			std::vector<DOF_t> dofs;
			std::vector<uint64_t> ids;
			get_active_dofs_quasi_hierarchical(el, dofs, ids);

			for (DOF_t dof : dofs) {
				if(dof.exists() && (get_mask(dof)&BATCH_PROCESS_BIT)) {refine_quasi_hierarchical(dof);}
			}
		}

		void unrefine_hierarchical(DofElem_t el) noexcept {
			GUTIL_ASSERT(batch_is_set && "call set_batch_start() for synchronization");
			std::vector<DOF_t> dofs;
			std::vector<uint64_t> ids;
			get_active_dofs_full_hierarchical(el, dofs, ids);

			for (DOF_t dof : dofs) {
				if(dof.exists() && (get_mask(dof)&BATCH_PROCESS_BIT)) {unrefine_hierarchical(dof);}
			}
		}

		void unrefine_quasi_hierarchical(DofElem_t el) noexcept {
			GUTIL_ASSERT(batch_is_set && "call set_batch_start() for synchronization");
			std::vector<DOF_t> dofs;
			std::vector<uint64_t> ids;
			get_active_dofs_quasi_hierarchical(el, dofs, ids);
			for (DOF_t dof : dofs) {
				if(dof.exists() && (get_mask(dof)&BATCH_PROCESS_BIT)) {unrefine_quasi_hierarchical(dof);}
			}
		}

	};
















}