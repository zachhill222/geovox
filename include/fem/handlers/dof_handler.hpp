#pragma once

#include "gutil.hpp"

#include "simd_keys/simd_keys.hpp"
#include "util/util.hpp"

#include <concepts>
#include <cstdint>
#include <vector>
#include <span>
#include <type_traits>
#include <algorithm>
#include <thread>


namespace GV {


	/////////////////////////////////////////////////////////////////////////////
	/// Concepts to check compatibility of the mesh and DOF types
	/////////////////////////////////////////////////////////////////////////////
	template<typename Mesh_t>
	concept VoxelMeshType = requires(const Mesh_t& mesh, 
		typename Mesh_t::Elem_t el, 
		typename Mesh_t::Vert_t vtx) {

		//necessary aliases
		typename Mesh_t::Elem_t;
		typename Mesh_t::Face_t;
		typename Mesh_t::Edge_t;
		typename Mesh_t::Vert_t;
		typename Mesh_t::GeoPoint_t;

		//necessary constants
		{ mesh.max_depth } 			-> std::convertible_to<uint8_t>;

		//necessary queries
		{ mesh.is_active(el) }		-> std::same_as<bool>;
		{ mesh.is_conformal(vtx) }	-> std::same_as<bool>;
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
	/// We assume that the dof handler is for some sort of hierarchical scheme.
	/// We have a struct of supported types that can be useful for compile-time
	/// choices of functions.
	/////////////////////////////////////////////////////////////////////////////
	struct DofHierarchicalVariants {
		static constexpr uint64_t NONE{0};
		
		static constexpr uint64_t Conformal			= 0b001;
		static constexpr uint64_t QuasiHierarchical = 0b010;
		static constexpr uint64_t TrueHierarchical  = 0b100;
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
		static constexpr uint64_t HFlag = DofHierarchicalVariants::NONE;	//overwrite in a derived class as needed

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
		using MeshElem_t    = typename Mesh_t::Elem_t;
		using MeshVert_t    = typename Mesh_t::Vert_t;
		using MeshFeature_t = std::conditional_t<VERTEX_DOF, MeshVert_t, std::conditional_t<ELEMENT_DOF, MeshElem_t, void>>;
		static_assert(!std::same_as<MeshFeature_t,void>);

		static constexpr bool     IS_DEPTH_SEPARABLE  = DepthSeparableVoxelMeshType<Mesh_t>;
		
		//The maximum depth of a given mesh is specified at runtime. However, to avoid accidentally
		//requesting say depth 12 (2^(3*12) elements at depth 12, (2^33 -1)/7 ~ 10^10.3 total elements)
		//we set a maximum depth at compile time.
		const  			 uint8_t  max_depth;
		const 			 uint64_t max_possible_dofs;

		//For quasi-hierarchical refinement, there is generally a difference of at most 2 between an active element
		//and a dof. However, if multiple handlers are interested in the same mesh (e.g., Q1-iso-Q2 elements in a Stokes system),
		//we may specify this bound. It is up to the user to ensure that the bound is satisfied.
		uint8_t max_depth_distance{max_depth};

		using BASE::ACTIVE_BIT; 						//0b00000001;	
		static constexpr uint8_t REFINED_BIT 			= 0b00000010;
		static constexpr uint8_t INITIAL_DOF_BIT        = 0b00000100;
		
		//For hierarchical methods, we can use some of the free bits to store snapshots of the hierarchy.
		//snapshot 0 is the current, snapshot 1/A and snapshot 2/B are independent. snapshots 1 and 2 can be
		//set via set_snapshot(uint8_t i which) and the current can be restored from a snapshot (swaps
		//the current state with the snapshot state).
		static constexpr uint8_t SNAP_A_ACTIVE_BIT		= 0b00001000;
		static constexpr uint8_t SNAP_A_REFINED_BIT		= 0b00010000;
		static constexpr uint8_t SNAP_B_ACTIVE_BIT		= 0b00100000;
		static constexpr uint8_t SNAP_B_REFINED_BIT		= 0b01000000;

		//unused bits
		static constexpr uint8_t FREE_BITS 	            = 0b10000000;


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
		using BASE::threads;							//max hardware concurency by default
		
		public:
		using BASE::is_key_mask_unstable;
		using BASE::is_key_mask_stable;
		using BASE::begin_key_mask_stable;
		using BASE::end_key_mask_stable;
		using BASE::begin_key_mask_unstable;
		using BASE::end_key_mask_unstable;
		
		using BASE::is_active_keys_unstable;
		using BASE::is_active_keys_stable;
		using BASE::begin_active_keys_stable;
		using BASE::end_active_keys_stable;
		using BASE::begin_active_keys_unstable;
		using BASE::end_active_keys_unstable;

		using BASE::sort_active_keys;

		std::span<DOF_t>	active_dofs;
		const Mesh_t& 		mesh;					//link to the mesh, we can request refinement through const methods
		
		[[nodiscard]] bool is_current() const noexcept {
			return BASE::is_current() && are_spans_same_data(active_dofs, BASE::active_keys);
		}

		/////////////////////////////////////////////////////////////////////////
		/// Constructors. The dofhandler must be linked to the mesh at construction
		/// and the mesh must outlive the dofhandler.
		/////////////////////////////////////////////////////////////////////////
		DofHandler() = delete;
		DofHandler(const Mesh_t& m, size_t n_threads = std::thread::hardware_concurrency()) : 
				BASE(DOF_t::total_possible(m.max_depth), n_threads),
				max_depth(m.max_depth),
				max_possible_dofs{DOF_t::total_possible(m.max_depth)},
				mesh(m) {
					if (n_threads > std::thread::hardware_concurrency()) {
						GUTIL_ERROR("Requested pool size (", n_threads, ") exceeds what is seen by std::thread::hardware_concurrency() (",
								std::thread::hardware_concurrency(), ")");
					}
					GUTIL_ASSERT(max_depth == m.max_depth);
					GUTIL_ASSERT(max_depth<=BASE::KEY_MAX_DEPTH);
				}
				
		
		DofHandler(const DofHandler& other) : BASE(other), mesh(other.mesh) {
			active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(BASE::active_keys);
		}
		DofHandler(DofHandler&& other) : BASE(std::move(other)), mesh(other.mesh) {
			active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(BASE::active_keys);
		}

		[[nodiscard]] DofHandler& operator=(const DofHandler& other) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				key_mask = other.key_mask;
				BASE::active_keys = other.active_keys;
				active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(BASE::active_keys);
			}
			return *this;
		}
		[[nodiscard]] DofHandler& operator=(DofHandler&& other) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				key_mask = std::move(other.key_mask);
				BASE::active_keys = std::move(other.active_keys);
				active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(BASE::active_keys);
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

		//for vertex dofs, sometimes getting the correct depth is awkward.
		//determine if any (there is at most one) dof lives at the same geometric location
		//as the requested vertex
		template<typename Vert_t> requires(std::same_as<Vert_t,MeshVert_t> || std::same_as<Vert_t,DofVert_t>)
		[[nodiscard]] constexpr DofFeature_t get_dof_vertex(Vert_t vtx_) noexcept requires(VERTEX_DOF) {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			DofVert_t vtx = static_cast<DofVert_t>(vtx_);
			if (!DOF_t{vtx}.exists()) {return DofFeature_t::None();} //not a valid feature for periodic dofs
			//traverse to shallowest mesh key, then check going down
			while (vtx.parent().exists()) { vtx = vtx.parent(); }
			for (uint64_t dd=vtx.depth(); dd<=mesh.max_depth; ++dd) {
				//note that the linear index of any feature is determined by its non-periodic location
				//for periodic features, the cononical choice is the feature with smallest indices.
				if (BASE::is_active_stable(vtx.linear_index())) {return vtx;}
				else if (vtx.child().exists()) {vtx = vtx.child();}
				else {break;}
			}
			return DofFeature_t::None();
		}


		/////////////////////////////////////////////////////////////////////////
		/// Adapt masking function to DOFs and this classes's masks
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] uint8_t get_mask(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::get_mask(dof.linear_index());
		}
		[[nodiscard]] uint8_t& get_mask_ref(DOF_t dof) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::get_mask_ref(dof.linear_index());
		}
		[[nodiscard]] uint8_t get_mask_stable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::get_mask_stable(dof.linear_index());
		}
		[[nodiscard]] uint8_t get_mask_unstable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::get_mask_unstable(dof.linear_index());
		}
		[[nodiscard]] uint8_t& get_mask_ref_pseudo_const(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::get_mask_ref_pseudo_const(dof.linear_index());
		}


		[[nodiscard]] bool is_active(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::is_active(dof.linear_index());
		}
		[[nodiscard]] bool is_active_no_check(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::is_active_no_check(dof.linear_index());
		}
		[[nodiscard]] bool is_active_stable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::is_active_stable(dof.linear_index());
		}
		[[nodiscard]] bool is_active_unstable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::is_active_unstable(dof.linear_index());
		}
		[[nodiscard]] bool set_active_check_changed(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_active_check_changed(dof.linear_index(), val);
		}
		void set_active(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_active(dof.linear_index(), val);
		}

		
		[[nodiscard]] bool is_refined(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_refined_no_check(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_no_check<REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_refined_stable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_stable<REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_refined_unstable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_unstable<REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool set_refined_check_changed(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_bit_check_changed<REFINED_BIT>(dof.linear_index(), val);
		}
		void set_refined(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_bit<REFINED_BIT>(dof.linear_index(), val);
		}

		
		[[nodiscard]] bool is_initial_marked(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<INITIAL_DOF_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_initial_marked_stable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_stable<INITIAL_DOF_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_initial_marked_unstable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_unstable<INITIAL_DOF_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool set_initial_marked_check_changed(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_bit_check_changed<INITIAL_DOF_BIT>(dof.linear_index(), val);
		}
		void set_initial_marked(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_bit<INITIAL_DOF_BIT>(dof.linear_index(), val);
		}

		[[nodiscard]] bool is_snapshot_A_active(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<SNAP_A_ACTIVE_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_snapshot_A_refined(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<SNAP_A_REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_snapshot_B_active(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<SNAP_B_ACTIVE_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_snapshot_B_refined(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<SNAP_B_REFINED_BIT>(dof.linear_index());
		}

		/////////////////////////////////////////////////////////////////////////
		/// Methods to set and restore snapshots
		/////////////////////////////////////////////////////////////////////////
		void clear_snapshot(uint8_t which = 3) {
			uint8_t mask{0};
			if (which==3 || which==1) {mask |= (SNAP_A_ACTIVE_BIT | SNAP_A_REFINED_BIT); }
			if (which==3 || which==2) {mask |= (SNAP_B_ACTIVE_BIT | SNAP_B_REFINED_BIT); }
			GV_BEGIN_MASK_UNSTABLE;
			BASE::unconditional_bitwise_and_all_masks(~mask);
			GV_END_MASK_UNSTABLE;
		}

		void set_snapshot(uint8_t which) {
			if (which!=1 && which!=2) {
				GUTIL_ERROR("Only set_snapshot(1) and set_snapshot(2) are supported");
				return;
			}

			const uint8_t s_active = (which==1) ? SNAP_A_ACTIVE_BIT  : SNAP_B_ACTIVE_BIT;
			const uint8_t s_refine = (which==1) ? SNAP_A_REFINED_BIT : SNAP_B_REFINED_BIT;
			auto action = [this, s_active, s_refine](std::span<uint8_t> list) {
				GUTIL_SIMD()
				for (size_t i=0; i<list.size(); ++i) {
					uint8_t& mask = list[i];

					if (mask&ACTIVE_BIT) {mask|=s_active;}
					else {mask&=~s_active;}

					if (mask&REFINED_BIT) {mask|=s_refine;}
					else {mask&=~s_refine;}
				}
			};

			GV_BEGIN_MASK_UNSTABLE;
			BASE::dispatch_parallel_key_mask(action);
			GV_END_MASK_UNSTABLE;
		}

		void restore_snapshot(uint8_t which) {
			if (which!=1 && which!=2) {
				GUTIL_ERROR("Only restore_snapshot(1) and restore_snapshot(2) are supported");
				return;
			}

			//swap the current data with the specified snapshot data.
			//then update the current dofs.
			const uint8_t s_active = (which==1) ? SNAP_A_ACTIVE_BIT  : SNAP_B_ACTIVE_BIT;
			const uint8_t s_refine = (which==1) ? SNAP_A_REFINED_BIT : SNAP_B_REFINED_BIT;
			
			auto swap_active = [s_active] (uint8_t& byte) {
				const bool differ = static_cast<bool>(byte&s_active) != static_cast<bool>(byte&ACTIVE_BIT);
				byte ^= (s_active | ACTIVE_BIT) * static_cast<uint8_t>(differ);
			};

			auto swap_refined = [s_refine] (uint8_t& byte) {
				const bool differ = static_cast<bool>(byte&s_refine) != static_cast<bool>(byte&REFINED_BIT);
				byte ^= (s_refine | REFINED_BIT) * static_cast<uint8_t>(differ);
			};


			auto action = [this, swap_active, swap_refined](std::span<uint8_t> list) {
				GUTIL_SIMD()
				for (size_t i=0; i<list.size(); ++i) {
					uint8_t& mask = list[i];
					swap_active(mask);
					swap_refined(mask);
				}
			};

			{
				GV_BEGIN_MASK_UNSTABLE;
				BASE::dispatch_parallel_key_mask(action);
				GV_END_MASK_UNSTABLE;
			}
			collect_dofs();
		}

		gutil::BinSortVector<DOF_t> collect_snapshot(uint8_t which=0) const noexcept {
			GUTIL_ASSERT(which<3);
			//which=0 -> get current
			//which=1 -> get snapshot 1/A
			//which=2 -> get snapshot 2/B

			if (which==0) {
				GV_BEGIN_STABLE
				//note that active_keys is sorted, but has type uint64_t
				//we must copy the result as DOF_t and then copy the sort status
				gutil::BinSortVector<DOF_t> result(active_dofs.begin(), active_dofs.end());
				result.set_bin_fun([](DOF_t dof){return (int) dof.depth();});
				result.bins = this->sorter.bins;
				result.n_bins_ = this->sorter.n_bins_;
				result.n_bits_ = this->sorter.n_bits_;
				GV_END_STABLE
				return result;
			}
			
			//we need to collect and then sort the dofs.
			uint8_t mask = (which==1) ? SNAP_A_ACTIVE_BIT : SNAP_B_ACTIVE_BIT;

			const size_t n_threads = threads.n_threads()==0 ? 1 : threads.n_threads();
			const size_t n_keys_per_thread = key_mask.size()/n_threads;
			std::vector<std::vector<DOF_t>> thread_keys(n_threads);
			
			auto job = [mask, n_keys_per_thread, &thread_keys](std::span<const uint8_t> masks, size_t tid) {
				size_t key_index = tid*n_keys_per_thread;	//we need to track the location of the mask we are examing
				for (size_t i=0; i<masks.size(); ++i, ++key_index) {
					if (masks[i]&mask) {thread_keys[tid].push_back(DOF_t::MakeFromIndex(key_index));}
				}
			};

			{
				GV_BEGIN_STABLE
				dispatch_parallel_key_mask_const(job);
				threads.wait_idle();
				GV_END_STABLE
			}
			for (size_t tid=1; tid<n_threads; ++tid) {
				thread_keys[0].insert(thread_keys[0].end(),
									std::make_move_iterator(thread_keys[tid].begin()),
									std::make_move_iterator(thread_keys[tid].end()));
			}
			
			gutil::BinSortVector<DOF_t> result(std::move(thread_keys[0]));
			result.set_n_bins( (int) max_depth);

			result.set_threadpool(threads);
			result.sort([](DOF_t dof){return (int) dof.depth();});
			result.sort_bins();
			result.clear_threadpool();
			return result;
		}


		///////////////////////////////////////////////////////////////////////////
		/// Define functions to sort DOFs into bins and to look up global dof numbers
		///////////////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr int dof_key_bin(uint64_t dof_key) noexcept {
			return DOF_t{dof_key}.depth();
		}

		//get a single dof global number
		[[nodiscard]] size_t global_number(DOF_t dof) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return BASE::lookup_key(dof.key, &dof_key_bin);
		}

		[[nodiscard]] DOF_t operator[](size_t idx) const noexcept {
			GUTIL_ASSERT(idx<active_dofs.size());
			return active_dofs[idx];
		}

		//sort the dofs by increasing global index and get their global index
		void get_dof_number_local_sort(std::span<DOF_t> dofs, std::span<uint64_t> global_index) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			BASE::lookup_keys_and_sort(
				BASE::reinterpret_key_span<uint64_t,DOF_t>(dofs), global_index, mesh.max_depth+1, &dof_key_bin);
		}
		

		/////////////////////////////////////////////////////////////////////////
		/// Initialization. The mesh must be in a conformal state.
		/// Additionally, we assume that this initial mesh is not very large.
		/////////////////////////////////////////////////////////////////////////
		void init_dofs() noexcept {
			GUTIL_ASSERT(mesh.is_current() && mesh.is_depth_field_correct());
			{
				GV_BEGIN_UNSTABLE
				BASE::clear();

				//note that all elements have the same encoding. periodicity only affects the methods.
				std::span<const DofElem_t> dof_elements = mesh.template elements_as_span<DOF_t::PERIOD>();
				BASE::active_keys.resize(DOF_t::N_DOF_PER_ELEM * dof_elements.size(), uint64_t(-1));

				GUTIL_SIMD()
				for (size_t i=0; i<dof_elements.size(); ++i) {
					DOF_t::dofs_on_elem_simd_raw(dof_elements[i].decode_simd(), 	//returns the raw key of the element's cartesian form
							&BASE::active_keys[i*DOF_t::N_DOF_PER_ELEM]);
				}

				#ifndef NDEBUG
				GUTIL_ASSERT(std::find(BASE::active_keys.begin(), BASE::active_keys.end(), uint64_t(-1)) == BASE::active_keys.end());
				#endif

				auto it = gutil::sort_and_unique(BASE::active_keys, [](uint64_t a, uint64_t b) {return b<a;});	//large dofs on the left
				BASE::active_keys.erase(it, BASE::active_keys.end());
				std::erase_if(BASE::active_keys, [](uint64_t a){return !DOF_t{a}.is_valid();});			//cleanup any singleton bad dofs
				BASE::active_keys.shrink_to_fit();
				active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(BASE::active_keys);
				BASE::is_collected_.store(true);

				GUTIL_OMP(parallel for)
				for (size_t i=0; i<BASE::active_keys.size(); ++i) {
					set_active(DOF_t{BASE::active_keys[i]}, true);
					set_initial_marked(DOF_t{BASE::active_keys[i]}, true);
				}

				GV_END_UNSTABLE
			}//release mutex (collect dofs needs to re-aquire the unique lock)
			sort_active_keys(mesh.max_depth+1, &DofHandler::dof_key_bin);
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_all_dofs_conformal());
		}

		void clear() noexcept {
			GV_BEGIN_UNSTABLE
			BASE::clear();
			GV_END_UNSTABLE
		}
		
		/////////////////////////////////////////////////////////////////////////
		/// Book keeping methods
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] size_t n_dofs() const noexcept {
			return active_dofs.size();
		}

		void collect_dofs() noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			BASE::collect_active_keys<DOF_t>();
			active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(BASE::active_keys);
			BASE::sort_active_keys(mesh.max_depth+1, &DofHandler::dof_key_bin);
			#ifndef NDEBUG
				if (!mesh.has_pending_refine_requests()) {
					GUTIL_ASSERT(is_all_dofs_conformal());
				}
			#endif
		}

		/// ensure that every active dof is at a conformal vertex in the mesh
		[[nodiscard]] bool is_all_dofs_conformal() const noexcept {
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(mesh.is_current());

			size_t count=0;
			{
				GV_BEGIN_STABLE
				auto m_lock = mesh.begin_key_mask_stable();

				size_t n_threads = threads.n_threads()==0 ? 1 : threads.n_threads();
				std::vector<size_t> thread_count(n_threads);

				BASE::dispatch_parallel_active_keys_const([&](std::span<const uint64_t> keys, int tid) {
					for (size_t i=0; i<keys.size(); ++i) {
						if (!mesh.is_conformal(DofFeature_t{BASE::active_keys[i]})) {
							DOF_t dof{keys[i]};
							GUTIL_ASSERT(dof == DOF_t{keys[i]});
							GUTIL_ERROR("active_dof[",i,"] ", dof, " at ", DofFeature_t{dof.key}, " is non-conformal");
							thread_count[tid]++;
						}
					}
				});

				threads.wait_idle();
				GV_END_STABLE

				for (size_t c : thread_count) {count+=c;}
			}
			
			return count==0;
		}

		
		/////////////////////////////////////////////////////////////////////////
		/// Queries. Most of these need to be callable from a quadrature loop over
		/// the mesh elements.
		/////////////////////////////////////////////////////////////////////////
		template<typename Elem_t> requires(std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t> )
		[[nodiscard]] std::vector<DOF_t> get_active_dofs_conformal(Elem_t el_) const noexcept {
			GV_BEGIN_MASK_STABLE

			GUTIL_ASSERT(is_current());
			//assume that dofs only exist at the features of active elements
			//and that all such features correspond to a dof
			//this is the same as getting the "basis_s" dofs in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			std::vector<DOF_t> result;
			result.reserve(DOF_t::N_DOF_PER_ELEM);

			for (DofFeature_t feat : features(el)) {
				const DOF_t dof{feat};
				if (dof.exists() && is_active(dof)) {
					result.push_back(dof);
				}
			}
			GV_END_MASK_STABLE
			return result;
		}

		template<typename Elem_t, typename Predicate = std::nullptr_t>
			requires(std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t> ) 
					&& (std::same_as<Predicate,std::nullptr_t> || std::is_invocable_r_v<bool, Predicate, DOF_t>)
		[[nodiscard]] std::vector<DOF_t> get_active_dofs(Elem_t el_, Predicate&& pred = nullptr) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GV_ASSERT_KEY_MASK_STABLE_STATE

			//assume that dofs only exist at the features of active elements or the feature of a parent of an active element
			//additionally, if a feature at depth d is active, then its parent feature at depth d-1 cannot be active
			//this is the same as getting "basis_s U basis_a" in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			std::vector<DOF_t> result;
			result.reserve(DOF_t::N_DOF_PER_ELEM);

			for (int i=0; i<=max_depth_distance && el.exists(); ++i) {
				for (DofFeature_t feat : features(el)) {
					const DOF_t dof{feat};
					if constexpr (std::same_as<Predicate,std::nullptr_t>) {
						if (dof.exists() && is_active_stable(dof)) {result.push_back(dof); }
					}
					else {
						if (dof.exists() && pred(dof) && is_active_stable(dof)) {result.push_back(dof); }
					}
				}
				el = el.parent();
			}
			return result;
		}


		// When looking up dofs on many elements (e.g., multithreaded quadrature of a bilinear form over a mesh)
		// we can construct a lookup function per-thread that can be used for finding active dofs over an element.
		// Note that the max_depth_distance guarantee only applies to the current active dofs (which=0) and not snapshots
		// A or B (which=1,2).
		template<typename Elem_t, uint8_t which> requires(which<3 && (std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t>))
		[[nodiscard]] auto make_snapshot_dof_getter(std::vector<DOF_t>& result_vec) const noexcept {
			GUTIL_ASSERT(which<3);

			const uint8_t distance = (which==0) ? max_depth_distance : max_depth;

			return [this, distance, &result_vec](Elem_t el_) noexcept {
				GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
				GV_ASSERT_KEY_MASK_STABLE_STATE

				DofElem_t el = static_cast<DofElem_t>(el_);
				result_vec.clear();

				for (uint8_t i=0; i<=distance && el.exists(); ++i) {
					for (DofFeature_t feat : this->features(el)) {
						const DOF_t dof{feat};
						if (dof.exists()) {
							if constexpr (which==0) {if (is_active_no_check(dof)) {result_vec.push_back(dof);}}
							else if constexpr (which==1) {if (is_snapshot_A_active(dof)) {result_vec.push_back(dof);}}
							else if constexpr (which==2) {if (is_snapshot_A_active(dof)) {result_vec.push_back(dof);}}
						}
					}
					el = el.parent();
				}
			};
		}

		template<typename Elem_t>
		[[nodiscard]] std::function<void(Elem_t)> make_snapshot_dof_getter(uint8_t which, std::vector<DOF_t>& result_vec) const noexcept {
			//a runtime helper function to generate the correct dof getter.
			switch (which) {
				case 0:  return make_snapshot_dof_getter<Elem_t, 0>(result_vec);
				case 1:  return make_snapshot_dof_getter<Elem_t, 1>(result_vec);
				default: return make_snapshot_dof_getter<Elem_t, 2>(result_vec);
			}
		}


		////////////////////////////////////////////////////////////////////////////////////////////
		/// Primary method for bulk gathering dofs.
		///
		/// Given a collection of elements, find all dofs whos support overlaps them and satisfy a predicate. 
		/// Only dofs at a depth at or above the provided element will be considered, so the elements should 
		/// be active mesh elements. If you know the maximum difference in depth between the elements and the dofs
		/// you are looking for, you can pass that as n_depths.
		///
		/// For example, pass a predicate 	auto pred = [ACTIVE_BIT](uint8_t dof_mask) { return dof_mask&ACTIVE_BIT; }
		/// to collect the active dofs. Alternatively, the dof can be passed to the predicate.
		///
		/// The possible signatures are bool(uint8_t), bool(uint8_t,DOF_t), bool(DOF_t,uint8_t), bool(DOF_t).
		///
		/// You may also pass an action to take on the dof_mask.
		/// The action must have the signature void(uint8_t&). To work with the active dof representation,
		/// you will need to look up its global index.
		///
		/// Note that this is a single-threaded SIMD operation. Feel free to use the thread pool to dispatch
		/// several queries (there may be duplicate dofs between queries, which may or may not be what you want).
		/// If you are passing an action, be cautious of race conditions when doing this.
		///
		/// It is not necessary that all input elements are at the same depth.
		///
		/// Depending on the Action, stable/unstable needs to be set by the caller
		//////////////////////////////////////////////////////////////////////////////////////////////
		template<typename Predicate, typename Action=std::nullptr_t>
		[[nodiscard]] std::vector<DOF_t> get_dofs_impl(
				std::span<const DofElem_t> 	elems,
				uint8_t 					n_depths,
				Predicate&& 				pred,
				Action&& 					action=nullptr) noexcept
		{	
			GUTIL_PROFILE_FUNCTION();

			//sanity checks and argument deduction
			constexpr bool PRED_MASK = std::is_invocable_r_v<bool, Predicate, uint8_t>;
			constexpr bool PRED_DOF  = std::is_invocable_r_v<bool, Predicate, DOF_t>;
			constexpr bool PRED_BOTH = std::is_invocable_r_v<bool, Predicate, uint8_t, DOF_t>;
			constexpr bool PRED_REV  = std::is_invocable_r_v<bool, Predicate, DOF_t, uint8_t>;
			static_assert(PRED_MASK + PRED_DOF + PRED_BOTH + PRED_REV, 
									"The predicate argument could not be determined");

			constexpr bool HAS_ACTION = !std::same_as<std::nullptr_t,Action>;
			static_assert(!HAS_ACTION || std::is_invocable_r_v<void,Action,uint8_t&>, 
									"The action must haave the signature void(uint8_t&)");


			n_depths = std::min(n_depths, max_depth);						//make sure n_depths is feasible
			std::vector<DofElem_t> dof_elems{elems.begin(), elems.end()};	//copy the elements so we are free to manipulate them.

			GUTIL_SIMD()													//ensure elements are in cartesian form
			for (size_t i=0; i<dof_elems.size(); ++i) {
				GUTIL_ASSERT(dof_elems[i].exists());
				dof_elems[i].key = DofElem_t::decode_simd(dof_elems[i].key);
			}

			std::vector<DOF_t> result{};									//space for the result
			std::vector<DOF_t> scratch{};									//space for potential results

			for (uint8_t dd=0; dd<n_depths && dof_elems.size()>0; ++dd) {	//gather dofs at each depth	
				std::erase_if(dof_elems, 									//drop any elements that were processed to depth 0
					[](DofElem_t el) {return !el.exists();});

				scratch.resize(dof_elems.size() * DOF_t::N_DOF_PER_ELEM);	//ensure the scratch is perfectly sized (only shrinks)
				
				#ifndef NDEBUG
				//test that we write to every memory location in the scratch buffer
				std::fill(scratch.begin(), scratch.end(), DOF_t{uint64_t(-1)});
				#endif

				GUTIL_SIMD()
				for (size_t i=0; i<dof_elems.size(); ++i) {					//each element fills its buffer of potential dofs
					DOF_t::dofs_on_elem_simd(dof_elems[i].key, 
									&scratch[i*DOF_t::N_DOF_PER_ELEM]);
				}

				GUTIL_ASSERT(std::find(scratch.begin(),scratch.end(), DOF_t{uint64_t(-1)}) == scratch.end());

				auto it = gutil::sort_and_unique(scratch, threads);			
				scratch.erase(it, scratch.end());							//erases all but one non-existant dofs (there could be 1 non-existant element)
				// if (!scratch.back().exists()) {scratch.pop_back();}		//all dofs exist now.
				std::erase_if(scratch, [](DOF_t dof){ return !dof.exists() || !dof.is_valid();});	//TODO: get rid of this

				//gather valid dofs
				result.reserve(result.size() + scratch.size());
				for (DOF_t dof : scratch) {
					GUTIL_ASSERT(dof.exists() && dof.is_valid());

					if constexpr (PRED_DOF) {								//maybe we don't have to fetch the mask
						if (!pred(dof)) {continue;}
						result.push_back(dof);
						if constexpr (HAS_ACTION) {action(get_mask_ref(dof));}
					}
					else {
						uint8_t& byte = get_mask_ref(dof);
						if 		constexpr 	(PRED_MASK) {if (!pred(byte)) 	  {continue;}}
						else if constexpr 	(PRED_BOTH) {if (!pred(byte,dof)) {continue;}}
						else if constexpr 	(PRED_REV) 	{if (!pred(dof,byte)) {continue;}}
						result.push_back(dof);
						if constexpr (HAS_ACTION) {action(byte);}
					}
				}

				//step each element up
				GUTIL_SIMD()
				for (size_t i=0; i<dof_elems.size(); ++i) {
					dof_elems[i] = DofElem_t{dof_elems[i].parent_simd()};
				}
			}

			//final pass to ensure all dofs are unique
			//note that this is required because we do not know that all of the input elements were at the
			//same depth. The dofs are mostly sorted in decreasing order, so keep that order
			std::sort(result.begin(), result.end(), [](DOF_t a, DOF_t b) { return b < a; });
			auto it = std::unique(result.begin(), result.end());
			result.erase(it, result.end());
			return result;
		}


		///////////////////////////////////////////////////////////////////////
		/// A few methods for calling actions on each unique dof in a region
		///////////////////////////////////////////////////////////////////////
		template<typename T=double, typename Action>
		void gather_hierarchical_dofs(Action&& action, DofElem_t el, T* x, T* y, T* z, uint32_t N) const noexcept {
			GV_BEGIN_STABLE

			GUTIL_ASSERT(is_current());
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
					if (dof.exists() && is_active_stable(dof)) {
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

			GV_END_STABLE
		}


		template<typename T=double, typename Action, typename Vert_t> requires(std::same_as<Vert_t,MeshVert_t> || std::same_as<Vert_t,DofVert_t>)
		void gather_hierarchical_dofs_at_vertex(Action&& action, Vert_t vtx) const noexcept {
			GV_BEGIN_STABLE

			GUTIL_ASSERT(is_current());
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
			while (dv.depth() < max_depth) { dv = dv.child(); }

			DofElem_t elems[8];
			dv.elements_simd(elems);

			
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

			for (uint8_t level=0; level<=max_depth; ++level) {
				seen.clear();
				for (uint8_t e=0; e<8; ++e) {
					if (!elem_exists[e] || !elems[e].exists()) { continue; }

					DofElem_t el = elems[e];
					GUTIL_ASSERT(!el.is_encoded());

					const uint64_t delta = max_depth - el.depth();
					const uint64_t span  = uint64_t{1} << delta;

					GUTIL_ASSERT(eff_i[e]>=span*el.i_simd());
					GUTIL_ASSERT(eff_j[e]>=span*el.j_simd());
					GUTIL_ASSERT(eff_k[e]>=span*el.k_simd());
					const T x = T{2}*T(eff_i[e] - el.i_simd()*span)/T(span) - T{1};
					const T y = T{2}*T(eff_j[e] - el.j_simd()*span)/T(span) - T{1};
					const T z = T{2}*T(eff_k[e] - el.k_simd()*span)/T(span) - T{1};

					std::array<DOF_t, DOF_t::N_DOF_PER_ELEM> basis;
					DOF_t::dofs_on_elem_simd(el.key, &basis[0]);
					for (DOF_t dof : basis) {
						if (!dof.exists() || !is_active_stable(dof)) { continue; }
						uint64_t key = static_cast<uint64_t>(dof);
						if (std::find(seen.begin(), seen.end(), key) != seen.end()) { continue; }
						seen.push_back(key);

						[[maybe_unused]] uint64_t global_n;
						if constexpr (ACTION_NEEDS_GLOBAL_INDEX) {
							global_n = global_number(dof);
							if (active_dofs[global_n]!=dof) {
								
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

			GV_END_STABLE
		}


		/////////////////////////////////////////////////////////////////////////
		/// A bare essential activate(dof) method that request the mesh to resolve
		/// the dof support. If the mesh request isn't needed, use set_active(dof,true).
		/////////////////////////////////////////////////////////////////////////
		void activate(DOF_t dof) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());

			uint8_t& byte = get_mask_ref(dof);
			if (byte&ACTIVE_BIT) {return;}
			byte|=ACTIVE_BIT;

			const uint8_t depth = dof.depth_u8();
			if (depth==0) {return;}

			for (DofElem_t spt : dof.support()) {			
				if (!spt.exists()) {continue;}
				MeshElem_t el = static_cast<MeshElem_t>(spt);
				if (mesh.read_depth_field(el) < depth) {
					GUTIL_ASSERT(mesh.is_active(el.parent()));	//the mesh should be respecting a 2-1 refinement rule
					mesh.request_refine(el.parent());			//this is a request. pushes the element to a mutable list. it is protected by a mutex.
				}
			}
		}


		/////////////////////////////////////////////////////////////////////////
		/// Querries to see how the mesh and dof interact. Specifically, if the mesh
		/// overlaps with dof support elements.
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool mesh_can_support_any(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());

			//true if ANY support element of the dof has an intersection with an active mesh element
			//the mesh depth field marks the deepest level of refinement at or below the given element
			const uint8_t depth = dof.depth_u8();
			for (DofElem_t d_el : dof.support()) {
				if (!d_el.exists()) {continue;}
				MeshElem_t m_el = static_cast<MeshElem_t>(d_el);
				if (m_el.exists() && mesh.read_depth_field(m_el)>=depth) {return true;}
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
				if (m_el.exists() && mesh.read_depth_field(m_el)<depth) {return false;}
			}
			return true;
		}

		
		//////////////////////////////////////////////////////////////////////////////////////
		/// For bulk refine/unrefine operations, the above operations won't quite work.
		/// If "can_refine" or "can_unrefine" is true of a batch of elements, then
		/// use these operations sequentially on the entire batch and/or parents/children.
		//////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool has_any_active_child(DOF_t dof) const noexcept {
			for (DOF_t c : dof.children()) {
				if (c.exists() && is_active_no_check(c)) {return true;}
			}
			return false;
		}

		[[nodiscard]] bool has_all_active_children(DOF_t dof) const noexcept {
			for (DOF_t c : dof.children()) {
				if (c.exists() && !is_active_no_check(c) && mesh_can_support_any(c)) {return false;}
			}
			return true;
		}

		[[nodiscard]] bool has_any_refined_child(DOF_t dof) const noexcept {
			for (DOF_t c : dof.children()) {
				if (c.exists() && is_refined_no_check(c)) {return true;}
			}
			return false;
		}

		[[nodiscard]] bool has_all_refined_children(DOF_t dof) const noexcept {
			for (DOF_t c : dof.children()) {
				if (c.exists() && !is_refined_no_check(c) && mesh_can_support_any(c)) {return false;}
			}
			return true;
		}

		[[nodiscard]] bool has_any_active_parent(DOF_t dof) const noexcept {
			for (DOF_t p : dof.parents()) {
				if (p.exists() && is_active_no_check(p)) {return true;}
			}
			return false;
		}

		[[nodiscard]] bool has_all_active_parents(DOF_t dof) const noexcept {
			for (DOF_t p : dof.parents()) {
				if (p.exists() && !is_active_no_check(p)) {return false;}
			}
			return true;
		}

		[[nodiscard]] bool has_any_refined_parent(DOF_t dof) const noexcept {
			for (DOF_t p : dof.parents()) {
				if (p.exists() && is_refined_no_check(p)) {return true;}
			}
			return false;
		}

		[[nodiscard]] bool has_all_refined_parents(DOF_t dof) const noexcept {
			for (DOF_t p : dof.parents()) {
				if (p.exists() && !is_refined_no_check(p)) {return false;}
			}
			return true;
		}


		//////////////////////////////////////////////////////////////////////////////////////
		/// select the subset of dofs that satisfy some predicate
		//////////////////////////////////////////////////////////////////////////////////////
		template<typename Predicate> requires(std::is_invocable_r_v<bool,Predicate,DOF_t>)
		std::vector<DOF_t> select_dofs(Predicate&& pred) const noexcept {
			return BASE::template select_active_keys<DOF_t>(std::forward<Predicate>(pred));
		}

		template<typename Predicate> requires(std::is_invocable_r_v<bool,Predicate,uint8_t>)
		std::vector<DOF_t> select_dofs_by_mask(Predicate&& pred) const noexcept {
			return BASE::template select_keys_by_mask<DOF_t>(std::forward<Predicate>(pred));
		}

	};


	template<VoxelMeshType MeshType, typename DofType>
	std::ostream& operator<<(std::ostream& os, const DofHandler<MeshType,DofType>& handler) {
		os << "DofHandler: " << DofType::name() + "\n";
		os << handler.summary();
		os << gutil::format(handler.n_dofs(),16) << " active dofs (keys)\n";

		for (int dd=0; dd<= (int) handler.max_depth; ++dd) {
			size_t count = handler.get_sorter().get_bin(dd).size();
			if (count>0) {
				std::cout << "\tdepth " << dd <<" : " << count << "\n";
			}
		}
		return os;
	}
}