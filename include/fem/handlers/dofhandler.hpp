#pragma once

#include "gutil.hpp"

#include "simd_keys/containers.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"

#include "util/byte_print.hpp"
#include "util/compatibility.hpp"

#include <concepts>
#include <cstdint>
#include <vector>
#include <span>
#include <type_traits>
#include <algorithm>
#include <thread>

#ifndef GV_MAX_RUNTIME_KEY_DEPTH
	#define GV_MAX_RUNTIME_KEY_DEPTH uint8_t{10}
#endif

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
		{ mesh.max_depth } 			-> std::convertible_to<uint8_t>;

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
		
		//The maximum depth of a given mesh is specified at runtime. However, to avoid accidentally
		//requesting say depth 12 (2^(3*12) elements at depth 12, (2^33 -1)/7 ~ 10^10.3 total elements)
		//we set a maximum depth at compile time.
		const  			 uint8_t  max_depth;
		const 			 uint64_t max_possible_dofs;

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
		using BASE::is_key_mask_unstable;
		using BASE::is_key_mask_stable;
		using BASE::start_key_mask_stable;
		using BASE::end_key_mask_stable;
		using BASE::start_key_mask_unstable;
		using BASE::end_key_mask_unstable;
		
		using BASE::is_active_keys_unstable;
		using BASE::is_active_keys_stable;
		using BASE::start_active_keys_stable;
		using BASE::end_active_keys_stable;
		using BASE::start_active_keys_unstable;
		using BASE::end_active_keys_unstable;

		std::span<DOF_t>					active_dofs;
		const Mesh_t& 						mesh;				//link to the mesh, we can request refinement through const methods
		mutable std::atomic<bool> 			batch_is_set{false};//check if the batch start bit has been set.

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
						GUTIL_LOG("Requested pool size (", n_threads, ") exceeds what is seen by std::thread::hardware_concurrency() (",
								std::thread::hardware_concurrency(), ")");
					}
					GUTIL_ASSERT(max_depth == m.max_depth);
					GUTIL_ASSERT(max_depth<GV_MAX_RUNTIME_KEY_DEPTH);
				}
				
		
		DofHandler(const DofHandler& other) : BASE(other), mesh(other.mesh) {
			active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(std::span<uint64_t>(BASE::active_keys));
		}
		DofHandler(DofHandler&& other) : BASE(std::move(other)), mesh(other.mesh) {
			active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(std::span<uint64_t>(BASE::active_keys));
		}

		[[nodiscard]] DofHandler& operator=(const DofHandler& other) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				key_mask = other.key_mask;
				BASE::active_keys = other.active_keys;
				active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(std::span<uint64_t>(BASE::active_keys));
			}
			return *this;
		}
		[[nodiscard]] DofHandler& operator=(DofHandler&& other) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				key_mask = std::move(other.key_mask);
				BASE::active_keys = std::move(other.active_keys);
				active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(std::span<uint64_t>(BASE::active_keys));
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
			for (uint64_t dd=vtx.depth(); dd<=Mesh_t::MAX_DEPTH; ++dd) {
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
		[[nodiscard]] bool set_active_compare_exchange(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_active_compare_exchange(dof.linear_index(), val);
		}
		void set_active(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_active(dof.linear_index(), val);
		}

		
		[[nodiscard]] bool is_refined(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_refined_stable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_stable<REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_refined_unstable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_unstable<REFINED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool set_refined_compare_exchange(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_bit_compare_exchange<REFINED_BIT>(dof.linear_index(), val);
		}
		void set_refined(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_bit<REFINED_BIT>(dof.linear_index(), val);
		}


		[[nodiscard]] bool is_coef_marked(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<COEF_MARKED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_coef_marked_stable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_stable<COEF_MARKED_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_coef_marked_unstable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_unstable<COEF_MARKED_BIT>(dof.linear_index());
		}
		//called via const ref, the mask is mutable
		[[nodiscard]] bool set_coef_marked_compare_exchange(DOF_t dof, bool val) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_bit_compare_exchange<COEF_MARKED_BIT>(dof.linear_index(), val);
		}
		//called via const ref, the mask is mutable
		void set_coef_marked(DOF_t dof, bool val) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_bit<COEF_MARKED_BIT>(dof.linear_index(), val);
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
		[[nodiscard]] bool set_initial_marked_compare_exchange(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_bit_compare_exchange<INITIAL_DOF_BIT>(dof.linear_index(), val);
		}
		void set_initial_marked(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_bit<INITIAL_DOF_BIT>(dof.linear_index(), val);
		}


		[[nodiscard]] bool is_batch_marked(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit<BATCH_PROCESS_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_batch_marked_stable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_stable<BATCH_PROCESS_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool is_batch_marked_unstable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::check_bit_unstable<BATCH_PROCESS_BIT>(dof.linear_index());
		}
		[[nodiscard]] bool set_batch_marked_compare_exchange(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); return BASE::set_bit_compare_exchange<BATCH_PROCESS_BIT>(dof.linear_index(), val);
		}
		void set_batch_marked(DOF_t dof, bool val) noexcept {
			GUTIL_ASSERT(dof.is_valid()); BASE::set_bit<BATCH_PROCESS_BIT>(dof.linear_index(), val);
		}


		///////////////////////////////////////////////////////////////////////////
		/// Define functions to sort DOFs into bins and to look up global dof numbers
		///////////////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr int8_t dof_key_bin(uint64_t dof_key) noexcept {
			int8_t bin = 0;
			DOF_t dof{dof_key};
			const uint64_t dd = dof.depth();
			const uint64_t split = (dd>0) ? (uint64_t{1}<<(dd-1)) : 0;
			
			if (dof.i() > split) { bin |= 1;}
			if (dof.j() > split) { bin |= 2;}
			if (dof.k() > split) { bin |= 4;}
			return bin;
		}

		[[nodiscard]] static constexpr int dof_bin(DOF_t dof) noexcept {
			int bin = 0;
			const uint64_t dd = dof.depth();
			const uint64_t split = (dd>0) ? (uint64_t{1}<<(dd-1)) : 0;
			
			if (dof.i() > split) { bin |= 1;}
			if (dof.j() > split) { bin |= 2;}
			if (dof.k() > split) { bin |= 4;}
			return bin;
		}

		//get a single dof global number
		[[nodiscard]] uint64_t global_number(DOF_t dof) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return BASE::lookup_key(dof.key, &dof_key_bin);
		}

		//sort the dofs by increasing global index and get their global index
		void get_dof_number_local_sort(std::span<DOF_t> dofs, std::span<uint64_t> global_index) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			BASE::lookup_keys_and_sort(
				BASE::reinterpret_key_span<uint64_t,DOF_t>(dofs), global_index, 8, &dof_key_bin);
		}
		

		/////////////////////////////////////////////////////////////////////////
		/// Initialization. The mesh must be in a conformal state.
		/////////////////////////////////////////////////////////////////////////
		void init_dofs() noexcept {
			//TODO: ensure the mesh is in a stable state after it inherits from the HybridKeyTracker
			BASE::clear();

			if (mesh.element_begin()==mesh.element_end()) {
				GUTIL_ABORT("ERROR: no elements found.")
			}

			{
				GV_BEGIN_UNSTABLE

				//TODO: dispatch to the mesh parallel pool after it inherits from the HybridKeyTracker
				const uint64_t n_threads     = threads.n_threads() == 0 ? 1 : threads.n_threads();
				const uint64_t el_per_thread = mesh.n_elements()/n_threads;

				for (uint64_t tid=0; tid<n_threads; ++tid) {
					const uint64_t start = tid * el_per_thread;
					const uint64_t end   = (tid==n_threads-1) ? mesh.n_elements() : start + el_per_thread;
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
									BASE::set_active(idx, true);

									//mark as explicitly refinable
									if (dof.depth() != max_depth) {
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
				threads.wait_idle();
				GV_END_UNSTABLE
			}//release mutex (collect dofs needs to re-aquire the unique lock)
			collect_dofs();
		}


		/////////////////////////////////////////////////////////////////////////
		/// Book keeping methods
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] size_t n_dofs() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			return active_dofs.size();
		}

		void collect_dofs() noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			BASE::collect_active_keys<DOF_t>();
			active_dofs = BASE::reinterpret_key_span<DOF_t,uint64_t>(std::span<uint64_t>(BASE::active_keys));
			BASE::sort_active_keys(8, &DofHandler::dof_key_bin);
		}

		/// mark the start of a wide (look at lots of dofs) process
		/// set_batch_start() sets the BATCH_PROCESS_BIT to true on all active dof masks
		void set_batch_start() noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			BASE::conditional_bitwise_or_all_masks(ACTIVE_BIT,BATCH_PROCESS_BIT);
			batch_is_set = true;
		}

		/// mark the end of a wide (look at lots of dofs) process
		/// set_batch_end() sets the BATCH_PROCESS_BIT to false on all dof masks
		void set_batch_end() noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
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
			GV_BEGIN_MASK_UNSTABLE

			GUTIL_ASSERT(BASE::is_current());
			GUTIL_ASSERT(is_sorted() && "DofHandler - the dofs must be sorted to get the correct global index");
			GUTIL_ASSERT(dofs.size()==global_idx.size());
			//assume that dofs only exist at the features of active elements
			//and that all such features correspond to a dof
			//this is the same as getting the "basis_s" dofs in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			size_t start_size = dofs.size();

			for (DofFeature_t feat : features(el)) {
				const DOF_t dof{feat};
				if (dof.exists() && is_active_unstable(dof)) {
					dofs.push_back(dof);
				}
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin()+start_size, dofs.end()}, 
						std::span<uint64_t>{global_idx.begin()+start_size, global_idx.end()});

			GV_END_MASK_UNSTABLE
		}

		template<typename Elem_t> requires(std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t> )
		void get_active_dofs_quasi_hierarchical(Elem_t el_, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			GV_BEGIN_ACTIVE_STABLE

			GUTIL_ASSERT(is_current() && "DofHandler - the dofs must be sorted to get the correct global index");
			GUTIL_ASSERT(dofs.size()==global_idx.size());
			//assume that dofs only exist at the features of active elements or the feature of a parent of an active element
			//additionally, if a feature at depth d is active, then its parent feature at depth d-1 cannot be active
			//this is the same as getting "basis_s U basis_a" in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			size_t start_size = dofs.size();

			for (int i=0; i<2 && el.exists(); ++i) {
				for (DofFeature_t feat : features(el)) {
					const DOF_t dof{feat};
					if (dof.exists() && is_active_stable(dof)) {
						dofs.push_back(dof);
					}
				}
				el = el.parent();
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin()+start_size, dofs.end()}, 
						std::span<uint64_t>{global_idx.begin()+start_size, global_idx.end()});

			GV_END_ACTIVE_STABLE
		}

		template<typename Elem_t> requires(std::same_as<Elem_t,DofElem_t> || std::same_as<Elem_t,MeshElem_t> )
		void get_active_dofs_full_hierarchical(Elem_t el_, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			GV_BEGIN_ACTIVE_STABLE

			GUTIL_ASSERT(is_current() && "DofHandler - the dofs must be sorted to get the correct global index");
			GUTIL_ASSERT(dofs.size()==global_idx.size());
			//assume that dofs only exist at the features of active elements or the feature of any ancestor of an active element
			//additionally, if a feature at depth d is active, then its parent feature at depth d-1 cannot be active
			//this is the same as getting "basis_s U basis_a" in a hierarchical method
			
			DofElem_t el = static_cast<DofElem_t>(el_);
			size_t start_size = dofs.size();

			while (el.exists()) {
				for (DofFeature_t feat : features(el)) {
					const DOF_t dof{feat};
					if (dof.exists() && is_active_stable(dof)) {
						dofs.push_back(dof);
					}
				}
				el = el.parent();
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin()+start_size, dofs.end()}, 
						std::span<uint64_t>{global_idx.begin()+start_size, global_idx.end()});

			GV_END_ACTIVE_STABLE
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
		/// to collect the active dofs. Similarly, you can only get dofs that are marked with the batch bit.
		/// Alternatively, the dof can be passed to the predicate.
		///
		/// The possible signatures are bool(uint8_t), bool(uint8_t,DOF_t), bool(DOF_t,uint8_t), bool(DOF_t).
		///
		/// You may also pass an action to take on the dof_mask. For example, turn off the batch bit.
		/// The action must have the signature void(uint_t&). To work with the active dof representation,
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
		template<typename Predicate, typename Action>
		[[nodiscard]] std::vector<DOF_t> get_dofs_impl(
				std::span<const DofElem_t> 	elems,
				Predicate&& 				pred,
				Action&& 					action,
				uint8_t 					n_depths) noexcept 
		{	
			GUTIL_TIMER("Gathering unique dofs on ", elems.size(), " elements");

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
				dof_elems[i] = DofElem_t{dof_elems[i].decode_simd()};
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

				std::sort(scratch.begin(), scratch.end(),					//clean up this depth
							[](DOF_t a, DOF_t b) {return b<a;});			//note that DOF_t{0} (the does not exist flag) will be the last element
				auto it = std::unique(scratch.begin(), scratch.end());
				scratch.erase(it, scratch.end());							//erases all but one non-existant dofs (there could be 1 non-existant element)
				// if (!scratch.back().exists()) {scratch.pop_back();}			//all dofs exist now.
				std::erase_if(scratch, [](DOF_t dof){ return !dof.exists() || !dof.is_valid();});

				//gather valid dofs
				result.reserve(result.size() + scratch.size());
				for (DOF_t dof : scratch) {
					// GUTIL_LOG(dof, " : ", print_bytes(dof.key));
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

			GUTIL_ASSERT(BASE::is_current());
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

			GUTIL_ASSERT(BASE::is_current());
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
		/// Refinement queries that can be called inside either a stable or unstable
		/// region.
		///
		///	From 'Natural hierarchical refinement for finite element methods'
		///	in International J. for Numerical Methods in Engineering (2003, DOI 10.1002/nme.601)
		///
		/// 	There are 3 rules to guide refinement:
		///
		///	1) The refining/unrefining of a dof at depth dd may activate or deactivate that dof
		///			or any of its children at depth dd+1.
		///	2) A dof on level dd+1>0 may be refined only when all its parents on level dd have
		///			been refined. (We track this by the INITIAL_DOF_BIT. here dd=0 is the root element)
		///	3) A dof on level dd may be unrefined only if a) it was previously refined and
		///			b) all its children on level dd+1 are not refined.
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
			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(is_active_no_check(dof));

			if (dof.depth() >= max_depth) {return false;}		//we can't refine past max depth
			
			uint8_t byte = BASE::get_mask_no_check(dof.linear_index());
			if (byte&REFINED_BIT) {return false;}				//we can't refine a dof twice
			if (byte&INITIAL_DOF_BIT) {return true;}			//an unrefined initial dof not at max depth can be refined
			
			for (const DOF_t p : dof.parents()) {
				if (p.exists() && mesh_can_support_any(p)) {	//if the mesh does not capture the entire [0,1]^3 space, don't consider irrelevant parents
					const uint8_t p_byte = BASE::get_mask_no_check(p.linear_index());
					if (!(bool)(p_byte&REFINED_BIT)) {return false;}//a dof can't be refined until ALL of its parents are refined
				}
			}
			return true;
		}

		[[nodiscard]] bool is_unrefinable(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			if (dof.depth()>=max_depth) {return false;}			//we call unrefine on the parent. at max depth it can't be a parent.
			const uint8_t byte = BASE::get_mask_no_check(dof.linear_index());
			if (byte&INITIAL_DOF_BIT) {return false;}			//we can't unrefine to be coarser than the initial dofs
			if (!(bool)(byte&REFINED_BIT)) {return false;}		//we can't unrefine a dof that hasn't been previously refined

			for (const DOF_t c : dof.children()) {				//we can't unrefine a dof if it has valid children that have been refined
				if (c.exists() && mesh_can_support_any(c)) {
					const uint8_t c_byte = BASE::get_mask_no_check(c.linear_index());
					if (c_byte&REFINED_BIT) {return false;}
				}
			}
			return true;
		}

		[[nodiscard]] bool can_deactivate(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			for (DOF_t p : dof.parents()) {						//rather than making a recursive unrefine, we check if child dofs can be deactivated
				if (p.exists()) {
					const uint8_t byte = BASE::get_mask_no_check(p.linear_index());
					if (byte&REFINED_BIT) {return false;}
				}
			}
			return true;
		}


		////////////////////////////////////////////////////////////////////
		/// Refinement operations that must be called from within an unstable mask region.
		///
		/// Note that we are only requesting the mesh to refine. There is no guarentee
		/// that it will resolve every requested support element (e.g., when a complex geometry is being modeled).
		/// Additionally, the mesh refinement will note be done until the refinement batch is over.
		/// 
		/// Note that the mesh has a depth field so that we do not have to see if the support of a dof
		/// is resolved by finer elements than needed. If that is the case, the depth marker will be larger
		/// than the element depth.
		////////////////////////////////////////////////////////////////////
		void activate(DOF_t dof) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());

			if (!set_active_compare_exchange(dof,true)) {		//return if the dof was already active
				return;
			}
																//when refining, it is essential to have the mesh be able to resolve the support
			const uint8_t depth = dof.depth_u8();
			if (depth==0) { return; }							//at depth 0, there is nothing to do
			for (DofElem_t spt : dof.support()) {			
				if (!spt.exists()) {continue;}
				MeshElem_t el = static_cast<MeshElem_t>(spt);
				if (el.exists() && mesh.read_depth(el) < depth) {
					GUTIL_ASSERT(mesh.is_active(el.parent()))				//the mesh should be respecting a 2-1 refinement rule
					mesh.refine(el.parent());								//this is a request. pushes the element to a mutable list. it is protected by a mutex.
				}
			}
		}

		void deactivate(DOF_t dof) noexcept {						//when un-refining, it is not essential to have the mesh un-refine as well.
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE						//mesh unrefinement should be done in some cleanup pass so that
			set_active(dof,false); 									//multiple dofhandlers can be organized
		}															

		void refine_quasi_hierarchical(DOF_t dof) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(is_active_unstable(dof));

			if(!is_refinable(dof)) {return;};
			for (DOF_t c : dof.children()) {
				if (c.exists()) { activate(c); }
			}

			uint8_t& byte = get_mask_ref(dof);
			byte&=~ACTIVE_BIT;							//deactivate parent
			byte|=REFINED_BIT;							//set refined
			GUTIL_ASSERT(is_refined_unstable(dof));
			GUTIL_ASSERT(!is_active_unstable(dof));
			GUTIL_ASSERT(is_unrefinable(dof) || (get_mask_unstable(dof)&INITIAL_DOF_BIT));
		}

		void refine_hierarchical(DOF_t dof) noexcept requires(VERTEX_DOF) {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(is_active_stable(dof));

			if(!is_refinable(dof)) {return;};
			const MeshFeature_t p_feat = static_cast<MeshFeature_t>(dof);
			for (DOF_t c : dof.children()) {
				if (c.exists() && static_cast<MeshFeature_t>(c).parent() != p_feat) {
					activate(c);
				}
			}

			uint8_t& byte = get_mask_ref(dof);
			byte|=REFINED_BIT;							//set refined
			GUTIL_ASSERT(is_refined_unstable(dof));
			GUTIL_ASSERT(is_active_unstable(dof));
			GUTIL_ASSERT(is_unrefinable(dof) || (get_mask_unstable(dof)&INITIAL_DOF_BIT));
		}

		void unrefine_quasi_hierarchical(DOF_t dof) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());

			uint8_t& byte = get_mask_ref(dof);

			if (byte&ACTIVE_BIT) { return; }						//in a Q-H scheme, the parent dof must not be active
			if(!is_unrefinable(dof)) {return;};
			
			activate(dof);											//processes mesh refinement request
			for (DOF_t c : dof.children()) {
				if (c.exists()) {
					GUTIL_ASSERT(!is_refined_unstable(c));
					if (can_deactivate(c)) {
						set_active(c, false);
					}
				}
			}

			byte|=ACTIVE_BIT;										//mark as active
			byte&=~REFINED_BIT;										//mark as not refined

			#ifndef NDEBUG
				bool success  = true;
				if (!is_active_unstable(dof)) {
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
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(is_active_unstable(dof));
			
			//in a quasi-hierarchical scheme, the parent dof (this) stays active
			if(!is_active_unstable(dof) || !is_unrefinable(dof)) {return;};
			for (DOF_t c : dof.children()) {
				if (c.exists()) {set_active(c, false);}
			}

			uint8_t& byte = get_mask_ref(dof);
			byte&=~REFINED_BIT;										//mark as not refined
			GUTIL_ASSERT(is_unrefinable(dof));
		}


		//////////////////////////////////////////////////////////////////////////////////////
		/// Primary refine/unrefine commands for bulk operations. Called from outside this class.
		//////////////////////////////////////////////////////////////////////////////////////
		template<typename Container_t>
		[[maybe_unused]] size_t refine_quasi_hierarchical(const Container_t& elems) noexcept {
			return refine_quasi_hierarchical(as_span(elems));
		}

		template<std::contiguous_iterator I>
		[[maybe_unused]] size_t refine_quasi_hierarchical(I begin, I end) noexcept {
			return refine_quasi_hierarchical(std::span<std::iter_value_t<I>>{begin, end});
		}

		template<typename Elem_t> requires (std::same_as<Elem_t,MeshElem_t> || std::same_as<Elem_t,DofElem_t>)
		[[maybe_unused]] size_t refine_quasi_hierarchical(std::span<const Elem_t> elems) noexcept {
			GUTIL_TIMER("Refining (QH) dofs on ", elems.size(), " elements");
			size_t n_start = active_dofs.size();
			
			//a dof must be active and not refined and in the current batch.
			constexpr uint8_t MASK = ACTIVE_BIT | BATCH_PROCESS_BIT;	//0b00010001
			auto pred = [MASK](uint8_t byte) -> bool {
				return ((byte&MASK) == MASK) && ((byte&REFINED_BIT) == 0);
			};

			auto action = [](uint8_t& byte) -> void {
				byte&=~BATCH_PROCESS_BIT;
			};

			BASE::mark_stale();
			{
				GV_BEGIN_MASK_UNSTABLE
				set_batch_start();
				std::vector<DOF_t> dofs = get_dofs_impl(BASE::reinterpret_key_span<DofElem_t,Elem_t>(elems), 
											std::move(pred), std::move(action), max_depth);
				
				// TODO use binsort so we recover the partition between depths. then refine in parallel at
				// each depth. Refine coarse to fine.
				std::sort(dofs.begin(), dofs.end(), [](DOF_t a, DOF_t b){return a.depth() < b.depth();});

				for (size_t i=0; i<dofs.size(); ++i) {
					#ifndef NDEBUG
						GUTIL_ASSERT(dofs[i].is_valid());
						uint8_t byte = get_mask_unstable(dofs[i]);
						GUTIL_ASSERT(byte&ACTIVE_BIT);
						GUTIL_ASSERT((byte&REFINED_BIT)==0);
						GUTIL_ASSERT((byte&BATCH_PROCESS_BIT)==0);
					#endif
					refine_quasi_hierarchical(dofs[i]);
				}

				set_batch_end();
				GV_END_MASK_UNSTABLE
			}

			collect_dofs();
			size_t n_end = active_dofs.size();
			GUTIL_LOG("n_dofs at start=", n_start, ", n_dofs at end=", n_end, " (", n_end-n_start, ")");
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(n_end>=n_start);
			return n_end - n_start;
		}


		template<typename Container_t>
		[[maybe_unused]] size_t unrefine_quasi_hierarchical(const Container_t& elems) noexcept {
			return unrefine_quasi_hierarchical(as_span(elems));
		}

		template<std::contiguous_iterator I>
		[[maybe_unused]] size_t unrefine_quasi_hierarchical(I begin, I end) noexcept {
			return unrefine_quasi_hierarchical(std::span<std::iter_value_t<I>>{begin, end});
		}

		template<typename Elem_t> requires (std::same_as<Elem_t,MeshElem_t> || std::same_as<Elem_t,DofElem_t>)
		[[maybe_unused]] size_t unrefine_quasi_hierarchical(std::span<const Elem_t> elems) noexcept {

			GUTIL_TIMER("Unrefining (QH) dofs on ", elems.size(), " elements");
			size_t n_start = active_dofs.size();
			
			//a dof must be inactive and refined and in the current batch
			//it also must have no refined children, so it must be at most one depth above.
			auto pred = [](uint8_t byte) -> bool {
				return ((byte&ACTIVE_BIT)==0) && ((byte&REFINED_BIT)) && ((byte&BATCH_PROCESS_BIT));
			};

			auto action = [](uint8_t& byte) -> void {
				byte&=~BATCH_PROCESS_BIT;
			};

			BASE::mark_stale();
			set_batch_start();
			GV_BEGIN_MASK_UNSTABLE	//the action clears a bit
			std::vector<DOF_t> dofs = get_dofs_impl(BASE::reinterpret_key_span<DofElem_t,Elem_t>(elems), 
										std::move(pred), std::move(action), max_depth);


			//unrefine from deepest to shallowest
			std::sort(dofs.begin(), dofs.end(), [](DOF_t a, DOF_t b){ return a.depth()>b.depth(); });

			for (size_t i=0; i<dofs.size(); ++i) {
				#ifndef NDEBUG
					GUTIL_ASSERT(dofs[i].is_valid());
					uint8_t byte = get_mask_unstable(dofs[i]);
					GUTIL_ASSERT((byte&ACTIVE_BIT)==0);
					GUTIL_ASSERT((byte&REFINED_BIT));
					GUTIL_ASSERT((byte&BATCH_PROCESS_BIT)==0);
				#endif
				unrefine_quasi_hierarchical(dofs[i]);
			}

			GV_END_MASK_UNSTABLE
			set_batch_end();
			collect_dofs();

			size_t n_end = active_dofs.size();
			GUTIL_LOG("n_dofs at start=", n_start, ", n_dofs at end=", n_end, " (", n_start-n_end, ")");
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(n_start>=n_end);
			return n_start-n_end;
		}

		// template<typename Elem_t> requires (std::same_as<Elem_t,MeshElem_t> || std::same_as<Elem_t,DofElem_t>)
		// void refine_hierarchical(std::span<const Elem_t> elems) noexcept {
		// 	BASE::mark_stale();
		// 	set_batch_start();

		// 	std::vector<DOF_t> dofs = get_active_dofs_impl<true,Elem_t>(elems, max_depth);
		// 	for (DOF_t dof : dofs) {
		// 		GUTIL_ASSERT(is_active_stable(dof));
		// 		refine_hierarchical(dof);
		// 	}

		// 	set_batch_end();
		// 	collect_dofs();
		// }

		// template<typename Elem_t> requires (std::same_as<Elem_t,MeshElem_t> || std::same_as<Elem_t,DofElem_t>)
		// void unrefine_quasi_hierarchical(std::span<const Elem_t> elems) noexcept {
		// 	BASE::mark_stale();
		// 	set_batch_start();

		// 	std::vector<DOF_t> dofs = get_active_dofs_impl<true,Elem_t>(elems, max_depth);
		// 	for (DOF_t dof : dofs) {
		// 		GUTIL_ASSERT(is_active_stable(dof));
		// 		MeshFeature_t feat = static_cast<MeshFeature_t>(dof);
		// 		unrefine_quasi_hierarchical(static_cast<DOF_t>(feat.parent()));
		// 	}

		// 	set_batch_end();
		// 	collect_dofs();
		// }

	};
















}