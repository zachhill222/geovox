#pragma once

#include "gutil.hpp"

#include <concepts>
#include <cstdint>
#include <vector>
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

	template<VoxelMeshType Mesh_t, typename DOF_t>
	struct AssignableFeature {
		using type = std::conditional_t< std::same_as< typename DOF_t::Key_t::NonPeriodicVariant, typename Mesh_t::VoxelVertex>,  typename Mesh_t::VoxelVertex,
					 std::conditional_t< std::same_as< typename DOF_t::Key_t::NonPeriodicVariant, typename Mesh_t::VoxelElement>, typename Mesh_t::VoxelElement,
					 std::conditional_t< std::same_as< typename DOF_t::Key_t::NonPeriodicVariant, typename Mesh_t::VoxelFace>,    typename Mesh_t::VoxelFace,
					 void>>>;
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
	template<VoxelMeshType MeshType, typename DOFType> requires (!std::same_as<typename AssignableFeature<MeshType,DOFType>::type, void>) 
	struct DofHandler {


		/////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////////////////////
		using Mesh_t = MeshType;
		using DOF_t  = DOFType;
		using Elem_t = typename Mesh_t::VoxelElement;
		using D_KEY  = typename DOF_t::Key_t;
		using M_KEY  = typename D_KEY::NonPeriodicVariant;
		static_assert(std::same_as<M_KEY, typename AssignableFeature<Mesh_t,DOF_t>::type>);

		static constexpr uint64_t MAX_DEPTH = Mesh_t::MAX_DEPTH;
		static constexpr bool IS_DEPTH_SEPARABLE = DepthSeparableVoxelMeshType<Mesh_t>;
		static constexpr uint64_t TOTAL_POSSIBLE_DOFS = total_possible<M_KEY>(MAX_DEPTH);

		static constexpr bool VERTEX_DOF  = std::same_as<M_KEY, typename Mesh_t::VoxelVertex>;
		static constexpr bool ELEMENT_DOF = std::same_as<M_KEY, typename Mesh_t::VoxelElement>;
		static constexpr bool FACE_DOF    = std::same_as<M_KEY, typename Mesh_t::VoxelFace>;

		/////////////////////////////////////////////////////////////////////////
		/// Storage. Store a vector<bool> for O(1) active queries.
		/// Additionally, store a compressed list of active dofs for tracking
		/// global DOF numbers. It is essential for fast quadrature that we may
		/// look up all active DOFs whos support OVERLAPS a given active element.
		/////////////////////////////////////////////////////////////////////////
		std::vector<bool> active_mask;
		std::vector<DOF_t> active_dofs{};
		gutil::BinSort<DOF_t> active_dof_sorter;
		mutable gutil::ThreadPool threads{};
		const Mesh_t& mesh;


		/////////////////////////////////////////////////////////////////////////
		/// Constructors. The dofhandler must be linked to the mesh at construction
		/// and the mesh must outlive the dofhandler.
		/////////////////////////////////////////////////////////////////////////
		DofHandler() = delete;
		DofHandler(const Mesh_t& m) : active_mask(TOTAL_POSSIBLE_DOFS), mesh(m) {}
		DofHandler(const Mesh_t& m, int n_threads) : active_mask(TOTAL_POSSIBLE_DOFS), threads{n_threads}, mesh{m} {}
		DofHandler(const DofHandler& other) : 
			active_mask{other.active_mask.begin(), other.active_mask.end()},
			active_dofs{other.active_dofs.begin(), other.active_dofs.end()},
			threads{other.threads.n_threads()},
			mesh(other.mesh) {}
		DofHandler(DofHandler&& other) :
			active_mask{std::move(other.active_mask)},
			active_dofs{std::move(other.active_dofs)},
			threads{other.threads.n_threads()},
			mesh{other.mesh} {}
		[[nodiscard]] DofHandler& operator=(const DofHandler& other) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				active_mask = other.active_mask;
				active_dofs = other.active_dofs;
				active_dof_sorter = other.active_dof_sorter;
			}
		}
		[[nodiscard]] DofHandler& operator=(DofHandler&&) noexcept {
			GUTIL_ASSERT(&mesh==&other.mesh);
			if (this != &other) {
				active_mask = std::move(other.active_mask);
				active_dofs = std::move(other.active_dofs);
				active_dof_sorter = std::move(other.active_dof_sorter);
			}
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

		[[nodiscard]] static constexpr auto features(Elem_t el) noexcept {
			if constexpr (VERTEX_DOF) { return el.vertices(); }
			else if constexpr (ELEMENT_DOF) { return std::array<Elem_t,1>{el}; }
			else { return el.faces(); }
		}

		[[nodiscard]] static constexpr auto elements(M_KEY key) noexcept {
			if constexpr (VERTEX_DOF) { return key.elements(); }
			else if constexpr (ELEMENT_DOF) { return std::array<Elem_t,1>{key}; }
			else { return key.elements(); }
		}


		/////////////////////////////////////////////////////////////////////////
		/// Utility methods.
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool is_active(DOF_t dof) { GUTIL_ASSERT(dof.is_valid()); return active_mask[dof.linear_index()];}

		void wait() const noexcept { threads.wait_idle(); }

		void set_active(DOF_t dof, bool val) noexcept { active_mask[dof.linear_index()] = val;}

		void clear() noexcept {
			std::fill(active_mask.begin(), active_mask.end());
			active_dofs.clear();
		}

		//sort the dofs by bin to improve lookup and caching
		[[nodiscard]] static constexpr int dof_bin(DOF_t dof) noexcept {
			int bin = 0;
			const uint64_t dd = dof.key.depth();
			const uint64_t split = (dd>0) ? (uint64_t{1}<<(dd-1)) : 0;	//split the index space into 8 octants. ties go to lower bin.
			
			if (dof.key.i() > split) { bin |= 1;}
			if (dof.key.j() > split) { bin |= 2;}
			if (dof.key.k() > split) { bin |= 4;}
			return bin;
		}

		void sort_dofs() noexcept {
			active_dof_sorter = (active_dofs.size() > 1024) ? 
					sort_dofs_multithread(active_dofs) : sort_dofs_singlethread(active_dofs);
		}

		gutil::BinSort<DOF_t> sort_dofs_multithread(std::span<DOF_t> dofs) const noexcept {
			gutil::BinSort<DOF_t> local_sorter(dofs, 8);
			local_sorter.sort([&](DOF_t dof){ return dof_bin(dof); });
			
			for (int i=0; i<8; ++i) {
				threads.submit( [](std::span<DOF_t> list){ std::sort(list.begin(), list.end()); }, local_sorter.get_bin(i));
			}
			threads.wait_idle();

			return local_sorter;
		}

		gutil::BinSort<DOF_t> sort_dofs_singlethread(std::span<DOF_t> dofs) const noexcept {
			gutil::BinSort<DOF_t> local_sorter(dofs, 8);
			local_sorter.sort([&](DOF_t dof){ return dof_bin(dof); });
			for (int i=0; i<8; ++i) {
				std::span<DOF_t> list = local_sorter.get_bin(i);
				std::sort(list.begin(), list.end());
			}
			return local_sorter;
		}

		//sort the dofs by increasing global index and get their global index
		void get_dof_number_local_sort(std::span<DOF_t> dofs, std::span<uint64_t> global_idx) const noexcept {
			auto local_sorter = sort_dofs_singlethread(dofs);
			uint64_t n = 0; //the local dof number (index into the dofs)
			for (int i=0; i<8; ++i) {
				auto local_list = local_sorter.get_bin(i);
				auto global_list = active_dof_sorter.get_bin(i);
				for (uint64_t j=0; j<local_sorter.bin_size(i); ++j, ++n) {
					DOF_t dof = local_list[j]; //subspan of the sorted original dofs
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
		/// TODO: support non-conformal initial state with affine constraints
		/////////////////////////////////////////////////////////////////////////
		void init_dofs() noexcept { dispatch_init_dofs(); wait(); }
		void dispatch_init_dofs() noexcept {
			clear();
			//to avoid false sharing in vector<bool>, we align the dispatched chunks to the word size.
			//this uses the fact that features and dofs share the same key numbering so two different
			//features in different chunks should not be able to access the same word that is stored in
			//the vector<bool>
			constexpr uint64_t chunk_size = 8*sizeof(size_t);
			const uint64_t n_chunks   = n_features()/chunk_size;
			const uint64_t n_workers  = static_cast<uint64_t>(threads.n_threads());
			const uint64_t chunks_per_thread = n_chunks/n_workers;

			for (uint64_t tid=0; tid<n_workers; ++tid) {
				const uint64_t start = tid * chunk_size * chunks_per_thread;
				const uint64_t end   = (tid==n_workers-1) ? n_features() : start + chunk_size*chunks_per_thread;
				auto job = [&](auto it, auto end) {
					while (it != end) {
						const D_KEY key = static_cast<D_KEY>(*it);
						if (*it == static_cast<M_KEY>(key)) {
							set_active(DOF_t{key}, true);
						}
						++it;
					}
				};
				GUTIL_ASSERT(start<=end);
				threads.submit(job, feature_begin()+start, feature_begin()+end);
			}
		}


		/////////////////////////////////////////////////////////////////////////
		/// Queries. Most of these need to be callable from a quadrature loop over
		/// the mesh elements.
		/////////////////////////////////////////////////////////////////////////
		void get_active_dofs_conformal(Elem_t el, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			//assume that dofs only exist at the features of active elements
			//and that all such features correspond to a dof
			GUTIL_ASSERT(mesh.is_active(el));
			for (M_KEY f : features(el)) {
				const D_KEY key = static_cast<D_KEY>(f);
				if (f == static_cast<M_KEY>(key)) {
					GUTIL_ASSERT(is_active(DOF_t{key}));
					dofs.emplace_back(key);
				}
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin(), dofs.end()}, 
						std::span<uint64_t>{global_idx.begin(), global_idx.end()});
		}

		void get_active_dofs_quasi_hierarchical(Elem_t el, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			//assume that dofs only exist at the features of active elements or the feature of a parent of an active element
			//additionally, if a feature at depth d is active, then its parent feature at depth d-1 cannot be active
			GUTIL_ASSERT(mesh.is_active(el));
			for (int i=0; i<2 && el.exists(); ++i) {
				for (M_KEY f : features(el)) {
					const D_KEY key = static_cast<D_KEY>(f);
					if (f == static_cast<M_KEY>(key)) {
						DOF_t dof{key};
						if (is_active(dof)) {dofs.push_back(dof);}
					}
				}
				el = el.parent();
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin(), dofs.end()}, 
						std::span<uint64_t>{global_idx.begin(), global_idx.end()});
		}

		void get_active_dofs_full_hierarchical(Elem_t el, std::vector<DOF_t>& dofs, std::vector<uint64_t>& global_idx) const noexcept {
			//assume that dofs only exist at the features of active elements or the feature of any ancestor of an active element
			//additionally, if a feature at depth d is active, then its parent feature at depth d-1 cannot be active
			GUTIL_ASSERT(mesh.is_active(el));
			while (el.exists()) {
				for (M_KEY f : features(el)) {
					const D_KEY key = static_cast<D_KEY>(f);
					if (f == static_cast<M_KEY>(key)) {
						DOF_t dof{key};
						if (is_active(dof)) {dofs.push_back(dof);}
					}
				}
				el = el.parent();
			}

			global_idx.resize(dofs.size());
			get_dof_number_local_sort(std::span<DOF_t>{dofs.begin(), dofs.end()}, 
						std::span<uint64_t>{global_idx.begin(), global_idx.end()});
		}

		[[nodiscard]] bool is_refinable(DOF_t dof) const noexcept {
			assert(dof.is_valid());
			if (dof.depth() >= BASE::MAX_DEPTH) {return false;}

			for (const DOF_t p : dof.parents()) {
				if (p.exists() && this->is_active(p)) {return false;}
			}
			return true;
		}

		[[nodiscard]] bool is_unrefinable(DOF_t dof) const noexcept {
			assert(dof.is_valid());
			if (dof.depth()==0) {return false;}
			if (!this->is_active(dof)) {return false;}

			for (const DOF_t c : dof.children()) {
				if (c.exists() && this->is_active(c)) {return false;}
			}
			return true;
		}

	};
















}