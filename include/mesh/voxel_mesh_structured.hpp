#pragma once

#include "gutil.hpp"

#include "util/macros.hpp"

#include "mesh/keys/voxel_key.hpp"
#include "mesh/unstructured_layer.hpp"

#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace GV {
	
	
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/// A mesh class for a voxel mesh at a single depth. This class will be used to define a hierarchical
	/// mesh by 'stacking' layers together. When not using this to build a hierarchy, setting MaxDepth=Depth
	/// is fine. The MaxDepth must be consistent across hierarchy levels.
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	template<uint64_t MaxDepth>
	struct StructuredVoxelMesh {


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		//////////////////////////////////////////////////////////////////////////////////////////////////
		using VoxelElement = VoxelElementKey<MaxDepth,0>;
		using VoxelVertex  = VoxelVertexKey<MaxDepth,0>;
		using VoxelFace    = VoxelFaceKey<MaxDepth,0>;
		using GeoPoint_t   = gutil::Point<3,double>;
		using Box_t        = gutil::Box<3,double>;

		uint64_t depth;										//the depth of this mesh
		static constexpr uint64_t MAX_DEPTH = MaxDepth;		//the maximum depth to maintain hierarchies
		
		static constexpr int DIMENSION = 3;	//TODO: support 2D
		[[nodiscard]] constexpr uint64_t axis_max_elements() const noexcept { return uint64_t{1} << depth; }
		[[nodiscard]] constexpr uint64_t max_elements() const noexcept { return uint64_t{1} << (DIMENSION*depth); }
		
		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Storage
		/// Allow storing the full mask of all possible elements or just a vector of active elements.
		//////////////////////////////////////////////////////////////////////////////////////////////////
		Box_t box;											//physical extents of the domain
		std::vector<bool> active_mask{};					//mask of active elements using a structured index
		UnstructuredLayer<MAX_DEPTH> unstructured{};		//a more traditional storage of the mesh

		void init_active_mask() noexcept {
			active_mask.resize(max_elements());
		}

		void delete_active_mask() noexcept {
			active_mask.clear();
			active_mask.shrink_to_fit();
		}


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Constructors and movement. Copying is disallowed.
		//////////////////////////////////////////////////////////////////////////////////////////////////
		StructuredVoxelMesh() = default;
		StructuredVoxelMesh(const StructuredVoxelMesh&) = delete;
		StructuredVoxelMesh& operator=(const StructuredVoxelMesh&) = delete;
		StructuredVoxelMesh(StructuredVoxelMesh&& other) noexcept : 
			depth{other.depth},
			box{std::move(other.box)},
			active_mask{std::move(other.active_mask)}, 
			unstructured{std::move(other.unstructured)} {}
		StructuredVoxelMesh& operator=(StructuredVoxelMesh&& other) noexcept {
			if (this != &other) {
				depth = other.depth;
				box = std::move(other.box);
				active_mask = std::move(other.active_mask);
				unstructured = std::move(other.unstructured);
			}
			return *this;
		}

		StructuredVoxelMesh(const Box_t& box, uint64_t depth) noexcept : depth{depth}, box{box} {unstructured.box = box;}


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Helpful simple queries
		//////////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool is_active(VoxelElement el) const noexcept {
			assert(active_mask.size() == max_elements());
			return active_mask[el.depth_linear_index()];
		}


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Convert to/from an unstructured mesh for a single layer
		//////////////////////////////////////////////////////////////////////////////////////////////////
		StructuredVoxelMesh(const UnstructuredLayer<MAX_DEPTH>& layer) noexcept {
			set_mask(false);

			for (uint64_t i=0; i<layer.n_elements(); ++i) {
				active_mask[layer.elements[i].depth_linear_index()] = true;
			}
		}

		void update_unstructured() noexcept {
			unstructured.clear();
			for (uint64_t i=0; i<max_elements(); ++i) {
				const VoxelElement el(depth,i);
				if (is_active(el)) { unstructured.push_back(el); }
			}
		}

		void update_unstructured_omp(int n_threads=-1) noexcept {
			if (n_threads < 0) {n_threads = omp_get_max_threads();}

			//dispatch jobs
			std::vector<std::vector<VoxelElement>> thread_elements(n_threads);
			auto action = [this,&thread_elements](VoxelElement el, int tid) {
				if (is_active(el)) {thread_elements[tid].push_back(el);}
			};
			for_each_element_omp(std::move(action), n_threads);

			//collect elements
			unstructured.clear();
			for (int tid=0; tid<n_threads; ++tid) {
				unstructured.elements.insert( unstructured.elements.end(),
									std::make_move_iterator(thread_elements[tid].begin()),
									std::make_move_iterator(thread_elements[tid].end()));
			}
		}


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Set the mask.
		//////////////////////////////////////////////////////////////////////////////////////////////////
		template<typename Predicate>
		void set_mask(Predicate&& pred) noexcept {
			unstructured.clear();
			init_active_mask();

			for_each_index( [&](uint64_t i) { active_mask[i] = pred(VoxelElement(depth,i)); });
		}

		void set_mask(const bool val=false) noexcept {
			unstructured.clear();
			active_mask.resize(max_elements());
			std::fill(active_mask.begin(), active_mask.end(), val);
		}

		void set_element(std::span<const VoxelElement> list, const bool val) noexcept {
			for (VoxelElement el : list) { 
				if (el.exists()) {
					active_mask[el.depth_linear_index()] = val;	
				}
			}
		}

		void set_element(VoxelElement el, bool val) noexcept {
			if (el.exists()) {
				active_mask[el.depth_linear_index()] = val;
			}
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Loop over every possible element. To loop over active elements only, use the unstructured mesh.
		/// Note Action is allowed to edit the mesh if the caller can capture a non-const reference.
		///
		/// Looping over elements in an ADI fashion is possible, but should be explicitly written.
		//////////////////////////////////////////////////////////////////////////////////////////////////
		#ifdef _OPENMP
		template<typename Action>
		void for_each_element_omp(Action&& action, int n_threads=-1) const noexcept {
			constexpr uint64_t CHUNK = 512;
			const uint64_t n_chunk = (max_elements()+CHUNK-1) / CHUNK;
			if (n_threads < 0) { n_threads = omp_get_max_threads(); }

			//each thread gets one contiguous sequence of chunks.
			//for example, thread 0 may get chunks [0,5), thread 1 chunks [5,10), and thread 2 chunks [10,12)
			//with the size of each range of chunks split evenly with the remainder sent to the last thread.
			#pragma omp parallel for schedule(static) num_threads(n_threads)
			for (uint64_t c=0; c<n_chunk; ++c) {
				const int tid = omp_get_thread_num();
				const uint64_t c_start = c*CHUNK;
				const uint64_t c_end   = std::min(c_start+CHUNK, max_elements());
				
				VoxelElement el(depth, c_start);
				for (uint64_t idx=c_start; idx<c_end; ++idx, ++el) {
					GUTIL_ASSERT(el.depth_linear_index() == idx);
					if constexpr (std::is_invocable_v<Action, VoxelElement, int>) {
						action(el, tid);
					}
					else {
						action(el);
					}
				}
			}
		}

		template<typename Action>
		void for_each_index_omp(Action&& action, int n_threads=-1) const noexcept {
			constexpr uint64_t CHUNK = 512;
			const uint64_t n_chunk = (max_elements()+CHUNK-1) / CHUNK;
			if (n_threads < 0) { n_threads = omp_get_max_threads(); }

			//each thread gets one contiguous sequence of chunks.
			//for example, thread 0 may get chunks [0,5), thread 1 chunks [5,10), and thread 2 chunks [10,12)
			//with the size of each range of chunks split evenly with the remainder sent to the last thread.
			#pragma omp parallel for schedule(static) num_threads(n_threads)
			for (uint64_t c=0; c<n_chunk; ++c) {
				const int tid = omp_get_thread_num();
				const uint64_t c_start = c*CHUNK;
				const uint64_t c_end   = std::min(c_start+CHUNK, max_elements());
				
				for (uint64_t idx=c_start; idx<c_end; ++idx) {
					if constexpr (std::is_invocable_v<Action, uint64_t, int>) {
						action(idx, tid);
					}
					else {
						action(idx);
					}
				}
			}
		}

		template<typename Action>
		void for_each_element_simd(Action&& action) const noexcept {
			constexpr uint64_t WORD_SIZE = sizeof(size_t);
			#pragma omp simd safelen(WORD_SIZE)
			for (uint64_t i=0; i<max_elements(); ++i) {
				VoxelElement el(depth,i);
				action(el);
			}
		}

		template<typename Action>
		void for_each_index_simd(Action&& action) const noexcept {
			constexpr uint64_t WORD_SIZE = sizeof(size_t);
			#pragma omp simd safelen(WORD_SIZE)
			for (uint64_t i=0; i<max_elements(); ++i) {
				action(i);
			}
		}

		#else

		template<typename Action>
		void for_each_element_omp(Action&& action, int n_threads) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			for_each_element(std::forward<Action>(action));
		}

		template<typename Action>
		void for_each_element_simd(Action&& action) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			for_each_element(std::forward<Action>(action));
		}

		template<typename Action>
		void for_each_index_omp(Action&& action, int n_threads) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			for_each_index(std::forward<Action>(action));
		}

		template<typename Action>
		void for_each_index_simd(Action&& action) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			for_each_index(std::forward<Action>(action));
		}

		#endif


		template<typename Action>
		void for_each_element(Action&& action) const noexcept {
			VoxelElement el(depth,0);
			for (uint64_t i=0; i<max_elements(); ++i, ++el) {
				assert( el == VoxelElement(depth,i) );
				action(el);
			}
		}

		template<typename Action>
		void for_each_index(Action&& action) const noexcept {
			for (uint64_t i=0; i<max_elements(); ++i) {
				action(i);
			}
		}
	};
}