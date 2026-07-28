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
		using Mesh_t       = StructuredVoxelMesh<MaxDepth>;

		uint64_t depth;										//the depth of this mesh
		int max_omp_threads{0};								//how many OpenMP threads this layer is allowed to use
		static constexpr uint64_t MAX_DEPTH = MaxDepth;		//the maximum depth to maintain hierarchies
		static_assert(MAX_DEPTH < 16, "Max depth is too large to store in 4 bits");

		static constexpr int DIMENSION = 3;	//TODO: support 2D
		[[nodiscard]] constexpr uint64_t axis_max_elements() const noexcept { return uint64_t{1} << depth; }
		[[nodiscard]] constexpr uint64_t max_elements() const noexcept { return uint64_t{1} << (DIMENSION*depth); }
		
		static constexpr unsigned char ACTIVE_BIT = 0b00000001;
		static constexpr unsigned char DEPTH_MASK = 0b00011110;
		static constexpr unsigned char FREE_MASK  = 0b11100000;

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Storage
		/// Allow storing the full mask of all possible elements or just a vector of active elements.
		/// We choose to store element masks as a vector<unsigned char> rather than a vector<bool> so that
		/// different elements can be safely written to from different threads and so that the remaining bits
		/// can be used for a depth field for the hierarchical/layered mesh.
		//////////////////////////////////////////////////////////////////////////////////////////////////
		Box_t box;											//physical extents of the domain
		std::vector<unsigned char> element_mask{};				//mask of active elements using a structured index
		UnstructuredLayer<MAX_DEPTH> unstructured{};		//a more traditional storage of the mesh

		void init_element_mask() noexcept {
			element_mask.resize(max_elements(), 0);
		}

		void delete_element_mask() noexcept {
			element_mask.clear();
			element_mask.shrink_to_fit();
		}


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Constructors and movement
		//////////////////////////////////////////////////////////////////////////////////////////////////
		StructuredVoxelMesh() = default;
		StructuredVoxelMesh(const StructuredVoxelMesh& other) :
			depth{other.depth},
			max_omp_threads{other.max_omp_threads},
			box{other.box},
			element_mask{other.element_mask},
			unstructured{other.unstructured}
			{
				GUTIL_DEBUG(gutil::Logger::log("WARNING - copying a potentially large mesh");)
			}
		StructuredVoxelMesh& operator=(const StructuredVoxelMesh&) = delete;
		StructuredVoxelMesh(StructuredVoxelMesh&& other) noexcept : 
			depth{other.depth},
			max_omp_threads{other.max_omp_threads},
			box{std::move(other.box)},
			element_mask{std::move(other.element_mask)}, 
			unstructured{std::move(other.unstructured)} {}
		StructuredVoxelMesh& operator=(StructuredVoxelMesh&& other) noexcept {
			if (this != &other) {
				depth = other.depth;
				max_omp_threads = other.max_omp_threads;
				box = std::move(other.box);
				element_mask = std::move(other.element_mask);
				unstructured = std::move(other.unstructured);
			}
			return *this;
		}

		StructuredVoxelMesh(const Box_t& box, uint64_t depth) noexcept : depth{depth}, box{box} {unstructured.box = box;}


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Helpful simple queries and routines to set the mask
		//////////////////////////////////////////////////////////////////////////////////////////////////
	private:
		GUTIL_DECLARE_SIMD(uniform(vec))
		[[nodiscard]] static constexpr bool is_active_impl(uint64_t idx, const unsigned char* vec) noexcept {
			return vec[idx]&ACTIVE_BIT;
		}

		GUTIL_DECLARE_SIMD(uniform(vec))
		static constexpr void set_active_impl(uint64_t idx, unsigned char* vec, bool val) noexcept {
			val ? vec[idx]|=ACTIVE_BIT : vec[idx]&=~ACTIVE_BIT;
		}

		GUTIL_DECLARE_SIMD(uniform(vec))
		[[nodiscard]] static constexpr unsigned char read_depth_impl(uint64_t idx, const unsigned char* vec) noexcept {
			return (vec[idx]&DEPTH_MASK) >> 1;
		}

		GUTIL_DECLARE_SIMD(uniform(vec))
		static constexpr void set_depth_impl(uint64_t idx, unsigned char* vec, unsigned char val) noexcept {
			vec[idx]&=~DEPTH_MASK;
			vec[idx]|= (val<<1)&DEPTH_MASK;
		}
	public:

		[[nodiscard]] bool is_active(VoxelElement el) const noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			return is_active_impl(el.depth_linear_index(), &element_mask[0]);
		}

		[[nodiscard]] unsigned char read_depth(VoxelElement el) const noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			return read_depth_impl(el.depth_linear_index(), &element_mask[0]);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_active(uint64_t idx) const noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			return is_active_impl(idx, &element_mask[0]);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] unsigned char read_depth(uint64_t idx) const noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			return (element_mask[idx]&DEPTH_MASK) >> 1;
		}

		void set_active(VoxelElement el, bool val) noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			set_active_impl(el.depth_linear_index(), &element_mask[0], val);
		}

		void set_depth(VoxelElement el, unsigned char val) noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			set_depth_impl(el.depth_linear_index(), &element_mask[0], val);
		}

		GUTIL_DECLARE_SIMD()
		void set_active(uint64_t idx, bool val) noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			set_active_impl(idx, &element_mask[0], val);
		}

		GUTIL_DECLARE_SIMD()
		void set_depth(uint64_t idx, unsigned char val) noexcept {
			GUTIL_ASSERT(element_mask.size() == max_elements());
			set_depth_impl(idx, &element_mask[0], val);
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Convert to/from an unstructured mesh for a single layer
		//////////////////////////////////////////////////////////////////////////////////////////////////
		void update_unstructured() noexcept {
			unstructured.clear();
			for (uint64_t i=0; i<max_elements(); ++i) {
				const VoxelElement el(depth,i);
				if (is_active(el)) { unstructured.push_back(el); }
			}
		}
		
		void update_unstructured_omp() noexcept {
			if (max_omp_threads==0) {
				update_unstructured();
				return;
			}
			
			//let every thread work on its own section, then collect the active elements
			std::vector<std::vector<VoxelElement>> thread_elements(max_omp_threads);
			for_each_index_omp( [&](uint64_t idx, int tid) {
				if (is_active(idx)) { thread_elements[tid].emplace_back(depth, idx);}
			});

			//collect elements
			unstructured.clear();
			for (int tid=0; tid<max_omp_threads; ++tid) {
				unstructured.elements.insert( unstructured.elements.end(),
									std::make_move_iterator(thread_elements[tid].begin()),
									std::make_move_iterator(thread_elements[tid].end()));
			}
		}

		void collect_vertices_omp() noexcept {
			if (max_omp_threads==0) {
				collect_vertices();
				return;
			}

			//let every thread work on its own section, then collect the vertices
			std::vector<std::vector<VoxelVertex>> thread_vertices(max_omp_threads);
			for_each_element_omp( [&](VoxelElement el, int tid) {
				if (is_active(el)) {
					for (VoxelVertex vtx : el.vertices()) {
						thread_vertices[tid].push_back(vtx.reduced_key());
					}
				}
			});

			//collect vertices
			unstructured.vertices.clear();
			for (int tid=0; tid<max_omp_threads; ++tid) {
				unstructured.vertices.insert( unstructured.vertices.end(),
									std::make_move_iterator(thread_vertices[tid].begin()),
									std::make_move_iterator(thread_vertices[tid].end()));
			}

			//ensure that there are no duplicates
			std::sort(unstructured.vertices.begin(), unstructured.vertices.end());
				auto last = std::unique(unstructured.vertices.begin(), unstructured.vertices.end());
				unstructured.vertices.erase(last, unstructured.vertices.end());
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Set the mask.
		//////////////////////////////////////////////////////////////////////////////////////////////////
		template<typename Predicate, typename... Args>
		void set_mask(Predicate&& pred, Args&&... args) noexcept {
			unstructured.clear();
			init_element_mask();

			for_each_index( [&](uint64_t i) {
				element_mask[i] = pred(VoxelElement{depth,i}, std::forward<Args>(args)...) ? 
						element_mask[i]|ACTIVE_BIT : element_mask[i]&~ACTIVE_BIT;
			});
		}

		template<typename Predicate, typename... Args>
		void set_mask_omp(Predicate&& pred, Args&&... args) noexcept {
			unstructured.clear();
			init_element_mask();

			for_each_index_omp( [&](uint64_t i) {
				element_mask[i] = pred(VoxelElement{depth,i}, std::forward<Args>(args)...) ? 
						element_mask[i]|ACTIVE_BIT : element_mask[i]&~ACTIVE_BIT;
			});
		}

		void set_mask(unsigned char val) noexcept {
			unstructured.clear();
			element_mask.resize(max_elements());
			std::fill(element_mask.begin(), element_mask.end(), val);
		}

		void set_all_active(bool val) noexcept {
			unstructured.clear();
			element_mask.resize(max_elements());
			GUTIL_SIMD()
			for (uint64_t i=0; i<max_elements(); ++i) {
				set_active(i,val);
			}
		}

		void set_all_depth(unsigned char val) noexcept {
			unstructured.clear();
			element_mask.resize(max_elements());
			GUTIL_SIMD()
			for (uint64_t i=0; i<max_elements(); ++i) {
				set_depth(i,val);
			}
		}

		void set_active(std::span<const VoxelElement> list, bool val) noexcept {
			for (uint64_t i=0; i<list.size(); ++i) { 
				set_active(list[i], val);
			}
		}


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Fill out the standard mesh interface
		//////////////////////////////////////////////////////////////////////////////////////////////////
		void color() noexcept { unstructured.color(); }
		void collect_vertices() noexcept { unstructured.collect_vertices(); }
		[[nodiscard]] uint64_t vertex_index(VoxelVertex vtx) const noexcept { return unstructured.vertex_index(vtx); }
		[[nodiscard]] GeoPoint_t geo_coord(VoxelVertex vtx) const noexcept {
			{return box.low + (box.high-box.low)*vtx.normalized_coordinate();}
		}

		[[nodiscard]] uint64_t n_elements() const noexcept { return unstructured.n_elements(); }
		[[nodiscard]] uint64_t n_vertices() const noexcept { return unstructured.n_vertices(); }

		auto element_begin() const noexcept { return unstructured.element_begin(); }
		auto element_end() 	 const noexcept { return unstructured.element_end();   }
		auto vertex_begin()  const noexcept { return unstructured.vertex_begin();  }
		auto vertex_end()    const noexcept { return unstructured.vertex_end();    }

		auto element_begin() noexcept { return unstructured.element_begin(); }
		auto element_end()   noexcept { return unstructured.element_end();   }
		auto vertex_begin()  noexcept { return unstructured.vertex_begin();  }
		auto vertex_end()    noexcept { return unstructured.vertex_end();    }


		//////////////////////////////////////////////////////////////////////////////////////////////////
		/// Loop over every possible element. To loop over active elements only, use the unstructured mesh.
		/// Note Action is allowed to edit the mesh if the caller can capture a non-const reference.
		///
		/// Looping over elements in an ADI fashion is possible, but should be explicitly written.
		//////////////////////////////////////////////////////////////////////////////////////////////////
		#ifdef _OPENMP
		template<typename Action, typename... Args>
		void for_each_element_omp(Action&& action, Args&&... args) const noexcept {
			GUTIL_ASSERT(max_omp_threads>0);
			GUTIL_OMP(parallel num_threads(max_omp_threads))
			{
				const uint64_t tid = static_cast<uint64_t>(omp_get_thread_num());
				const uint64_t n_per_thread = n_elements()/max_omp_threads;
				const uint64_t start = tid*n_per_thread;
				const uint64_t end = (tid==max_omp_threads-1) ? n_elements() : start + n_per_thread;

				for (uint64_t idx=start; idx<end; ++idx) {
					if constexpr (std::is_invocable_v<Action, VoxelElement, int, Args...>) {
						action(unstructured.elements[idx], tid, std::forward<Args>(args)...);
					}
					else {
						action(unstructured.elements[idx], std::forward<Args>(args)...);
					}
				}
			}
		}

		template<typename Action, typename... Args>
		void for_each_index_omp(Action&& action, Args&&... args) const noexcept {
			GUTIL_ASSERT(max_omp_threads>0);
			GUTIL_OMP(parallel num_threads(max_omp_threads))
			{
				const uint64_t tid = static_cast<uint64_t>(omp_get_thread_num());
				const uint64_t n_per_thread = max_elements()/max_omp_threads;
				const uint64_t start = tid*n_per_thread;
				const uint64_t end = (tid==max_omp_threads-1) ? max_elements() : start + n_per_thread;

				for (uint64_t idx=start; idx<end; ++idx) {
					if constexpr (std::is_invocable_v<Action, uint64_t, int, Args...>) {
						action(idx, tid, std::forward<Args>(args)...);
					}
					else {
						action(idx, std::forward<Args>(args)...);
					}
				}
			}
		}

		template<typename Action, typename... Args>
		void for_each_element_simd(Action&& action, Args&&...args) const noexcept {
			GUTIL_SIMD()
			for (uint64_t i=0; i<n_elements(); ++i) {
				action(unstructured.elements[i], std::forward<Args>(args)...);
			}
		}

		template<typename Action, typename... Args>
		void for_each_index_simd(Action&& action, Args&&... args) const noexcept {
			GUTIL_SIMD()
			for (uint64_t i=0; i<max_elements(); ++i) {
				action(i, std::forward<Args>(args)...);
			}
		}
		#else

		template<typename Action, typename... Args>
		void for_each_element_omp(Action&& action, Args&&... args) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			if constexpr (std::is_invocable_v<Action, int, Args...>) {
				for_each_element(std::forward<Action>(action), 0, std::forward<Args>(args)...);
			}
			else {
				for_each_element(std::forward<Action>(action), std::forward<Args>(args)...);
			}
		}

		template<typename Action, typename... Args>
		void for_each_element_simd(Action&& action, Args&&... args) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			for_each_element(std::forward<Action>(action), std::forward<Args>(args)...);
		}

		template<typename Action, typename... Args>
		void for_each_index_omp(Action&& action, Args&&... args) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			if constexpr (std::is_invocable_v<Action, int, Args...>) {
				for_each_index(std::forward<Action>(action), 0, std::forward<Args>(args)...);
			}
			else {
				for_each_index(std::forward<Action>(action), std::forward<Args>(args)...);
			}
		}

		template<typename Action, typename... Args>
		void for_each_index_simd(Action&& action, Args&&... args) const noexcept {
			gutil::Logger::error("Not compiled with OpenMP. Did you forget -fopenmp?");
			for_each_index(std::forward<Action>(action), std::forward<Args>(args)...);
		}
		#endif


		template<typename Action, typename... Args>
		void for_each_element(Action&& action, Args&&... args) const noexcept {
			for (uint64_t i=0; i<n_elements(); ++i) {
				action(unstructured.elements[i], std::forward<Args>(args)...);
			}
		}

		template<typename Action, typename... Args>
		void for_each_index(Action&& action, Args&&... args) const noexcept {
			for (uint64_t i=0; i<max_elements(); ++i) {
				action(i, std::forward<Args>(args)...);
			}
		}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods to write to a VTK file
		/////////////////////////////////////////////////////////////////////////////////////////////////
		void save_as_ascii(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,true>(filename, *this, description);}
		void save_as_binary(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,false>(filename, *this, description);}

		template<typename... Lookup_ts>
		inline void append_cell_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_cell_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		inline void append_cell_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_cell_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		inline void append_point_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_point_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		inline void append_point_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_point_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}
	};
}