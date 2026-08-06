#pragma once

#include "gutil.hpp"

#include "mesh/keys/voxel_key.hpp"
#include "mesh/vtk_file_io.hpp"
#include "mesh/voxel_mesh.hpp"
#include "util/concepts.hpp"
#include "util/macros.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <array>
#include <span>
#include <bitset>

#include <iostream>
#include <sstream>
#include <fstream>



namespace GV {
	//forward declare iterator classes
	template<uint64_t MAX_DEPTH, typename Element_t, bool CONST_FLAG>
	struct IteratorBase;

	template<uint64_t MAX_DEPTH_=10>
	struct UnstructuredVoxelMesh {
		//mesh features are never periodic
		static constexpr uint64_t MAX_DEPTH = MAX_DEPTH_;
		using VoxelElement = VoxelElementKey<MAX_DEPTH,0>;
		using VoxelVertex  = VoxelVertexKey<MAX_DEPTH,0>;
		using VoxelFace    = VoxelFaceKey<MAX_DEPTH,0>;
		using Mesh_t       = UnstructuredVoxelMesh<MAX_DEPTH>; //this mesh type
		using GeoPoint_t   = gutil::Point<3,double>;

		//random access iterator class to loop through the elements
		//this wraps the individual vector iterators but wraps to the next depth if possible
		using Iterator  = IteratorBase<MAX_DEPTH,VoxelElement,false>;
		using CIterator = IteratorBase<MAX_DEPTH,VoxelElement,true>;
	
		//check if OPENMP is enabled
		#ifdef _OPENMP
		static constexpr bool OPENMP = true;
		#else
		static constexpr bool OPENMP = false;
		#endif
	private:
		//allocate storage:
		// one vector of active elements per possible depth
		// each vector is maintained so that the elements are sorted
		// by default, vectors are sorted by their linear index first, but
		// in some circumstances it is good to have elements sorted by color first then by linear index
		//Note that the term "linear index" is the index of the element in the largest feasible mesh (i.e., an octree mesh with depth MAX_DEPTH)
		//The linear index determines the vertices. The location that the element is stored in the allocated vectors determines its index within the 
		//current mesh.
		std::array<std::vector<VoxelElement>,MAX_DEPTH+1> elements;

		//In addition to tracking each of the active elements as element objects (so that color can be tracket
		//and elements looped over in a contiguous manner), we keep a bitset of all possible elements over all depths
		//to make querries easier. This bitset tracks which elements are active by their linear index.
		static constexpr uint64_t TOTAL_POSSIBLE_ELEMENTS = total_possible<VoxelElement>(MAX_DEPTH);
		std::bitset<TOTAL_POSSIBLE_ELEMENTS>* active_elem = new std::bitset<TOTAL_POSSIBLE_ELEMENTS>(0);

		//for saving solutions, it is nice to have access to de-duplicated vertices
		//to get the index of a vertex on an element, use element.vertex(k).reduced_key() to get the coarsest vertex at that location.
		//this will be sorted by vertex linear index for easier lookup and consistency
		//marked as mutable so that classes with a const reference to the mesh can still save data.
		mutable std::vector<VoxelVertex> vertices;

		//comparators to help sort elements
		static inline bool compare_index(const VoxelElement left, const VoxelElement right) {return left<right;}
		static bool compare_color(const VoxelElement left, const VoxelElement right) {
			if (left.color() < right.color()) {return true;}
			if (left.color() == right.color()) {return compare_index(left,right);} //call the compare_index for consistency
			return false;
		}

		//various flags for sanity checks
		mutable bool _vertices_found_ = false;
		bool _sort_by_index_  = true;
		bool _sort_by_color_  = false;
		bool _colored_		  = false;
		uint64_t _n_colors_   = 0;

		//store first index of each color (and one past end) for easier looping by color
		std::array<std::vector<uint64_t>,MAX_DEPTH+1> color_block_index;

		//store a list of elements that classes with a const reference to the mesh can use to request
		//element refinement or unrefinement
		//TODO: we could make a vector per omp thread
		mutable std::vector<VoxelElement> request_active;
		mutable std::vector<VoxelElement> request_deactive;
	public:
		//store the extents of the mesh
		const GeoPoint_t low;
		const GeoPoint_t high;

		//////////////////////////////////////////////////////////
		/// Constructors
		UnstructuredVoxelMesh() : low{0,0,0}, high{1,1,1} {}
		UnstructuredVoxelMesh(const GeoPoint_t& low_, const GeoPoint_t& high_) : low(low_), high(high_) {}
		explicit UnstructuredVoxelMesh(const VoxelMesh<MAX_DEPTH>& structured) : low(structured.low), high(structured.high) {
			auto get_active = [&](VoxelElement el) {
				if (structured.is_active(el)) {
					elements[el.depth()].push_back(el);
				}
			};

			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for (uint64_t dd=0; dd<=MAX_DEPTH; ++dd) {
				structured.template for_each_depth<VoxelElement>(dd, get_active);
			}

			//each loop was in increasing linear index, so the elements are already sorted by index
			_sort_by_index_ = true;
		}

		// explicit UnstructuredVoxelMesh(const StructuredVoxelMesh<MAX_DEPTH>& structured) : low(structured.low), high(structured.high) {
		// 	for (uint64_t i=0; i<structured.TOTAL_POSSIBLE_ELEMENTS; ++i) {
		// 		VoxelElement el(structured.DEPTH, i);
		// 		if (structured.is_active(el)) {
		// 			elements[el.depth()].push_back(el);
		// 		}
		// 	}

		// 	//each loop was in increasing linear index, so the elements are already sorted by index
		// 	_sort_by_index_ = true;
		// }
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Destructor
		~UnstructuredVoxelMesh() {delete active_elem;}
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Methods primarily for simple queries
		inline uint64_t n_colors() const {return _n_colors_;}
		inline bool is_colored() const {return _colored_;}
		inline bool is_color_sorted() const { if (_sort_by_color_) {assert(_colored_);} return _sort_by_color_;}
		inline bool is_index_sorted() const {return _sort_by_index_;}
		inline uint64_t n_vertices() const {assert(_vertices_found_); return vertices.size();}
		
		uint64_t n_elements_below(const uint64_t depth) const {
			uint64_t count=0;
			for (uint64_t dd=0; dd<std::min(depth,MAX_DEPTH+1); ++dd) {count+=elements[dd].size();}
			return count;
		}
		inline uint64_t n_elements() const {return n_elements_below(MAX_DEPTH+1);}
		inline GeoPoint_t geo_coord(const VoxelVertex vtx) const {return low + (high-low)*vtx.normalized_coordinate();}
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Methods primarily for accessing data arbitrarily and some standard container interfaces
		inline size_t size() const {return static_cast<size_t>(n_elements());}
		void clear() {
			active_elem->reset();
			vertices.clear();
			for (auto& list : elements) {list.clear();}
			_vertices_found_ = false;
			_sort_by_index_  = true;
			_colored_        = false;
			_sort_by_color_  = false;
		}
		void shrink_to_fit() {{for (auto& list : elements) {list.shrink_to_fit();}}}

		//methods to find or access elements
		uint64_t find_element(const VoxelElement el) const;
		uint64_t find_colored_element(const VoxelElement el) const;
		inline VoxelElement  get_element(const uint64_t idx) const {assert(idx<n_elements()); return *celement(idx);}
		inline VoxelElement& get_element(const uint64_t idx) {assert(idx<n_elements()); return *element(idx);};
		CIterator get_element(const VoxelElement el) const;
		Iterator get_element(const VoxelElement el);

		//local topolgy querries
		void collect_neighbors(std::vector<VoxelElement>& nbrs, const VoxelElement el) const;

		//check if elements are active (present in the mesh)
		inline bool is_active(const VoxelElement el) const {return is_active_impl<VoxelElement>(el);}
		inline bool is_active(const VoxelVertex vtx) const {return is_active_impl<VoxelVertex>(vtx);}
		inline bool is_active(const VoxelFace face) const {return is_active_impl<VoxelFace>(face);}
		bool is_any_active(std::span<const VoxelElement> els) const {
			bool result=false;
			for (const auto el : els) {result |= is_active(el);}
			return result;
		}


		//check if features are conformal or hanging
		//a feature is hanging if it has a corresponding element that exists but is not active
		inline bool is_hanging(const VoxelVertex vtx) const {return is_hanging_impl<VoxelVertex>(vtx);}
		inline bool is_hanging(const VoxelFace face) const {return is_hanging_impl<VoxelFace>(face);}

		//get iterators to start of elements
		inline Iterator  begin() {return Iterator::begin(elements);}
		inline CIterator cbegin() const {return CIterator::begin(elements);}
		inline CIterator begin() const {return cbegin();}

		//get iterators to arbitrary elements
		inline Iterator  element(const uint64_t idx) {return Iterator{elements, idx};}
		inline CIterator celement(const uint64_t idx) const {return CIterator{elements, idx};}
		inline CIterator element(const uint64_t idx) const {return celement(idx);}

		//get iterators to end of elements
		inline Iterator  end() {return Iterator::end(elements);}
		inline CIterator cend() const {return CIterator::end(elements);}
		inline CIterator end() const {return cend();}

		//get iterators to start of depth
		inline auto depth_begin(const uint64_t dd) {assert(dd<=MAX_DEPTH); return elements[dd].begin();}
		inline auto depth_cbegin(const uint64_t dd) const {assert(dd<=MAX_DEPTH); return elements[dd].cbegin();}
		inline auto depth_begin(const uint64_t dd) const {return depth_cbegin(dd);}

		//get iterators to end of depth
		inline auto depth_end(const uint64_t dd) {assert(dd<=MAX_DEPTH); return elements[dd].end();}
		inline auto depth_cend(const uint64_t dd) const {assert(dd<=MAX_DEPTH); return elements[dd].cend();}
		inline auto depth_end(const uint64_t dd) const {return depth_cend(dd);}
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Methods primarily for mesh coloring and parallelism 
		//sort the elements by linear index or color
		void sort_by_color();
		void sort_by_index();

		void color() {};
		void color_by_index();

		//get the span of elements of a given color at a given depth
		//note that the elements must be sorted by color
		inline std::span<const VoxelElement> color_block(const uint64_t dd, const uint64_t clr) const {
			assert(dd<=MAX_DEPTH);
			assert(clr+1 < color_block_index[dd].size());
			assert(_sort_by_color_);
			const uint64_t start = color_block_index[dd][clr];
			const uint64_t end   = color_block_index[dd][clr+1];
			return {elements[dd].data()+start, end-start};
		}
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Methods primarily for writing to vtk files
		void collect_vertices() const;
		uint64_t vertex_index(VoxelVertex vtx) const {
			auto it = std::lower_bound(vertices.begin(), vertices.end(), vtx);
			if (it == vertices.end() || *it!=vtx) {return uint64_t(-1);}
			return std::distance(vertices.begin(), it);
		}

		//get elements and vertices for easier writing to vtk files
		inline CIterator element_begin() const {return cbegin();}
		inline CIterator element_end() const {return cend();}
		inline auto vertex_begin() const {return vertices.cbegin();}
		inline auto vertex_end() const {return vertices.cend();}
		inline std::span<const VoxelVertex> get_vertices() const {return vertices;}
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
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Methods primarily for mesh manipulation
		template<typename Predicate>
		void remove_elements(Predicate&& pred) const requires (OPENMP);

		template<typename Predicate>
		void remove_elements(Predicate&& pred) const requires (!OPENMP);

		void make_disjoint();
		void set_depth(const uint64_t dd);
		inline void activate(const VoxelElement el) const {request_active.push_back(el);}
		inline void deactivate(const VoxelElement el) const {request_deactive.push_back(el);}
		inline void activate(std::span<const VoxelElement> els) const {request_active.insert(request_active.end(), els.begin(), els.end());}
		inline void deactivate(std::span<const VoxelElement> els) const {request_deactive.insert(request_deactive.end(), els.begin(), els.end());}
		void process_requests();
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Methods for looping over active elements
		template<typename Action>
		void for_each_active_element(Action&& action);
		template<typename Action>
		void for_each_active_element(Action&& action) const;
		template<typename Action>
		void for_each_active_element_omp(Action&& action) requires OPENMP;
		template<typename Action>
		void for_each_active_element_omp(Action&& action) const requires OPENMP;
		template<typename Action>
		void for_each_active_element_color_omp(Action&& action) const requires OPENMP;
		template<typename Action>
		void for_each_active_element_color_omp(const uint64_t clr, Action&& action) const requires OPENMP;
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Methods for looping over all possible features
		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> 
		inline void for_each_depth(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const requires (MeshFeatureType<Key_t,Mesh_t>) {
			for_each_depth_impl<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> 
		inline void for_each_depth_omp(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const requires (MeshFeatureType<Key_t,Mesh_t> && OPENMP) {
			for_each_depth_impl_omp<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
		}
		//////////////////////////////////////////////////////////

	private:
		//////////////////////////////////////////////////////////
		/// Some generic feature implementation methods
		template<typename Key_t> 
		bool is_active_impl(const Key_t key) const requires (MeshFeatureType<Key_t,Mesh_t>);
		template<typename Key_t> 
		bool is_hanging_impl(const Key_t key) const requires (MeshFeatureType<Key_t,Mesh_t>);
		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t>
		void for_each_depth_impl(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const requires (MeshFeatureType<Key_t,Mesh_t>);
		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t>
		void for_each_depth_impl_omp(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const requires (MeshFeatureType<Key_t,Mesh_t> && OPENMP);
		//////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////
		/// Some convenient manipulation tools
		inline void set_bitset_active(const uint64_t start, const uint64_t end);
		//////////////////////////////////////////////////////////
	};



	template<uint64_t MAX_DEPTH>
	template<typename Key_t>
	bool UnstructuredVoxelMesh<MAX_DEPTH>::is_hanging_impl(const Key_t key) const requires (MeshFeatureType<Key_t,Mesh_t>) {
		assert(key.is_valid());
		assert(is_active(key));
		for (const VoxelElement el : key.elements()) {
			if (el.exists() && !is_active(el)) {return true;}
		}
		return false;
	}

	template<uint64_t MAX_DEPTH>
	template<typename Key_t> 
	bool UnstructuredVoxelMesh<MAX_DEPTH>::is_active_impl(const Key_t key) const requires (MeshFeatureType<Key_t,Mesh_t>) {
		assert(key.is_valid());
		if constexpr (std::same_as<Key_t,VoxelElement>) {return active_elem->test(key.linear_index());}
		else {
			for (const VoxelElement el : key.elements()) {
				if (el.exists() && active_elem->test(el.linear_index)) {return true;}
			}
			return false;
		}
	}

	template<uint64_t MAX_DEPTH>
	uint64_t UnstructuredVoxelMesh<MAX_DEPTH>::find_element(const VoxelElement el) const {
		auto& list = elements[el.depth()];
		if (is_index_sorted()) {
			auto it = std::lower_bound(list.begin(), list.end(), el);
			if (it == list.end() || *it != el) {return uint64_t(-1);}
			return n_elements_below(el.depth()) + std::distance(list.begin(), it);
		}
		else if (is_color_sorted()) {
			return find_colored_element(el);
		}
		else {
			auto it = std::find(list.begin(), list.end(), el);
			if (it==list.end()) {return uint64_t(-1);}
			return n_elements_below(el.depth()) + std::distance(list.begin(), it);
		}
	}

	template<uint64_t MAX_DEPTH>
	uint64_t UnstructuredVoxelMesh<MAX_DEPTH>::find_colored_element(const VoxelElement el) const {
		assert(is_color_sorted());
		auto block = color_block(el.depth(), el.color());
		auto it = std::lower_bound(block.begin(), block.end(), el);
		if (it == block.end() || *it != el) {return uint64_t(-1);}
		return n_elements_below(el.depth()) + color_block_index[el.depth()][el.color()] + std::distance(block.begin(), it);
	}

	template<uint64_t MAX_DEPTH>
	typename UnstructuredVoxelMesh<MAX_DEPTH>::CIterator UnstructuredVoxelMesh<MAX_DEPTH>::get_element(const VoxelElement el) const {
		//get the actual stored element (including color data) given it's location in the hypothetical mesh
		//return the defualt (does not exist) element if the element is not active
		const uint64_t idx = find_element(el);
		if (idx != uint64_t(-1)) {return celement(idx);}
		return cend();
	}

	template<uint64_t MAX_DEPTH>
	typename UnstructuredVoxelMesh<MAX_DEPTH>::Iterator UnstructuredVoxelMesh<MAX_DEPTH>::get_element(const VoxelElement el) {
		//get the actual stored element (including color data) given it's location in the hypothetical mesh
		//return the defualt (does not exist) element if the element is not active
		const uint64_t idx = find_element(el);
		if (idx != uint64_t(-1)) {return element(idx);}
		return end();
	}

	template<uint64_t MAX_DEPTH>
	template<typename Action>
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_active_element_color_omp(Action&& action) const requires OPENMP {
		assert(is_colored());
		assert(is_color_sorted());
		//process each color up to n_colors
		for (uint64_t clr=0; clr<n_colors(); ++clr) {
			for_each_active_element_color_omp(clr, std::forward<Action>(action));
		}
	}

	template<uint64_t MAX_DEPTH>
	template<typename Action>
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_active_element_color_omp(const uint64_t clr, Action&& action) const requires OPENMP {
		assert(is_colored());
		assert(is_color_sorted());
		assert(clr<n_colors());

		//process each color block per depth
		#pragma omp parallel if(!omp_in_parallel())
		{
			for (uint64_t dd=0; dd<=MAX_DEPTH; ++dd) {
				const auto block = color_block(dd, clr);
				const uint64_t N = block.size();
				#pragma omp for schedule(static, 512)
				for (uint64_t i=0; i<N; ++i) {
					action(block[i]);
				}
			}
		}
	}

	template<uint64_t MAX_DEPTH> //TODO: add a predicate for activation
	void UnstructuredVoxelMesh<MAX_DEPTH>::process_requests() {
		if (request_active.empty() && request_deactive.empty()) {return;}

		//if we change the structure, we will have to re-compute vertices
		vertices.clear();
		_vertices_found_ = false;
		_sort_by_index_  = false;
		_sort_by_color_  = false;

		//clean up request lists
		std::sort(request_active.begin(), request_active.end());
		auto last = std::unique(request_active.begin(), request_active.end());
		request_active.erase(last, request_active.end());
		std::erase_if(request_active, [&](VoxelElement el){return is_active(el);});

		std::sort(request_deactive.begin(), request_deactive.end());
		last = std::unique(request_deactive.begin(), request_deactive.end());
		request_deactive.erase(last, request_deactive.end());
		std::erase_if(request_deactive, [&](VoxelElement el){return !is_active(el);});

		//update the bitset
		for (VoxelElement el : request_active) {active_elem->set(el.linear_index(),true);}
		for (VoxelElement el : request_deactive) {active_elem->set(el.linear_index(),false);}

		//add the requested elements
		//TODO: the elements to insert are sorted in blocks by depth (lowest to highest)
		//insert each block at once.
		for (VoxelElement el : request_active) {
			elements[el.depth()].push_back(el);
		}
		request_active.clear();

		//to remove elements, the mesh must be sorted by index
		sort_by_index();

		GEOVOX_OMP(parallel for)
		for (uint64_t dd=0; dd<MAX_DEPTH; ++dd) {
			const VoxelElement first_element{dd,0};
			const VoxelElement last_element{dd+1,0};
			auto rem_begin = std::lower_bound(request_deactive.begin(), request_deactive.end(), first_element);
			auto rem_end   = std::lower_bound(request_deactive.begin(), request_deactive.end(), last_element);
			
			if (rem_begin==rem_end) {continue;}

			auto& list = elements[dd];
			std::vector<VoxelElement> new_list;
			new_list.reserve(list.size());
			std::set_difference(list.begin(), list.end(), rem_begin, rem_end, std::back_inserter(new_list));
			list = std::move(new_list);
		}
		request_deactive.clear();

		make_disjoint();
	}

	template<uint64_t MAX_DEPTH> //TODO: add a predicate for activation
	void UnstructuredVoxelMesh<MAX_DEPTH>::make_disjoint() {
		for (uint64_t dd=0; dd<MAX_DEPTH; ++dd) {
			const auto& list = elements[dd];
			for (const VoxelElement el : list) {
				const auto children = el.children();
				if (is_any_active(children)) {
					deactivate(el);
					for (const auto child : children) {
						activate(child);
					}
				}
			}
		}

		//the process_requests -> make_disjoint may recurse one or two times.
		process_requests();
	}

	template<uint64_t MAX_DEPTH>
	template<typename Predicate>
	void UnstructuredVoxelMesh<MAX_DEPTH>::remove_elements(Predicate&& pred) const requires (!OPENMP) {
		auto action = [&](VoxelElement el) {if (pred(el)) {deactivate(el);}};
		for_each_active_element(action);
	}

	template<uint64_t MAX_DEPTH>
	template<typename Predicate>
	void UnstructuredVoxelMesh<MAX_DEPTH>::remove_elements(Predicate&& pred) const requires (OPENMP) {
		const uint64_t omp_threads = omp_get_max_threads();
		std::vector<std::vector<VoxelElement>> remove_lists(omp_threads);
		auto action = [&](VoxelElement el) {if (pred(el)) {remove_lists[omp_get_thread_num()].push_back(el);}};
		for_each_active_element_omp(action);
		for (auto& list : remove_lists) {deactivate(list);}
	}

	template<uint64_t MAX_DEPTH>
	template<typename Action>
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_active_element(Action&& action) {
		for (Iterator it=begin(); it!=end(); ++it){
			action(*it);
		}
	}

	template<uint64_t MAX_DEPTH>
	template<typename Action>
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_active_element(Action&& action) const {
		for (CIterator it=cbegin(); it!=cend(); ++it){
			action(*it);
		}
	}

	template<uint64_t MAX_DEPTH>
	template<typename Action>
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_active_element_omp(Action&& action) requires OPENMP {
		const uint64_t N_ELS      = n_elements();
		const uint64_t CHUNK_SIZE = 512; //number of elements to process at once
		
		#pragma omp parallel for schedule(static)
		for (uint64_t start=0; start<N_ELS; start+=CHUNK_SIZE)
		{
			const uint64_t end = std::min(start+CHUNK_SIZE, N_ELS);
			Iterator it = element(start);
			for (uint64_t i=start; i<end; ++i) {
				action(*it);
				++it;
			}
		}
	}

	template<uint64_t MAX_DEPTH>
	template<typename Action>
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_active_element_omp(Action&& action) const requires OPENMP {
		const uint64_t N_ELS      = n_elements();
		const uint64_t CHUNK_SIZE = 512; //number of elements to process at once
		
		#pragma omp parallel for schedule(static)
		for (uint64_t start=0; start<N_ELS; start+=CHUNK_SIZE)
		{
			const uint64_t end = std::min(start+CHUNK_SIZE, N_ELS);
			CIterator it = celement(start);
			for (uint64_t i=start; i<end; ++i) {
				action(*it);
				++it;
			}
		}
	}


	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::set_depth(const uint64_t dd) {
		clear();

		const uint64_t N = uint64_t{1}<<(3*dd); //2^dd elements per side, 3d array
		elements[dd].resize(N); 

		#pragma omp simd
		for (uint64_t i=0; i<N; ++i) {
			elements[dd][i] = VoxelElement{dd,i};
		}

		set_bitset_active(VoxelElement::depth_linear_start(dd), VoxelElement::depth_linear_start(dd+1));
	}


	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::sort_by_index() {
		GEOVOX_OMP(parallel for)
		for (uint64_t k=0; k<=MAX_DEPTH; ++k) {
			auto& list = elements[k];
			std::sort(list.begin(), list.end(), compare_index);
		}

		_sort_by_index_ = true;
		_sort_by_color_ = false;
	}

	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::sort_by_color() {
		assert(is_colored() && "call color_mesh before sorting by color");
		if(is_color_sorted()) {return;}

		//sort by color
		uint64_t max_color = 0;
		#ifdef _OPENMP
		#pragma omp parallel for reduction(max:max_color)
		#endif
		for (uint64_t k=0; k<=MAX_DEPTH; ++k) {
			auto& list = elements[k];
			std::sort(list.begin(), list.end(), compare_color);
			if (!list.empty()) {
				max_color = std::max(max_color, list.back().color());
			}
		}

		//update convenient data members
		_sort_by_index_ = false;
		_sort_by_color_ = true;
		_n_colors_      = max_color + 1;

		//set color boundaries
		GEOVOX_OMP(parallel for)
		for (uint64_t k=0; k<=MAX_DEPTH; ++k) {
			const auto& list = elements[k];
			auto& clr_block = color_block_index[k];

			clr_block.assign(_n_colors_+1,0);
			uint64_t clr = 0;
			for (uint64_t idx=0; idx<list.size() && clr<=_n_colors_; ++idx) {
				const uint64_t el_color = list[idx].color();
				while (clr <= el_color) {
					clr_block[clr] = idx;
					++clr;
				}
			}

			//all color blocks were found, the rest are empty
			std::fill(clr_block.begin()+clr, clr_block.end(), list.size());
		}
	}

	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::color_by_index() {
		//assign a color 0-7 based on the parity of i,j,k
		//if the mesh elements are disjoint, this guarantees that no two elements
		//with the same color will intersect at their boundaries.
		for (uint64_t dd=0; dd<=MAX_DEPTH; ++dd) {
			#pragma omp simd
			for (uint64_t i=0; i<elements[dd].size(); ++i) {
				VoxelElement& el = elements[dd][i];
				const uint64_t clr = (el.depth()*8) + ((el.i()&1) | ((el.j()&1)<<1) | ((el.k()&1)<<2));
				el.set_color(clr);
			}
		}
		_colored_  = true;
	}

	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::collect_neighbors(std::vector<VoxelElement>& nbrs, const VoxelElement el) const {
		assert(is_active(el));

		//an element is a neighbor if it is an active neighbor at the same depth
		//or an active ancestor of any neigbor (even inactive) at that depth
		//note that if element A is at a coarser/lower depth than element B, then A may be a neighbor of B, but B cannot be a neigbor of A
		//so long as we color from the top to the bottom, this is ok.
		for (VoxelElement nbr : el.neighbors()) {
			if (is_active(nbr)) {
				nbrs.push_back(nbr);
				continue;
			}

			//see if nbr has an active ancestor
			while (nbr.depth() > 0) {
				nbr = nbr.parent();
				if (is_active(nbr)) {
					nbrs.push_back(nbr);
					break;
				}
			}
		}
	}

	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::collect_vertices() const {
		vertices.clear();
		for (const VoxelElement el : *this) {
			for (VoxelVertex vtx : el.vertices()) {
				const VoxelVertex r_vtx = vtx.reduced_key();
				vertices.emplace_back(r_vtx);
			}
		}
		std::sort(vertices.begin(), vertices.end());
		auto last = std::unique(vertices.begin(), vertices.end());
		vertices.erase(last, vertices.end());
		_vertices_found_ = true;
	}

	template<uint64_t MAX_DEPTH>
	template<typename Key_t, typename Action, typename Predicate>
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_depth_impl_omp(const uint64_t depth, Action&& action, Predicate&& pred) const 
				requires (MeshFeatureType<Key_t,Mesh_t> && OPENMP) {
		const uint64_t start = Key_t::depth_linear_start(depth);
		const uint64_t end   = Key_t::depth_linear_start(depth+1);
		const uint64_t n     = end - start;
		const uint64_t chunk = 512;
		const uint64_t n_chk = (n+chunk-1) / chunk;

		//each thread gets one contiguous sequence of chunks.
		//for example, thread 0 may get chunks [0,5), thread 1 chunks [5,10), and thread 2 chunks [10,12)
		//with the size of each range of chunks split evenly with the remainder sent to the last thread.
		//action(key,0) will always be called on the lowest index features and action(key,1) the next lowest block, and so on.
		#pragma omp parallel for schedule(static)
		for (uint64_t c=0; c<n_chk; ++c) {
			const int tid = omp_get_thread_num();
			const uint64_t c_start = start + c*chunk;
			const uint64_t c_end   = std::min(c_start+chunk, end);
			
			Key_t key(depth, c_start - start);
			for (uint64_t idx=c_start; idx<c_end; ++idx, ++key) {
				assert(key.linear_index() == idx);
				if constexpr (!NULLPTR_T<Predicate>) {
					if (!pred(key)) {
						continue;
					}
				}
				//if the action can accept the thread id, pass that along
				if constexpr (std::is_invocable_v<Action, Key_t, int>) {
					action(key, tid);
				}
				else {
					action(key);
				}
			}
		}
	}
	
	template<uint64_t MAX_DEPTH>
	template<typename Key_t, typename Action, typename Predicate> 
	void UnstructuredVoxelMesh<MAX_DEPTH>::for_each_depth_impl(const uint64_t depth, Action&& action, Predicate&& pred) const
				requires (MeshFeatureType<Key_t,Mesh_t>) {
		const uint64_t start = Key_t::depth_linear_start(depth);
		const uint64_t end   = Key_t::depth_linear_start(depth+1);
		Key_t key(depth,0); //element/vertex/face object (acts as an iterator)

		for (uint64_t idx=start; idx<end; ++idx, ++key) {
			assert(key.linear_index() == idx);
			if constexpr (!NULLPTR_T<Predicate>) {
				if (!pred(key)) {
					continue;
				}
			}
			action(key);
		}
	}

	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::set_bitset_active(const uint64_t start, const uint64_t end) {
		assert(start<end);
		assert(end<TOTAL_POSSIBLE_ELEMENTS);

		for (uint64_t idx = start; idx<end; ++idx) {
			active_elem->set(idx);
		}
	}

	//Implement the Iterator class over all elements
	template<uint64_t MAX_DEPTH, typename Element_t, bool CONST_FLAG>
	struct IteratorBase	{
		//necessary aliases for the standard library
		using iterator_category = std::forward_iterator_tag;
		using value_type		= Element_t;
		using difference_type	= std::ptrdiff_t;
		using pointer			= std::conditional_t<CONST_FLAG, Element_t const*, Element_t*>;
		using reference			= std::conditional_t<CONST_FLAG, const Element_t&, Element_t&>;

		//convenient aliases
		using container_type	= std::array<std::vector<Element_t>,MAX_DEPTH+1>;
		using container_ref		= std::conditional_t<CONST_FLAG, const container_type&, container_type&>;
		using container_ptr 	= std::conditional_t<CONST_FLAG, container_type const*, container_type*>;

		//current position: elements[depth][idx]
		container_ptr elements;
		uint64_t depth;
		uint64_t idx;
		
		//constructor
		IteratorBase(container_ref els, uint64_t dd, uint64_t ii) : elements(&els), depth(dd), idx(ii) {advance_to_valid();}
		IteratorBase(container_ref els, uint64_t n) : elements(&els), depth(0), idx(0) {(*this)+=n; advance_to_valid();}

		//implicit conversion from non-const to const
		IteratorBase(const IteratorBase<MAX_DEPTH,Element_t,false>& it) requires (CONST_FLAG) 
			: elements(it.elements), depth(it.depth), idx(it.idx) {}

		//begin/end iterators
		static IteratorBase end(container_ref els) {return IteratorBase{els, MAX_DEPTH+1, 0};}
		static IteratorBase begin(container_ref els) {return IteratorBase{els,0,0};}

		//random access operations
		reference operator[](uint64_t n) const {return *(*this+n);}

		//advance to the next valid depth/index pair (if the current is valid, they aren't changed)
		//this should just skip over any empty depths
		void advance_to_valid() {
			while (depth<=MAX_DEPTH && idx>= (*elements)[depth].size()) {
				++depth;
				idx = 0;
			}
		}

		uint64_t total_length() const {
			uint64_t len=0;
			for (const auto& list : *elements) {len+=list.size();}
			return len;
		}

		uint64_t index() const {
			if (depth>MAX_DEPTH) {return total_length();}

			uint64_t el_idx = 0;
			for (uint64_t dd=0; dd<depth; ++dd) {
				el_idx += (*elements)[dd].size();
			}
			el_idx += idx;
			return el_idx;
		}

		//access data
		reference operator*() const {
			assert(depth <= MAX_DEPTH);
			assert(idx < (*elements)[depth].size());
			return (*elements)[depth][idx];
		}

		pointer operator->() const {
			assert(depth <= MAX_DEPTH);
			assert(idx < (*elements)[depth].size());
			return &(*elements)[depth][idx];
		}

		IteratorBase& operator++() {
			++idx;
			advance_to_valid();
			return *this;
		}

		IteratorBase operator++(int) {
			IteratorBase tmp = *this;
			++(*this);
			return tmp;
		}

		IteratorBase& operator+=(uint64_t n) {
			//move to the correct depth and update the increment
			while (n>0 && depth<=MAX_DEPTH) {
				const uint64_t remaining_in_depth = (*elements)[depth].size() - idx;
				if (n<remaining_in_depth) {idx += n; n=0;}
				else {
					n -= remaining_in_depth;
					++depth;
					idx = 0;
				}
			}
			advance_to_valid();
			return *this;
		}

		IteratorBase operator+(uint64_t n) const {
			IteratorBase tmp = *this;
			tmp+=n;
			return tmp;
		}

		friend IteratorBase operator+(uint64_t n, const IteratorBase& it) {
			return it+n;
		}

		IteratorBase& operator-=(uint64_t n) {
			uint64_t flat = index();
			assert(n<=flat);
			*this = IteratorBase(*elements, flat-n);
			return *this;
		}

		IteratorBase operator-(uint64_t n) const {
			IteratorBase tmp = *this;
			tmp-=n;
			return tmp;
		}

		//implement ordering and comparisons
		difference_type operator-(const IteratorBase& other) const {
			return static_cast<difference_type>(index()) - static_cast<difference_type>(other.index());
		}

		bool operator<(const IteratorBase& other) const {return (*this - other) < 0;}
		bool operator>(const IteratorBase& other) const {return (*this - other) > 0;}
		bool operator<=(const IteratorBase& other) const {return (*this - other) <= 0;}
		bool operator>=(const IteratorBase& other) const {return (*this - other) >= 0;}

		bool operator==(const IteratorBase& other) const {
			return depth==other.depth && idx==other.idx;
		}

		bool operator!=(const IteratorBase& other) const {
			return depth!=other.depth || idx!=other.idx;
		}
	};



}