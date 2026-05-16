#pragma once

#include "mesh/keys/voxel_key.hpp"
#include "mesh/vtk_file_io.hpp"
#include "mesh/voxel_mesh.hpp"
#include "util/concepts.hpp"
#include "util/point.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <array>
#include <span>

#include <iostream>
#include <sstream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace GV
{
	//forward declare iterator classes
	template<uint64_t MAX_DEPTH, typename Element_t, bool CONST_FLAG>
	struct IteratorBase;

	template<uint64_t MAX_DEPTH_=10>
	class UnstructuredVoxelMesh
	{
	public:
		//mesh features are never periodic
		static constexpr uint64_t MAX_DEPTH = MAX_DEPTH_;
		using VoxelElement = VoxelElementKey<MAX_DEPTH,0>;
		using VoxelVertex  = VoxelVertexKey<MAX_DEPTH,0>;
		using VoxelFace    = VoxelFaceKey<MAX_DEPTH,0>;
		using Mesh_t       = UnstructuredVoxelMesh<MAX_DEPTH>; //this mesh type

		//random access iterator class to loop through the elements
		//this wraps the individual vector iterators but wraps to the next depth if possible
		using Iterator = IteratorBase<MAX_DEPTH,VoxelElement,false>;
		using CIterator = IteratorBase<MAX_DEPTH,VoxelElement,true>;
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

		//for saving solutions, it is nice to have access to de-duplicated vertices
		//to get the index of a vertex on an element, use element.vertex(k).reduced_key() to get the coarsest vertex at that location.
		//this will be sorted by vertex linear index for easier lookup and consistency
		std::vector<VoxelVertex> vertices;

		//comparators to help sort elements
		static inline bool compare_index(const VoxelElement left, const VoxelElement right) {return left<right;}
		static bool compare_color(const VoxelElement left, const VoxelElement right) {
			if (left.color() < right.color()) {return true;}
			if (left.color() == right.color()) {return compare_index(left,right);} //call the compare_index for consistency
			return false;
		}

		//varous flags for sanity checks
		bool _vertices_found_ = false;
		bool _sort_by_index_  = true;
		bool _sort_by_color_  = false;
		bool _colored_		  = false;
		uint64_t _n_colors_   = 0;

		//store first index of each color (and one past end) for easier looping by color
		std::array<std::vector<uint64_t>,MAX_DEPTH+1> color_block_index;

		//store the extents of the mesh
		Point<3,double> _low_;
		Point<3,double> _high_;
	public:
		//constructors
		UnstructuredVoxelMesh() : _low_{0,0,0}, _high_{1,1,1} {}
		UnstructuredVoxelMesh(const Point<3,double>& low, const Point<3,double>& high) : _low_(low), _high_(high) {}
		explicit UnstructuredVoxelMesh(const VoxelMesh<MAX_DEPTH>& structured) : _low_(structured.low), _high_(structured.high) {
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

		//methods to querry sizes
		inline uint64_t n_colors() const {return _n_colors_;}
		inline bool is_colored() const {return _colored_;}
		inline bool is_color_sorted() const { if (_sort_by_color_) {assert(_colored_);} return _sort_by_color_;}
		inline bool is_index_sorted() const {return _sort_by_index_;}
		inline uint64_t n_vertices() const {assert(_vertices_found_); return vertices.size();}
		uint64_t n_elements() const {
			uint64_t n=0;
			for (const auto& list : elements) {n+=list.size();}
			return n;
		}
		inline Point<3,double> geo_coord(const VoxelVertex vtx) const {return _low_ + (_high_-_low_)*vtx.normalized_coordinate();}


		//methods to find or access elements by their compressed index idx
		uint64_t find_element(const VoxelElement el) const;
		VoxelElement get_element(const uint64_t idx) const;
		VoxelElement& get_element(const uint64_t idx);

		//get iterators to start of elements
		inline Iterator begin() {return Iterator::begin(elements);}
		inline CIterator cbegin() const {return CIterator::begin(elements);}
		inline CIterator begin() const {return cbegin();}

		//get iterators to end of elements
		inline Iterator end() {return Iterator::end(elements);}
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

		//sort the elements by linear index or color
		void sort_by_color();
		void sort_by_index();

		//color the elements so that no two elements with the same color are contained in the support of the same dof
		//TODO: think about how this can be done iteratively with different passes of dofhandlers
		template<typename DOF_t>
		void color(std::span<const DOF_t> dofs);

		//color the elements so that no two elements with the same color share a vertex (conformal or non-conformal)
		void color();

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

		//collect the vertices, usually only used before writing to a vtk file
		void collect_vertices();
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
		void save_as_ascii(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,true>(filename, *this, description);}
		void save_as_binary(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,false>(filename, *this, description);}
	};


	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::sort_by_index() {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
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
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
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
	void UnstructuredVoxelMesh<MAX_DEPTH>::color() {
		//assign a color 0-7 based on the parity of i,j,k
		//if the mesh elements are disjoint, this guarantees that no two elements
		//with the same color will intersect at their boundaries.
		for (uint64_t dd=0; dd<=MAX_DEPTH; ++dd) {
			#pragma omp simd
			for (uint64_t i=0; i<elements[dd].size(); ++i) {
				VoxelElement& el = elements[dd][i];
				const uint64_t clr = (el.i()&1) | ((el.j()&1)<<1) | ((el.k()&1)<<2);
				el.set_color(clr);
			}
		}
		_colored_  = true;
	}

	template<uint64_t MAX_DEPTH>
	void UnstructuredVoxelMesh<MAX_DEPTH>::collect_vertices() {
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



	//Implement the Iterator class over all elements
	template<uint64_t MAX_DEPTH, typename Element_t, bool CONST_FLAG>
	struct IteratorBase
	{
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

		//implicit conversion from non-const to const
		IteratorBase(const IteratorBase<MAX_DEPTH,Element_t,false>& it) requires (CONST_FLAG) 
			: elements(it.elements), depth(it.depth), idx(it.idx) {}

		//begin/end iterators
		static IteratorBase end(container_ref els) {return IteratorBase{els, MAX_DEPTH+1, 0};}
		static IteratorBase begin(container_ref els) {return IteratorBase{els,0,0};}

		//advance to the next valid depth/index pair (if the current is valid, they aren't changed)
		//this should just skip over any empty depths
		void advance_to_valid() {
			while (depth<=MAX_DEPTH && idx>= (*elements)[depth].size()) {
				++depth;
				idx = 0;
			}
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

		bool operator==(const IteratorBase& other) const {
			return depth==other.depth && idx==other.idx;
		}

		bool operator!=(const IteratorBase& other) const {
			return depth!=other.depth || idx!=other.idx;
		}
	};



}