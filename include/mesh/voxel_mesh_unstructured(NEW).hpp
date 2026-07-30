#pragma once

#include "gutil.hpp"

#include "util/concepts.hpp"
#include "util/macros.hpp"

#include "mesh/keys/voxel_key.hpp"
#include "mesh/vtk_file_io.hpp"
#include "mesh/voxel_mesh_structured.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <array>
#include <span>
#include <mutex>

#include <iostream>
#include <sstream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV {


	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Forward declare iterator class
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename Feature_t, typename LayerMesh_t, bool CONST_FLAG>
			requires( std::same_as<Feature_t, typename LayerMesh_t::VoxelElement> || std::same_as<Feature_t, typename LayerMesh_t::VoxelVertex>)
	struct IteratorBase;


	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/// An unstructured voxel mesh. Uses 'layers' of structured meshes at each depth.
	/// The initial mesh is assumed to satisfy a 2-1 refinement constraint. Using the request refine/coarsen
	/// will maintain the 2-1 invariant.
	///
	/// Each layer contains a vector<unsigned char> with one entry for each possible element at that layer.
	/// One bit is reserved for an active flag and four bits are reserved for a refined depth field tracker.
	///
	/// Suppose D is the depth field that corresponds to some element E. If E is active, then D=E.depth.
	/// If E is not active but has active descendants, then D=max(C.depth) over all active descendants C.
	/// If E is not active and has no active descendants, then D=0.
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	template<uint64_t MaxDepth=10>
	struct UnstructuredVoxelMesh {
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////////////////////////////////////////////
		using VoxelElement = VoxelElementKey<MaxDepth,0>;
		using VoxelVertex  = VoxelVertexKey<MaxDepth,0>;
		using VoxelFace    = VoxelFaceKey<MaxDepth,0>;
		using GeoPoint_t   = gutil::Point<3,double>;
		using Box_t        = gutil::Box<3,double>;
		using Mesh_t       = UnstructuredVoxelMesh<MaxDepth>;

		using S_Layer_t    = StructuredVoxelMesh<MaxDepth>;	//the mesh type that organizes each depth
		using U_Layer_t    = UnstructuredLayer<MaxDepth>;	//the unstructured type for each individual layer

		static constexpr uint64_t MAX_DEPTH = MaxDepth;		//the maximum depth to maintain hierarchies
		static constexpr uint64_t TOTAL_POSSIBLE_ELEMENTS = total_possible<VoxelElement>(MAX_DEPTH);

		//random access iterator class to loop through the elements
		//this wraps the individual vector iterators but wraps to the next depth if possible
		using ElementIterator  = IteratorBase<VoxelElement,S_Layer_t,false>;
		using CElementIterator = IteratorBase<VoxelElement,S_Layer_t,true>;
		using VertexIterator   = IteratorBase<VoxelVertex,S_Layer_t,false>;
		using CVertexIterator  = IteratorBase<VoxelVertex,S_Layer_t,true>;
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Data
		/////////////////////////////////////////////////////////////////////////////////////////////////
		const Box_t box;

	protected:
		std::array<S_Layer_t,MAX_DEPTH+1> s_layers{};	//the primary mesh storage

		mutable gutil::ThreadPool large_pool{};			//thread pool with the number of available threads equal to the physical cores
		mutable gutil::ThreadPool small_pool{2};		//smaller thread pool to allow openmp parallelism within the dispatched tasks
		mutable std::mutex mtx;


		//store a list of elements that classes with a const reference to the mesh can use to request
		//element refinement or unrefinement
		mutable std::array<std::vector<VoxelElement>,MAX_DEPTH+1> request_refine{};
		mutable std::array<std::vector<VoxelElement>,MAX_DEPTH+1> request_unrefine{};

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// A few utility methods
		/////////////////////////////////////////////////////////////////////////////////////////////////
		void init_depths() noexcept {
			for (uint64_t dd=0; dd<=MAX_DEPTH; ++dd) {
				s_layers[dd].depth = dd;
				s_layers[dd].box = box;
				s_layers[dd].init_element_mask();
				#ifdef _OPENMP
					s_layers[dd].max_omp_threads = omp_get_max_threads()/small_pool.n_threads();
				#endif
			}
		}

		template<typename T>
		static constexpr void sort_and_unique(std::vector<T>& list) noexcept {
			std::sort(list.begin(), list.end());
			auto last = std::unique(list.begin(), list.end());
			list.erase(last, list.end());
		}
		
	public:
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Constructors
		/////////////////////////////////////////////////////////////////////////////////////////////////
		UnstructuredVoxelMesh() : box{{0,0,0}, {1,1,1}} { init_depths(); set_depth(0); }
		UnstructuredVoxelMesh(const Box_t& box, uint64_t depth=0) : box{box} { init_depths(); set_depth(depth); }

		UnstructuredVoxelMesh(const UnstructuredVoxelMesh&) = delete;
		UnstructuredVoxelMesh(UnstructuredVoxelMesh&& other) :
			box{std::move(other.box)},
			s_layers{std::move(other.s_layers)} {}
		
		UnstructuredVoxelMesh& operator=(const UnstructuredVoxelMesh&) = delete;
		UnstructuredVoxelMesh& operator=(UnstructuredVoxelMesh&&) = delete;
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Simple queries and commands
		/////////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool is_active(VoxelElement el) const noexcept { return s_layers[el.depth()].is_active(el); }
		[[nodiscard]] unsigned char read_depth(VoxelElement el) const noexcept { return s_layers[el.depth()].read_depth(el);}
		[[nodiscard]] const S_Layer_t& get_layer(uint64_t dd) const noexcept {
			GUTIL_ASSERT(dd<=MAX_DEPTH)
			return s_layers[dd];
		}

		void set_depth(uint64_t depth) noexcept {
			if (depth>MAX_DEPTH) { return; }

			for (S_Layer_t& layer : s_layers) {
				large_pool.submit( [&](){
					layer.set_all_active(depth==layer.depth);
					if (layer.depth<=depth) {
						layer.set_all_depth(static_cast<unsigned char>(depth));
					}
				});
			}
			large_pool.wait_idle();
		}

		void update_unstructured() noexcept {
			for (S_Layer_t& layer : s_layers) {
				small_pool.submit( [&](){ layer.update_unstructured_omp(); });
			}
			small_pool.wait_idle();
		}

		void collect_vertices() noexcept {
			for (S_Layer_t& layer : s_layers) {
				small_pool.submit( [&](){ layer.collect_vertices_omp(); });
			}
			small_pool.wait_idle();

			//deduplicate by moving 'coarse' vertices to their correct depth.
			//note that layers with no active elements may have vertices.
			for (uint64_t dd=MAX_DEPTH; dd>0; --dd) {
				for (VoxelVertex vtx : s_layers[dd].unstructured.vertices) {
					if (vtx.depth() != dd) {
						assert(vtx.depth() < dd);
						s_layers[vtx.depth()].unstructured.vertices.push_back(vtx);
					}
				}

				//sort and de-duplicate the next list
				sort_and_unique( s_layers[dd-1].unstructured.vertices );

				//clean up the current list (already sorted, just delete the bad vertices)
				std::erase_if( s_layers[dd].unstructured.vertices, [dd](VoxelVertex vtx){ return vtx.depth() < dd; });
			}
		}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods for assigning DOFs to features
		/////////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool is_conformal(VoxelElement el) const noexcept {
			//all active elements are trivially conformal as they do not overlap
			assert(el.exists());
			return is_active(el);
		}

		template<typename Feature_t> requires (std::same_as<Feature_t,VoxelVertex> || std::same_as<Feature_t,VoxelFace>)
		[[nodiscard]] bool is_conformal(Feature_t f) const noexcept {
			//a face/vertex is conformal if all of its elements at its depth are active
			//for assigning dofs, the depth of the vertex is essential, so we do not
			//check if there is an 'equivalent' conformal vertex at the same geometric location
			assert(f.is_valid());
			const uint64_t dd = f.depth();
			for (VoxelElement el : f.elements()) {
				if (el.exists()) {
					if (!s_layers[dd].is_active(el)) {return false;}
				}
			}
			return true;
		}

		[[nodiscard]] VoxelVertex get_conformal(VoxelVertex vtx) const noexcept {
			//if there is a conformal vertex at the same geometric location, get it.
			//if there is no such vertex, the the DOES_NOT_EXIST vertex is returned.
			//we check from the shallowest to deepest vertex at the same location
			assert(vtx.is_valid());
			while(vtx.parent().exists()) {vtx = vtx.parent();}
			for (uint64_t dd=vtx.depth(); dd<=MAX_DEPTH; ++dd) {
				if ( is_conformal(vtx) ) { return vtx; }
				else if ( vtx.child().exists() ) { vtx = vtx.child(); }
				else { break; }
			}
			return VoxelVertex::None();
		}



		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods primarily for writing to vtk files
		/////////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] GeoPoint_t geo_coord(const VoxelVertex vtx) const noexcept {return box.low + (box.high-box.low)*vtx.normalized_coordinate();}
		
		[[nodiscard]] uint64_t n_elements() const noexcept {
			//the unstructured layers must be up to date
			uint64_t n=0;
			for (const S_Layer_t& layer : s_layers) { n+=layer.n_elements(); }
			return n;
		}

		[[nodiscard]] uint64_t n_vertices() const noexcept {
			//the unstructured layers must be up to date and the vertices collected
			uint64_t n=0;
			for (const S_Layer_t& layer : s_layers) { n+=layer.n_vertices(); }
			return n;
		}


		[[nodiscard]] uint64_t vertex_index(VoxelVertex vtx) const noexcept {
			//the unstructured layers must be up to date and the vertices collected
			assert(vtx.is_valid());
			const uint64_t depth = vtx.depth();
			uint64_t n=0;
			for (uint64_t dd=0; dd<depth; ++dd) { n+=s_layers[dd].n_vertices(); }
			return n + s_layers[depth].vertex_index(vtx);
		}

		void save_as_ascii(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,true>(filename, *this, description);}
		void save_as_binary(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,false>(filename, *this, description);}

		template<typename... Lookup_ts>
		void append_cell_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_cell_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		void append_cell_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_cell_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		void append_point_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_point_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		void append_point_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_point_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods for manipulating the mesh
		/////////////////////////////////////////////////////////////////////////////////////////////////
		void refine(VoxelElement el) const noexcept {
			GUTIL_ASSERT(el.depth()<MAX_DEPTH);
			std::lock_guard<std::mutex> lock(mtx);
			request_refine[el.depth()].push_back(el);
		}
		
		template<typename Predicate = std::nullptr_t>
		void unrefine(VoxelElement el, Predicate&& pred=nullptr) const noexcept {
			GUTIL_ASSERT(el.depth()>0);
			//no sibling may have an active descendant
			//all active siblings must satisfy the predicate
			const unsigned char dd = static_cast<unsigned char>(el.depth());
			for (VoxelElement sib : el.parent().children()) {
				if (s_layers[dd].read_depth(sib) > dd) {return;}
				if constexpr (!NULLPTR_T<Predicate>) {
					if (s_layers[dd].is_active(sib) && !pred(sib)) {return;}
				}
			}
			//the cell that we want to be active is tracked
			std::lock_guard<std::mutex> lock(mtx);
			request_unrefine[dd-1].push_back(el.parent());
		}

		template<typename Predicate>
		void refine(Predicate&& pred) const noexcept {
			auto action = [&](VoxelElement el, uint64_t dd) {
				if (pred(el)) request_refine[dd].push_back(el);
			};
			
			auto job = [&](uint64_t dd) { s_layers[dd].for_each_element(action, dd); };

			for (uint64_t dd=0; dd<MAX_DEPTH; ++dd) {
				large_pool.submit( job, dd );
			}
			large_pool.wait_idle();
		}

		template<typename Predicate>
		void unrefine(Predicate&& pred) const noexcept {
			auto action = [&](VoxelElement el, unsigned char dd) {
				//all active siblings must satisfy the predicate to unrefine
				for (VoxelElement sib : el.parent().children()) {
					//a sibling must not have an active descendent to be able to unrefine
					//additionally, all active siblings must satisfy the predicate
					if (s_layers[dd].read_depth(sib) > static_cast<uint64_t>(dd)) {return;}
					if (s_layers[dd].is_active(sib) && !pred(sib)) {return;}
				}
				//note that the parent will be added 8 times...but will be deduplicated later
				//probably more efficient to do that than to track if it has already been aded
				request_unrefine[dd-1].push_back(el.parent());
			};
			auto job = [&](unsigned char dd) { s_layers[dd].for_each_element(action, dd); };

			for (unsigned char dd=1; dd<=static_cast<unsigned char>(MAX_DEPTH); ++dd) {
				large_pool.submit(job, dd);
			}
			large_pool.wait_idle();
		}

		void synchronize_depth_field() noexcept;

		void process_unrefine() noexcept;

		template<typename Predicate = std::nullptr_t>
		void process_refine(Predicate&& pred = nullptr) noexcept;

		void init_color() noexcept {
			for (uint64_t dd=0; dd<=MAX_DEPTH; ++dd) {
				auto job = [&,dd]() { s_layers[dd].unstructured.color(); };
				large_pool.submit(job);
			}
			large_pool.wait_idle();
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Iterators for a standardized interface: TODO add a face iterator
		/////////////////////////////////////////////////////////////////////////////////////////////////
		CElementIterator element_begin() const {return CElementIterator::begin(s_layers);}
		CElementIterator element_end()   const {return CElementIterator::end(s_layers);}
		auto element_begin(uint64_t dd)  const {return s_layers[dd].unstructured.elements.cbegin();}
		auto element_end(uint64_t dd)    const {return s_layers[dd].unstructured.elements.cend();}

		ElementIterator element_begin() {return ElementIterator::begin(s_layers);}
		ElementIterator element_end()   {return ElementIterator::end(s_layers);}
		auto element_begin(uint64_t dd) {return s_layers[dd].unstructured.elements.begin();}
		auto element_end(uint64_t dd)   {return s_layers[dd].unstructured.elements.end();}
		
		CVertexIterator vertex_begin() const {return CVertexIterator::begin(s_layers);}
		CVertexIterator vertex_end()   const {return CVertexIterator::end(s_layers);}
		auto vertex_begin(uint64_t dd) const {return s_layers[dd].unstructured.vertices.cbegin();}
		auto vertex_end(uint64_t dd)   const {return s_layers[dd].unstructured.vertices.cend();}

		VertexIterator vertex_begin()  {return VertexIterator::begin(s_layers);}
		VertexIterator vertex_end()    {return VertexIterator::end(s_layers);}
		auto vertex_begin(uint64_t dd) {return s_layers[dd].unstructured.vertices.begin();}
		auto vertex_end(uint64_t dd)   {return s_layers[dd].unstructured.vertices.end();}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// A few simple mesh refinement operations that are thread safe when called at the same depth
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//TODO: when coloring globally, it might be better to return pointers to the actual elements rather than copies
		[[nodiscard]] std::vector<VoxelElement> neighbors(VoxelElement el) const noexcept {
			GUTIL_ASSERT(el.is_valid())
			GUTIL_ASSERT(is_active(el))

			const uint64_t dd = el.depth();
			if (!s_layers[dd].is_active(el)) { return {}; }

			//a coarse cell at depth dd surrounded by cells at depth dd+1 will have 54 neighbors
			std::vector<VoxelElement> list;
			list.reserve(54);

			for (VoxelElement nbr : el.neighbors()) {
				if (!nbr.is_valid()) { continue; }

				if ( s_layers[dd].is_active(nbr) ) { list.push_back(nbr); }
				else if ( dd>0 && s_layers[dd-1].is_active(nbr.parent())) { list.push_back(nbr.parent()); }
				else if ( dd<MAX_DEPTH ) {
					
					//to only get the correct fine neighbors, we need to be careful
					//for each axis, if el's index is even (low), then the low depth-dd-neighbor's children
					//must have an odd (high) axis index. Similar rules apply to corner and other neighbors.
					
					//track if the nbr child needs low/high/any bits for each axis
					//note that el.i/j/k and nbr.i/j/k are at most one apart
					const int64_t di = static_cast<int64_t>(nbr.i()) - static_cast<int64_t>(el.i());
					const int64_t dj = static_cast<int64_t>(nbr.j()) - static_cast<int64_t>(el.j());
					const int64_t dk = static_cast<int64_t>(nbr.k()) - static_cast<int64_t>(el.k());

					for (VoxelElement c : nbr.children()) {
						if (!s_layers[dd+1].is_active(c)) {continue;}

						//check if each axis index is ok (any/low/high)
						if ( (di>0 && static_cast<bool>(c.i()&1)) || (di<0 && !static_cast<bool>(c.i()&1)) ) {continue;}
						if ( (dj>0 && static_cast<bool>(c.j()&1)) || (dj<0 && !static_cast<bool>(c.j()&1)) ) {continue;}
						if ( (dk>0 && static_cast<bool>(c.k()&1)) || (dk<0 && !static_cast<bool>(c.k()&1)) ) {continue;}
						list.push_back(c);
					}
				}
			}
			return list;
		}
	};


	//////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Implementations
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	template<uint64_t MaxDepth>
	void UnstructuredVoxelMesh<MaxDepth>::process_unrefine() noexcept {
		GUTIL_TIMER("process_unrefine : ", n_elements(), " current elements");
		{
			//ensure the unrefine lists don't contain duplicates
			//additionally, add elements to unrefine so that the 2-1 refinement rule is respected
			//note that elements marked as 'unrefine' are elements that should be activated and
			//if they have no active descendant, the request is removed.
			auto job = [&](unsigned char dd) {
				sort_and_unique(request_unrefine[dd]);
				//this is only a valid unrefinement target if it is not active and its depth field is greater
				//than its own depth. The latter guarantees the former as the elements are disjoint.
				//additionally, rather than adding cells to unrefine, we only refine cells that will still
				//satisfy the 2-1 rule.
				std::erase_if(request_unrefine[dd], [&,dd](VoxelElement el) {
					if (s_layers[dd].read_depth(el) <= dd) {return true;}
					for (VoxelElement c : el.children()) {
						if (!s_layers[dd+1].is_active(c)) { continue; }
						for (VoxelElement nbr : neighbors(c)) {
							if (nbr.depth() > static_cast<uint64_t>(dd+1)) {
								return true; //activating will break 2-1	
							}
						}
					}
					return false; //we are ok to activate
				});
			};

			for (unsigned char dd=0; dd<=static_cast<unsigned char>(MAX_DEPTH); ++dd) {
				large_pool.submit(job, dd);
			}
			large_pool.wait_idle();
		}

		{
			//we can process in 2 parallel batches due to the 2-1 rule
			//deactivate specified elements and its siblings at level dd,
			//then activate its parent at level dd-1
			auto job = [&](uint64_t dd) {
				GUTIL_OMP(parallel for num_threads(omp_get_max_threads()/2) if(request_unrefine[dd].size()>4096))
				for (size_t i=0; i<request_unrefine[dd].size(); ++i) {
					VoxelElement el = request_unrefine[dd][i];
					s_layers[dd+1].set_active(el.children(), false);
					s_layers[dd].set_active(el, true);
				}
				request_unrefine[dd].clear();
			};

			for (uint64_t dd=0; dd<MAX_DEPTH; dd+=2) {large_pool.submit(job,dd);}
			large_pool.wait_idle();

			for (uint64_t dd=1; dd<MAX_DEPTH; dd+=2) {large_pool.submit(job,dd);}
			large_pool.wait_idle();
		}

		synchronize_depth_field();
	}

	template<uint64_t MaxDepth>
	template<typename Predicate>
	void UnstructuredVoxelMesh<MaxDepth>::process_refine(Predicate&& pred) noexcept {
		GUTIL_TIMER("process_refine : ", n_elements(), " current elements");

		//ensure the refine lists don't contain duplicates
		//additionally, add elements to refine so that the 2-1 refinement rule is respected
		//note that elements at the MAX_DEPTH cannot be refined and elements at depth 0
		//cannot have neighbors that are 'too coarse'
		for (unsigned char dd=static_cast<unsigned char>(MAX_DEPTH)-1; dd>=1; --dd) {
			sort_and_unique(request_refine[dd]);
			for (VoxelElement el : request_refine[dd]) {
				for (VoxelElement nbr : neighbors(el)) {
					if (static_cast<unsigned char>(nbr.depth()) == dd-1) {
						request_refine[dd-1].push_back(nbr);	
					}
				}
			}
		}
		sort_and_unique(request_refine[0]);

		//we can process in 2 parallel batches due to the 2-1 rule
		//deactivate specified elements at level dd,
		//then activate its children that satisfy the predicate at level dd+1
		auto job = [&](unsigned char dd) {
			GUTIL_OMP(parallel for num_threads(omp_get_max_threads()/2) if(request_refine[dd].size()>4096) schedule(guided,512))
			for (size_t i=0; i<request_refine[dd].size(); ++i) {
				VoxelElement el = request_refine[dd][i];
				s_layers[dd].set_active(el, false);
				
				if constexpr (NULLPTR_T<Predicate>) {
					s_layers[dd+1].set_active(el.children(), true);
				}
				else {
					for (VoxelElement chi : el.children()) {
						if (pred(chi)) {
							s_layers[dd+1].set_active(chi,true);
						}
					}
				}
			}
			request_refine[dd].clear();
		};

		for (unsigned char dd=0; dd<static_cast<unsigned char>(MAX_DEPTH); dd+=2) {large_pool.submit(job, dd);}
		large_pool.wait_idle();

		for (unsigned char dd=1; dd<static_cast<unsigned char>(MAX_DEPTH); dd+=2) {large_pool.submit(job,dd);}
		large_pool.wait_idle();

		synchronize_depth_field();
	}

	template<uint64_t MaxDepth>
	void UnstructuredVoxelMesh<MaxDepth>::synchronize_depth_field() noexcept {
		GUTIL_TIMER("synchronize_depth_field : ", TOTAL_POSSIBLE_ELEMENTS, " total elements");

		auto action = [&](auto idx, auto dd) {
			if (dd==static_cast<unsigned char>(MAX_DEPTH)) {
				s_layers[dd].set_depth(idx, s_layers[dd].is_active(idx) ? dd : 0);
				return;
			}

			if (s_layers[dd].is_active(idx)) {s_layers[dd].set_depth(idx,dd);}
			else {
				unsigned char cd = 0;
				for (VoxelElement c : VoxelElement{static_cast<uint64_t>(dd),idx}.children()) {
					cd = std::max(cd, s_layers[dd+1].read_depth(c));
				}
				s_layers[dd].set_depth(idx,cd);
			}
		};

		for (unsigned char dd=static_cast<unsigned char>(MAX_DEPTH+1); dd>0; --dd) {
			s_layers[dd-1].for_each_index_simd(action, dd-1);
		}
	}


	
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// Implement the Iterator class over all elements or vertices.
	// Element iterators are only valid from when update_unstructured() is called to the next time the mesh is altered.
	// Vertex iterators are similar, but collect_vertices() must also be called.
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename Feature_t, typename LayerMesh_t, bool CONST_FLAG>
			requires( std::same_as<Feature_t, typename LayerMesh_t::VoxelElement> || std::same_as<Feature_t, typename LayerMesh_t::VoxelVertex>)
	struct IteratorBase	{
		//necessary aliases for the standard library
		using iterator_category = std::forward_iterator_tag;
		using value_type		= Feature_t;
		using difference_type	= std::ptrdiff_t;
		using pointer			= std::conditional_t<CONST_FLAG, Feature_t const*, Feature_t*>;
		using reference			= std::conditional_t<CONST_FLAG, const Feature_t&, Feature_t&>;

		//convenient aliases
		static constexpr uint64_t MAX_DEPTH = LayerMesh_t::MAX_DEPTH;
		using container_type	= std::array<LayerMesh_t,MAX_DEPTH+1>;
		using container_ref		= std::conditional_t<CONST_FLAG, const container_type&, container_type&>;
		using container_ptr 	= std::conditional_t<CONST_FLAG, const container_type*, container_type*>;

		using list_type         = std::vector<Feature_t>;
		using list_ref          = std::conditional_t<CONST_FLAG, const list_type&, list_type&>;
		using list_ptr          = std::conditional_t<CONST_FLAG, const list_type*, list_type*>;

		//get reference to the feature list
		[[nodiscard]] list_ref get_list(uint64_t depth) const { 
			if constexpr (std::same_as<Feature_t, typename LayerMesh_t::VoxelElement>) {
				return (*layers)[depth].unstructured.elements;
			}
			else {
				return (*layers)[depth].unstructured.vertices;
			}
		}

		//current position: layers[depth].feature[idx]
		container_ptr layers;
		uint64_t depth;
		uint64_t idx;
		
		//constructor
		IteratorBase(container_ref layers, uint64_t dd, uint64_t ii) : layers(&layers), depth(dd), idx(ii) {advance_to_valid();}
		IteratorBase(container_ref layers, uint64_t n) : layers(&layers), depth(0), idx(0) {(*this)+=n; advance_to_valid();}

		//implicit conversion from non-const to const
		IteratorBase(const IteratorBase<Feature_t,LayerMesh_t,false>& it) requires (CONST_FLAG) 
			: layers(it.layers), depth(it.depth), idx(it.idx) {}

		//begin/end iterators
		static IteratorBase end(container_ref layers) {return IteratorBase{layers, MAX_DEPTH+1, 0};}
		static IteratorBase begin(container_ref layers) { return IteratorBase{layers,0,0};}

		//random access operations
		reference operator[](uint64_t n) const {return *(*this+n);}

		//advance to the next valid depth/index pair (if the current is valid, they aren't changed)
		//this should just skip over any empty depths
		void advance_to_valid() {
			while (depth<=MAX_DEPTH && idx>= get_list(depth).size()) {
				++depth;
				idx = 0;
			}
		}

		uint64_t total_length() const {
			uint64_t len=0;
			for (int dd=0; dd<=MAX_DEPTH; ++dd) {len+=get_list(dd).size();}
			return len;
		}

		uint64_t index() const {
			if (depth>MAX_DEPTH) {return total_length();}

			uint64_t el_idx = 0;
			for (uint64_t dd=0; dd<depth; ++dd) {
				el_idx += get_list(dd).size();
			}
			el_idx += idx;
			return el_idx;
		}

		//access data
		reference operator*() const {
			assert(depth <= MAX_DEPTH);
			assert(idx < get_list(depth).size());
			return get_list(depth)[idx];
		}

		pointer operator->() const {
			assert(depth <= MAX_DEPTH);
			assert(idx < get_list(depth).size());
			return &get_list(depth)[idx];
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
				const uint64_t remaining_in_depth = get_list(depth).size() - idx;
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
			*this = IteratorBase(*layers, flat-n);
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