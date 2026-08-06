#pragma once

#include "gutil.hpp"

#include "util/concepts.hpp"
#include "util/macros.hpp"

#include "simd_keys/containers.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"
#include "mesh/vtk_file_io.hpp"
#include "mesh/voxel_mesh_structured.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <array>
#include <span>
#include <mutex>
#include <atomic>

#include <iostream>
#include <sstream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV {


	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/// An unstructured voxel mesh. Uses 'layers' of structured meshes at each depth.
	/// The initial mesh is assumed to satisfy a 2-1 refinement constraint. Using the request refine/coarsen
	/// will maintain the 2-1 invariant.
	///
	/// Each layer contains a vector<uint8_t> with one entry for each possible element at that layer.
	/// One bit is reserved for an active flag and four bits are reserved for a refined depth field tracker.
	///
	/// Suppose D is the depth field that corresponds to some element E. If E is active, then D=E.depth.
	/// If E is not active but has active descendants, then D=max(C.depth) over all active descendants C.
	/// If E is not active and has no active descendants, then D=0.
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T=double>
	struct UnstructuredVoxelMesh : public Keys::HybridKeyTracker {
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Aliases and masks
		/////////////////////////////////////////////////////////////////////////////////////////////////
		using BASE         	= Keys::HybridKeyTracker;

		using Mesh_t       	= UnstructuredVoxelMesh<T>;
		using Elem_t   		= Keys::VoxelElement<0>;
		using Vert_t   		= Keys::VoxelVertex<0>;
		using Face_t   		= void;
		using Edge_t   		= void;

		using GeoPoint_t	= gutil::Point<3,T>;
		using Box_t        	= gutil::Box<3,T>;
		using Scalar_t     	= T;

		using BASE::ACTIVE_BIT; 					//0b00000001;	
		static constexpr uint8_t DEPTH_BITS 		= 0b00011110;
		static constexpr uint8_t FREE_BITS 	        = 0b11100000;


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Data
		/////////////////////////////////////////////////////////////////////////////////////////////////
		const uint8_t				max_depth;
		const uint64_t				max_possible_elements;
		
		const Box_t 				box;							//physical extents of the domain
		const GeoPoint_t 			diag;							//diagonal/sidelength of the domain
		const GeoPoint_t 			inv_diag;						//reciprocal of the sidelength
		
		std::vector<Vert_t> 		tracked_vertices;				//a compressed list of 'active' vertices, primarily used for visualization. reduced keys are stored.
		gutil::BinSort<Vert_t>		vertex_sorter;					//sort the vertices for better lookup and deduplication
		mutable std::atomic<bool>	is_vertices_collected_{false};	

		protected:
		std::span<Elem_t>			active_elements;				//a view into BASE::active_keys (std::vector<uint64_t>) re-interpreted as Elem_t
		using BASE::key_mask;										//vector<uint8_t> of size max_possible_elements
		using BASE::threads;
		using BASE::sorter;											//the bins of this sorter will correspond to each depth in the mesh, will need to re-interpret the spans

		mutable std::mutex							request_mutex{};			//sync refine requests. TODO: make requests per-thread-per-depth if it's slow
		mutable std::vector<std::vector<Elem_t>> 	request_refine_list{};		//allow classes with const-ref to request refinement
		mutable std::vector<std::vector<Elem_t>>	request_unrefine_list{};	//allow classes with const-ref to request unrefinement

		public:
		using BASE::is_key_mask_unstable;									//synchronization tools used in macros
		using BASE::is_key_mask_stable;										//exposing to external classes can be helpul
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


		//a few helper methods to check that we haven't forgot to call something like collect_vertices etc.
		[[nodiscard]] bool is_vertices_collected() const noexcept {return is_vertices_collected_.load();}
		[[nodiscard]] bool is_active_elements_linked() const noexcept {
			return reinterpret_cast<uintptr_t>(active_elements.data())==reinterpret_cast<uintptr_t>(BASE::active_keys.data()) &&
					reinterpret_cast<uintptr_t>(active_elements.data()+active_elements.size())==reinterpret_cast<uintptr_t>(BASE::active_keys.data()+BASE::active_keys.size());
		}
		[[nodiscard]] bool is_current() const noexcept {	//for most processes we only care about the elements
			return BASE::is_current() && is_active_elements_linked();
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Constructors
		/////////////////////////////////////////////////////////////////////////////////////////////////
		UnstructuredVoxelMesh(const Box_t& box = Box_t{{0,0,0},{1,1,1}}, uint8_t max_depth_ = 8) : 
			BASE(Elem_t::total_possible(std::min(max_depth_,(uint8_t)BASE::KEY_MAX_DEPTH))),
			max_depth{std::min(max_depth_, (uint8_t)BASE::KEY_MAX_DEPTH)},
			max_possible_elements{Elem_t::total_possible(max_depth)},
			box{box}, 
			diag{box.sidelength()}, 
			inv_diag{Scalar_t{1}/diag} {
				if (max_depth_>BASE::KEY_MAX_DEPTH) {
					GUTIL_LOG("Depth ", max_depth_, " was requested, but the data type only support up to ", BASE::KEY_MAX_DEPTH);
				}

				request_refine_list.resize(max_depth+1);		//index max_depth is valid
				request_unrefine_list.resize(max_depth+1);
			}
		
		UnstructuredVoxelMesh(const UnstructuredVoxelMesh&) = delete;
		UnstructuredVoxelMesh(UnstructuredVoxelMesh&& other) = default;
		UnstructuredVoxelMesh& operator=(const UnstructuredVoxelMesh&) = delete;
		UnstructuredVoxelMesh& operator=(UnstructuredVoxelMesh&&) = delete;
		~UnstructuredVoxelMesh() {std::lock_guard<std::mutex> lock(request_mutex);}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Adapt masking function to VoxelElements and this classes's masks
		/////////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] uint8_t get_mask(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::get_mask(el.linear_index());
		}
		[[nodiscard]] uint8_t& get_mask_ref(Elem_t el) noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::get_mask_ref(el.linear_index());
		}
		[[nodiscard]] uint8_t get_mask_stable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::get_mask_stable(el.linear_index());
		}
		[[nodiscard]] uint8_t get_mask_unstable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::get_mask_unstable(el.linear_index());
		}
		[[nodiscard]] uint8_t& get_mask_ref_pseudo_const(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::get_mask_ref_pseudo_const(el.linear_index());
		}


		[[nodiscard]] bool is_active(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::is_active(el.linear_index());
		}
		[[nodiscard]] bool is_active_no_check(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::is_active_no_check(el.linear_index());
		}
		[[nodiscard]] bool is_active_stable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::is_active_stable(el.linear_index());
		}
		[[nodiscard]] bool is_active_unstable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::is_active_unstable(el.linear_index());
		}
		[[nodiscard]] bool set_active_check_changed(Elem_t el, bool val) noexcept {
			GUTIL_ASSERT(el.is_valid()); return BASE::set_active_check_changed(el.linear_index(), val);
		}
		void set_active(Elem_t el, bool val) noexcept {
			GUTIL_ASSERT(el.is_valid()); BASE::set_active(el.linear_index(), val);
		}

		[[nodiscard]] uint8_t read_depth_field(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return (get_mask(el)&DEPTH_BITS)>>1;
		}
		[[nodiscard]] uint8_t read_depth_field_stable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return (get_mask_stable(el)&DEPTH_BITS)>>1;
		}
		[[nodiscard]] uint8_t read_depth_field_unstable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return (get_mask_unstable(el)&DEPTH_BITS)>>1;
		}
		[[nodiscard]] bool set_depth_field_check_changed(Elem_t el, uint8_t val) noexcept {
			GUTIL_ASSERT(el.is_valid()); 
			uint8_t& byte = get_mask_ref(el);
			uint8_t old = (byte&DEPTH_BITS)>>1;
			byte&=~DEPTH_BITS;
			byte|=(DEPTH_BITS&(val<<1));
			return val!=old;
		}
		void set_depth_field(Elem_t el, uint8_t val) noexcept {
			GUTIL_ASSERT(el.is_valid()); 
			uint8_t& byte = get_mask_ref(el);
			byte&=~DEPTH_BITS;
			byte|=(DEPTH_BITS&(val<<1));
		}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Bin functions for sorting elements and vertices
		/////////////////////////////////////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		static constexpr int element_key_bin(uint64_t key) noexcept {
			return static_cast<int>(Elem_t{key}.depth());
		}

		GUTIL_DECLARE_SIMD()
		static constexpr int vertex_key_bin(uint64_t key) noexcept {
			GUTIL_ASSERT(Vert_t{key}.is_valid());
			key = Keys::Mesh3D::ReducedVertex_SIMD(key);
			return static_cast<int>(Keys::Mesh3D::Depth(key));
		}

		static constexpr int vertex_bin(Vert_t vtx) noexcept {
			return static_cast<int>(vtx.reduced_key().depth());
		}

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Simple queries and commands
		/////////////////////////////////////////////////////////////////////////////////////////////////
		void set_depth(uint8_t depth) noexcept {
			BASE::clear();
			{
				GV_BEGIN_UNSTABLE

				uint64_t start = Elem_t::elements_below_depth(depth);
				uint64_t end   = Elem_t::elements_below_depth(depth+1);
				std::fill(key_mask.begin()+start, key_mask.begin()+end, ACTIVE_BIT);
				is_vertices_collected_.store(false);
				GV_END_UNSTABLE
			}
			collect_elements();
		}

		void collect_elements() noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GV_ASSERT_KEY_MASK_STABLE_STATE
			GUTIL_TIMER("collecting active elements");
			BASE::collect_active_keys<Elem_t>();
			active_elements = BASE::reinterpret_key_span<Elem_t,uint64_t>(std::span<uint64_t>(BASE::active_keys));
			BASE::sort_active_keys(max_depth, &UnstructuredVoxelMesh::element_key_bin);
		}

		void collect_vertices() noexcept {
			GV_BEGIN_STABLE  	//The base class+elements must be stable, we are only altering the vertices
			GUTIL_TIMER("Collecting active vertices");
			GUTIL_ASSERT(is_current());
			is_vertices_collected_.store(false);

			const size_t n_threads = threads.n_threads()==0 ? 1 : threads.n_threads();
			std::vector<std::vector<Vert_t>> thread_verts(n_threads);

			auto job = [&thread_verts](std::span<const uint64_t> el_keys, size_t tid) {
				auto& verts = thread_verts[tid];
				verts.resize(8*el_keys.size());
				#ifndef NDEBUG
					std::fill(verts.begin(), verts.end(), Vert_t{uint64_t(-1)});
				#endif

				GUTIL_SIMD()
				for (size_t i=0; i<el_keys.size(); ++i) {
					Elem_t{el_keys[i]}.vertices_simd(verts.data() + 8*i);
					for (size_t j=0; j<8; ++j) {
						verts[8*i + j].reduced_key_simd_in_place();
					}
				}

				#ifndef NDEBUG
					GUTIL_ASSERT(std::find(verts.begin(), verts.end(), Vert_t{uint64_t(-1)})==verts.end());
				#endif

				BASE::sort_and_unique(verts);
				verts.shrink_to_fit();
			};

			BASE::dispatch_parallel_active_keys_const(job);
			tracked_vertices.clear();
			threads.wait_idle();

			for(size_t i=0; i<n_threads; ++i) {
				tracked_vertices.insert(tracked_vertices.end(), std::make_move_iterator(thread_verts[i].begin()),
								std::make_move_iterator(thread_verts[i].end()));
			}
			thread_verts.clear();

			vertex_sorter = gutil::BinSort<Vert_t>(tracked_vertices, max_depth);
			vertex_sorter.dispatch_sort(&vertex_bin, &threads);
			threads.wait_idle();

			for (int n=0; n<vertex_sorter.n_bins(); ++n) {
				threads.submit([](auto a, auto b){std::sort(a,b);}, vertex_sorter.begin(n), vertex_sorter.end(n));
			}
			threads.wait_idle();
			is_vertices_collected_.store(true);
			GV_END_STABLE
		}


		//find the active element that the point belongs to and transform to the reference coordinate in [-1,1]^3
		GUTIL_DECLARE_SIMD()
		uint64_t find_element_raw_key(Scalar_t* x, Scalar_t* y, Scalar_t* z) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE

			uint64_t result{0};

			//get normalized coordinate at depth 0 as local values
			Scalar_t lx = Scalar_t{2}*( *x - box.low[0])*inv_diag[0] - Scalar_t{1};
			Scalar_t ly = Scalar_t{2}*( *y - box.low[1])*inv_diag[1] - Scalar_t{1};
			Scalar_t lz = Scalar_t{2}*( *z - box.low[2])*inv_diag[2] - Scalar_t{1};
			GUTIL_ASSERT(Scalar_t{-1} <= x && x <= Scalar_t{1});
			GUTIL_ASSERT(Scalar_t{-1} <= y && y <= Scalar_t{1});
			GUTIL_ASSERT(Scalar_t{-1} <= z && z <= Scalar_t{1});

			//get the element at each depth that contains the point
			//the x,y,z values should not correspond to a vertex for predicable results
			Elem_t el{0,0};						//current element, morton encoded
			for (uint8_t dd=0; dd<=max_depth; ++dd) {
				
				if (is_active_stable(el)) {
					result = el.key;
					*x     = lx;
					*y     = ly;
					*z     = lz;
				}
				

				//get the child/octant the point belongs to
				uint64_t idx = 0;
				if (lx>Scalar_t{0}) {idx|=0b001;}
				if (ly>Scalar_t{0}) {idx|=0b010;}
				if (lz>Scalar_t{0}) {idx|=0b100;}

				//go to new element (get the distance to the new center and re-scale)
				el = el.children_simd() + idx;
				lx = Scalar_t{2}*( lx - ((idx&0b001) ? Scalar_t{0.5} : -Scalar_t{0.5}) );
				ly = Scalar_t{2}*( ly - ((idx&0b010) ? Scalar_t{0.5} : -Scalar_t{0.5}) );
				lz = Scalar_t{2}*( lz - ((idx&0b100) ? Scalar_t{0.5} : -Scalar_t{0.5}) );
			}
			return result;
		}



		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods for assigning DOFs to features
		///
		/// A conformal vertex can only be contained in the closure of an active element if it is one of 
		/// its vertices (i.e., it cannot lay on an edge, face, or interior of an active element)
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template<uint8_t Period> requires(Period<8)
		[[nodiscard]] bool is_conformal(Keys::VoxelVertex<Period> vtx) const noexcept {
		    GV_ASSERT_KEY_MASK_STABLE_STATE

		    using P_Elem_t = Keys::VoxelElement<Period>;
		    using P_Vert_t = Keys::VoxelVertex<Period>;

		    P_Vert_t native = vtx.reduced_key();
		    const uint8_t native_depth = native.depth_u8();
		    if (native_depth==0) {return true;}

		    const uint64_t par  = native.kji_pairity_simd();
		    const uint64_t bi   = (par&0b001) ? 0 : 1;
		    const uint64_t bj   = (par&0b010) ? 0 : 1;
		    const uint64_t bk   = (par&0b100) ? 0 : 1;

		    const uint64_t dd   = native_depth-1;
		    const uint64_t n_el = uint64_t{1} << dd;

		    //el_idx[0]/[2]/[4] = the LOW candidate (q-1 when bi=1, else q); [1]/[3]/[5] = the HIGH (q)
		    int64_t el_idx[6];
		    el_idx[0] = static_cast<int64_t>(native.i()/2) - static_cast<int64_t>(bi);
		    el_idx[2] = static_cast<int64_t>(native.j()/2) - static_cast<int64_t>(bj);
		    el_idx[4] = static_cast<int64_t>(native.k()/2) - static_cast<int64_t>(bk);
		    el_idx[1] = el_idx[0] + bi;
		    el_idx[3] = el_idx[2] + bj;
		    el_idx[5] = el_idx[4] + bk;

		    //periodic wrap: only relevant when bi=1 (the low candidate was actually computed) and it went negative
		    if constexpr (Period&0b001) {if (bi==1 && el_idx[0]<0) {el_idx[0]=n_el-1;} }
		    if constexpr (Period&0b010) {if (bj==1 && el_idx[2]<0) {el_idx[2]=n_el-1;} }
		    if constexpr (Period&0b100) {if (bk==1 && el_idx[4]<0) {el_idx[4]=n_el-1;} }

		    for (uint64_t di=0; di<=bi; ++di) {
		        if (el_idx[di]<0) { continue; }
		        for (uint64_t dj=0; dj<=bj; ++dj) {
		            if (el_idx[2+dj]<0) { continue; }
		            for (uint64_t dk=0; dk<=bk; ++dk) {
		                if (el_idx[4+dk]<0) { continue; }
		                P_Elem_t elem{dd, static_cast<uint64_t>(el_idx[di]),
		                                  static_cast<uint64_t>(el_idx[2+dj]),
		                                  static_cast<uint64_t>(el_idx[4+dk])};
		                if (!elem.exists()) { continue; }
		                if (is_active(Elem_t{elem})) {return false; }
		            }
		        }
		    }
		    return true;
		}


		[[nodiscard]] uint8_t min_active_depth() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(n_elements()>0);
			for (int dd=0; dd<=(int)max_depth; ++dd) {
				if (sorter.bin_size(dd) > 0) {return (uint8_t)dd;}
			}
			GUTIL_ERROR("there were no elements in the sorter");
			return 0;
		}

		[[nodiscard]] uint8_t max_active_depth() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(n_elements()>0);
			for (int dd= (int)max_depth; dd>=0; --dd) {
				if (sorter.bin_size((int)dd) > 0) {return (uint8_t)dd;}
			}
			GUTIL_ERROR("there were no elements in the sorter");
			return 0;
		}

		template<uint8_t Period> requires(Period<8)
		[[nodiscard]] Keys::VoxelVertex<Period> lowest_with_active_element(Keys::VoxelVertex<Period> vtx) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			//descend to root, check for active elements on the way up
			
			GUTIL_ASSERT(vtx.is_valid());
			const uint8_t min_depth = min_active_depth();
			const uint8_t max_depth = max_active_depth();
			while(vtx.depth()>min_depth) {vtx = vtx.parent();}
			GUTIL_ASSERT(vtx.is_valid() && vtx.depth()<=min_depth);

			Keys::VoxelElement<Period> els[8];
			while (vtx.depth()>=max_depth) {
				vtx.elements_simd(els);
				for (int i=0; i<8; ++i) {
					if (is_active_stable(static_cast<Elem_t>(els[i]))) { return vtx;}
				}
				vtx = vtx.child();
			}
			return Keys::VoxelVertex<Period>::None();
		}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods primarily for writing to vtk files
		/////////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] GeoPoint_t geo_coord(const Vert_t vtx) const noexcept {return box.low + (box.high-box.low)*vtx.normalized_coordinate();}
		[[nodiscard]] GeoPoint_t geo_center(const Elem_t el) const noexcept {return box.low + (box.high-box.low)*el.normalized_center();}
		
		[[nodiscard]] size_t n_elements() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return active_elements.size();
		}

		[[nodiscard]] size_t n_vertices() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			return tracked_vertices.size();
		}

		[[nodiscard]] size_t vertex_index(Vert_t vtx) const noexcept {
			//the unstructured layers must be up to date and the vertices collected
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			GUTIL_ASSERT(vtx.is_valid());
			vtx.reduced_key_simd_in_place();
			int bn = static_cast<int>(vtx.depth());
			std::span<const Vert_t> list = vertex_sorter.get_bin(bn);
			auto it = std::lower_bound(list.begin(), list.end(), vtx);
			return (it==list.end() || *it!=vtx) ? size_t(-1) : vertex_sorter.bin_start(bn) + static_cast<size_t>(std::distance(list.begin(), it));
		}



		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods for manipulating the mesh
		/////////////////////////////////////////////////////////////////////////////////////////////////
		void request_refine(Elem_t el) const noexcept {
			//this is only a request and does not alter the mesh
			//we only care about the safety of adding to the request list
			GUTIL_ASSERT(el.is_valid());
			GUTIL_ASSERT(el.depth()<max_depth);
			el = el.encode();
			std::lock_guard<std::mutex> lock(request_mutex);
			request_refine_list[el.depth()].push_back(el);
		}
		
		template<typename Predicate = std::nullptr_t>
		void request_unrefine(Elem_t el, Predicate&& pred=nullptr) const noexcept {
			//this is only a request and does not alter the mesh
			//we only care about the safety of adding to the request list
			GUTIL_ASSERT(el.is_valid());
			GUTIL_ASSERT(el.depth()>0);
			
			//no sibling may have an active descendant
			//all active siblings must satisfy the predicate
			const uint8_t dd = el.depth_u8();
			el = el.encode();
			Elem_t sib = Elem_t{el.siblings_simd()};
			for (int i=0; i<8; ++i, ++sib) {
				if (read_depth_field(sib) > dd) {return;}
				if constexpr (!NULLPTR_T<Predicate>) {
					if (is_active(sib) && !pred(sib)) {return;}
				}
			}
			//the cell that we want to be active is tracked
			std::lock_guard<std::mutex> lock(request_mutex);
			request_unrefine_list[dd-1].push_back(el.parent());
		}

		template<typename Predicate>
		void request_refine(Predicate&& pred) const noexcept {
			std::lock_guard<std::mutex> lock(request_mutex);
			auto job = [pred,this](std::span<const uint64_t> list, uint8_t dd) {
				std::span<const Elem_t> e_list = BASE::reinterpret_key_span<Elem_t,uint64_t>(list);
				for (Elem_t el : e_list) {
					if (pred(el)) {request_refine_list[dd].push_back(el);}
				}
			};

			for (uint8_t dd=0; dd<max_depth; ++dd) {
				threads.submit(job, sorter.get_bin(dd), dd);
			}
			threads.wait_idle();
		}

		template<typename Predicate>
		void request_unrefine(Predicate&& pred) const noexcept {
			std::lock_guard<std::mutex> lock(request_mutex);
			auto job = [pred,this](std::span<const Elem_t> list, uint8_t dd) {
				for (Elem_t el : list) {
					el = el.encode();
					Elem_t sib = el.siblings_simd();
					for (int i=0; i<8; ++i, ++sib) {
						if (read_depth_field_stable(sib) > dd) {continue;}
						if (is_active_stable(sib) && !pred(sib)) {continue;}
					}
					request_unrefine_list[dd-1].push_back(el.parent());
				}
			};

			for (uint8_t dd=1; dd<=max_depth; ++dd) {
				threads.submit(job, sorter.get_bin(dd), dd);
			}
			threads.wait_idle();
		}

		void synchronize_depth_field() noexcept;
		void propagate_depth_field(Elem_t el) noexcept;
		void process_unrefine() noexcept;

		template<typename Predicate = std::nullptr_t>
		void process_refine(Predicate&& pred = nullptr) noexcept;


		[[nodiscard]] std::vector<Elem_t> neighbors(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid())
			GUTIL_ASSERT(is_active_no_check(el))

			const uint64_t dd = el.depth();
			if (!is_active_no_check(el)) { return {}; }

			//a coarse cell at depth dd surrounded by cells at depth dd+1 will have 54 neighbors
			std::vector<Elem_t> list;
			list.reserve(54);

			for (Elem_t nbr : el.neighbors()) {
				if (!nbr.is_valid()) { continue; }

				if ( is_active_no_check(nbr) ) { list.push_back(nbr); }
				else if ( dd>0 && is_active_no_check(nbr.parent())) { list.push_back(nbr.parent()); }
				else if ( dd<max_depth ) {
					
					//to only get the correct fine neighbors, we need to be careful
					//for each axis, if el's index is even (low), then the low depth-dd-neighbor's children
					//must have an odd (high) axis index. Similar rules apply to corner and other neighbors.
					
					//track if the nbr child needs low/high/any bits for each axis
					//note that el.i/j/k and nbr.i/j/k are at most one apart
					const int64_t di = static_cast<int64_t>(nbr.i()) - static_cast<int64_t>(el.i());
					const int64_t dj = static_cast<int64_t>(nbr.j()) - static_cast<int64_t>(el.j());
					const int64_t dk = static_cast<int64_t>(nbr.k()) - static_cast<int64_t>(el.k());

					for (Elem_t c : nbr.children()) {
						if (!is_active_no_check(c)) {continue;}

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


		///////////////////////////////////////////////////////////////////////
		/// Some convenient methods for writing to vtk files.
		/// See /mesh/vtk_file_io.hpp for factories to make the lookup functions
		///////////////////////////////////////////////////////////////////////
		void save_as_ascii(const std::string& filename, const std::string& description = "") const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			print_topology_vtk<Mesh_t,true>(filename, *this, description);
		}
		void save_as_binary(const std::string& filename, const std::string& description = "") const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			print_topology_vtk<Mesh_t,false>(filename, *this, description);
		}

		template<typename... Lookup_ts>
		void append_cell_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			append_cell_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		void append_cell_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			append_cell_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		void append_point_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			append_point_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		void append_point_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			append_point_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Iterators for element access
		/////////////////////////////////////////////////////////////////////////////////////////////////
		auto element_begin() 			 const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return active_elements.cbegin();
		}
		auto element_end()   			 const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return active_elements.cend();
		}
		auto element_begin(uint8_t dd)  const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return sorter.begin(static_cast<int>(dd));
		}
		auto element_end(uint8_t dd)    const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return sorter.end(static_cast<int>(dd));
		}

		auto element_begin() 			 {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return active_elements.begin();
		}
		auto element_end()   			 {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return active_elements.end();
		}
		auto element_begin(uint8_t dd)  {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return sorter.begin(static_cast<int>(dd));
		}
		auto element_end(uint8_t dd)    {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return sorter.end(static_cast<int>(dd));
		}
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Iterators for vertex access
		/////////////////////////////////////////////////////////////////////////////////////////////////
		auto vertex_begin() 			 const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return tracked_vertices.cbegin();
		}
		auto vertex_end()   			 const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return tracked_vertices.cend();
		}
		auto vertex_begin(uint64_t dd)  const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return vertex_sorter.begin(static_cast<int>(dd));
		}
		auto vertex_end(uint64_t dd)    const {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return vertex_sorter.end(static_cast<int>(dd));
		}

		auto vertex_begin() 			 {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return tracked_vertices.begin();
		}
		auto vertex_end()   			 {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			return tracked_vertices.end();
		}
		auto vertex_begin(uint64_t dd)  {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return vertex_sorter.begin(static_cast<int>(dd));
		}
		auto vertex_end(uint64_t dd)    {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(dd<=max_depth);
			return vertex_sorter.end(static_cast<int>(dd));
		}
	};


	//////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Implementations
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T>
	void UnstructuredVoxelMesh<T>::process_unrefine() noexcept {
		GUTIL_ASSERT(is_current());
		GUTIL_TIMER("process_unrefine : ", n_elements(), " current elements");
		{
			GV_BEGIN_UNSTABLE
			is_vertices_collected_.store(false);
			std::lock_guard<std::mutex> lock(request_mutex);	//don't allow incoming requests
			{
				//ensure the unrefine lists don't contain duplicates
				//additionally, add elements to unrefine so that the 2-1 refinement rule is respected
				//note that elements marked as 'unrefine' are elements that should be activated and
				//if they have no active descendant, the request is removed.
				auto job = [&](uint8_t dd) {
					BASE::sort_and_unique(request_unrefine_list[dd]);
					//this is only a valid unrefinement target if it is not active and its depth field is greater
					//than its own depth. The latter guarantees the former as the elements are disjoint.
					//additionally, rather than adding cells to unrefine, we only refine cells that will still
					//satisfy the 2-1 rule.
					std::erase_if(request_unrefine_list[dd], [&,dd](Elem_t el) {
						if (read_depth_field(el) <= dd) {return true;}
						for (Elem_t c : el.children()) {
							if (!is_active(c)) { continue; }
							for (Elem_t nbr : neighbors(c)) {
								if (nbr.depth() > static_cast<uint64_t>(dd+1)) {
									return true; //activating will break 2-1	
								}
							}
						}
						return false; //we are ok to activate
					});
				};

				for (uint8_t dd=0; dd<=max_depth; ++dd) {
					threads.submit(job, dd);
				}
				threads.wait_idle();
			}
			
			
			const int n_threads = GUTIL_OMP_TERNARY(omp_get_max_threads()/2, 1);
			
			//we can process in 2 parallel batches due to the 2-1 rule
			//deactivate specified elements and its siblings at level dd,
			//then activate its parent at level dd-1

			//In order to update the depth field, track which elements were newely activated/deactivated.
			std::vector<std::vector<std::vector<Elem_t>>> updated(n_threads);	//thread_index -> depth_index -> element_index
			for (auto& list : updated) {list.resize(max_depth+1);}
			{
				auto job = [&](uint64_t dd) {
					GUTIL_OMP(parallel num_threads(n_threads))
					{
						auto& list = updated[omp_get_thread_num()][dd];
						GUTIL_OMP(for schedule(guided,512))
						for (size_t i=0; i<request_unrefine_list[dd].size(); ++i) {
							Elem_t el = request_unrefine_list[dd][i];
							auto chi = el.children();
							for (Elem_t c : chi) {set_active(c, false);}
							set_active(el, true);
							list.push_back(el);
							list.insert(list.end(), chi.begin(), chi.end());
						}
					}
					request_unrefine_list[dd].clear();
				};

				for (uint8_t dd=0; dd<max_depth; dd+=2) {threads.submit(job,dd);}
				threads.wait_idle();

				for (uint8_t dd=1; dd<max_depth; dd+=2) {threads.submit(job,dd);}
				threads.wait_idle();
			}

			//join per-thread activation results
			for (uint8_t dd=0; dd<=max_depth; ++dd) {
				for (int tid=1; tid<n_threads; ++tid) {
					updated[0][dd].insert(updated[0][dd].end(), 
						std::make_move_iterator(updated[tid][dd].begin()), std::make_move_iterator(updated[tid][dd].end()));
				}
			}

			// update depth field
			for (int8_t dd=max_depth; dd>=0; --dd) {
				for (Elem_t el : updated[0][dd]) {
					propagate_depth_field(el);
				}
			}

			GV_END_UNSTABLE
		}
		collect_elements();
	}

	template<typename T>
	template<typename Predicate>
	void UnstructuredVoxelMesh<T>::process_refine(Predicate&& pred) noexcept {
		GUTIL_ASSERT(is_current());
		GUTIL_TIMER("process_refine : ", n_elements(), " current elements");
		{
			GV_BEGIN_UNSTABLE
			is_vertices_collected_.store(false);
			std::lock_guard<std::mutex> lock(request_mutex);	//don't allow incoming requests

			//ensure the refine lists don't contain duplicates
			//additionally, add elements to refine so that the 2-1 refinement rule is respected
			//note that elements at the max_depth cannot be refined and elements at depth 0
			//cannot have neighbors that are 'too coarse'
			for (uint8_t dd=max_depth-1; dd>=1; --dd) {
				BASE::sort_and_unique(request_refine_list[dd]);
				for (Elem_t el : request_refine_list[dd]) {
					for (Elem_t nbr : neighbors(el)) {
						if (nbr.depth_u8() == dd-1) {
							request_refine_list[dd-1].push_back(nbr);
						}
					}
				}
			}
			BASE::sort_and_unique(request_refine_list[0]);

			//we can process in 2 parallel batches due to the 2-1 rule
			//deactivate specified elements at level dd,
			//then activate its children that satisfy the predicate at level dd+1
			//
			//In order to update the depth field, track which elements were newely activated/deactivated.
			const int n_threads = GUTIL_OMP_TERNARY(omp_get_max_threads()/2, 1);
			std::vector<std::vector<std::vector<Elem_t>>> updated(n_threads);
			for (auto& list : updated) {list.resize(max_depth+1);}
			auto job = [&](uint8_t dd) {
				GUTIL_OMP(parallel num_threads(n_threads))
				{	
					const int tid = GUTIL_OMP_TERNARY(omp_get_thread_num(),1);

					auto& list_d0 = updated[tid][dd];
					auto& list_d1 = updated[tid][dd+1];
					GUTIL_OMP(for schedule(guided,512))
					for (size_t i=0; i<request_refine_list[dd].size(); ++i) {
						Elem_t el = request_refine_list[dd][i];
						set_active(el, false);
						list_d0.push_back(el);

						if constexpr (NULLPTR_T<Predicate>) {
							auto chi = el.children();
							for (Elem_t c : chi) {set_active(c, true);}
							list_d1.insert(list_d1.end(), chi.begin(), chi.end());
						}
						else {
							for (Elem_t chi : el.children()) {
								if (pred(chi)) {
									set_active(chi,true);
									list_d1.push_back(chi);
								}
							}
						}
					}
				}
				request_refine_list[dd].clear();
			};

			for (uint8_t dd=0; dd<max_depth; dd+=2) {threads.submit(job, dd);}
			threads.wait_idle();

			for (uint8_t dd=1; dd<max_depth; dd+=2) {threads.submit(job,dd);}
			threads.wait_idle();

			//join per-thread activation results
			for (uint8_t dd=0; dd<=max_depth; ++dd) {
				for (int tid=1; tid<n_threads; ++tid) {
					updated[0][dd].insert(updated[0][dd].end(), 
						std::make_move_iterator(updated[tid][dd].begin()), std::make_move_iterator(updated[tid][dd].end()));
				}
			}

			// update depth field
			for (int8_t dd=max_depth; dd>=0; --dd) {
				for (Elem_t el : updated[0][dd]) {
					propagate_depth_field(el);
				}
			}

		GV_END_UNSTABLE
		}
		collect_elements();
	}

	template<typename T>
	void UnstructuredVoxelMesh<T>::synchronize_depth_field() noexcept {
		GV_ASSERT_ACTIVE_KEYS_UNSTABLE_STATE
		GUTIL_TIMER("synchronize_depth_field : ", max_possible_elements, " elements to check");

		auto action = [&](auto idx, auto dd) {
			if (dd==max_depth) {
				set_depth_field(idx, is_active(idx) ? dd : 0);
				return;
			}

			if (is_active(idx)) {set_depth_field(idx,dd);}
			else {
				uint8_t cd = 0;
				for (Elem_t c : Elem_t{static_cast<uint64_t>(dd),idx}.children()) {
					cd = std::max(cd, read_depth_field(c));
				}
				set_depth_field(idx,cd);
			}
		};

		for (uint8_t dd=max_depth+1; dd>0; --dd) {
			for_each_index_simd(action, dd-1);
		}
	}

	template<typename T>
	void UnstructuredVoxelMesh<T>::propagate_depth_field(Elem_t el) noexcept {
		GV_ASSERT_ACTIVE_KEYS_UNSTABLE_STATE
		const bool now_active = is_active_unstable(el);
		const uint8_t dd = el.depth_u8();
		uint8_t new_depth = now_active ? dd : 0;
		if (!now_active) {
			for (Elem_t c : el.children()) {
				if (c.exists() && c.depth()<=max_depth) {
					new_depth = std::max(new_depth, read_depth_field_unstable(c));
				}
			}
		}
		set_depth_field(el, new_depth);

		Elem_t cur = el;
		while (cur.depth() > 0) {
			Elem_t parent = cur.parent();
			uint8_t pd = parent.depth_u8();
			uint8_t old_val = read_depth_field_unstable(parent);
			uint8_t new_val = is_active_unstable(parent) ? pd : 0;
			if (!is_active_unstable(parent)) {
				for (Elem_t c : parent.children()) { new_val = std::max(new_val, read_depth_field_unstable(c)); }
			}
			if (new_val == old_val) break;   // nothing above this can change either
			set_depth_field(parent, new_val);
			cur = parent;
		}
	}
}
	









