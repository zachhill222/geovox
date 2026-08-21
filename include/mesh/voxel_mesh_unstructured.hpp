#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "simd_keys/simd_keys.hpp"

#include "mesh/vtk_file_io.hpp"

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
		const uint8_t								max_depth;
		const uint64_t								max_possible_elements;
		
		const Box_t 								box;						//physical extents of the domain
		const GeoPoint_t 							diag;						//diagonal/sidelength of the domain
		const GeoPoint_t 							inv_diag;					//reciprocal of the sidelength
		
		std::vector<Vert_t> 						tracked_vertices{};			//a compressed list of 'active' vertices, primarily used for visualization. reduced keys are stored.
		gutil::BinSort<Vert_t>						vertex_sorter{};			//sort the vertices for better lookup and deduplication
		mutable std::atomic<bool>					is_vertices_collected_{false};	
		mutable std::atomic<bool>					is_depth_explicitly_correct_{false};

		protected:
		std::span<Elem_t>							active_elements{};			//a view into BASE::active_keys (std::vector<uint64_t>) re-interpreted as Elem_t
		using BASE::key_mask;													//vector<uint8_t> of size max_possible_elements
		using BASE::threads;
		using BASE::sorter;														//the bins of this sorter will correspond to each depth in the mesh, will need to re-interpret the spans

		mutable std::mutex							request_mutex{};			//sync refine requests. TODO: make requests per-thread-per-depth if it's slow
		mutable std::vector<std::vector<Elem_t>> 	request_refine_list{};		//allow classes with const-ref to request refinement
		mutable std::vector<std::vector<Elem_t>>	request_unrefine_list{};	//allow classes with const-ref to request unrefinement
		std::atomic<bool>							is_elements_encoded_{true};	//track if the elements in active_keys are in morton(true) or cartesian(false) form.
		std::atomic<bool>							is_elements_depth_sorted_{false};	//track if the elements are sorted by color or depth
		std::atomic<bool>							is_elements_color_sorted_{false};	//track if the elements are sorted by color or depth


		public:
		using BASE::is_key_mask_unstable;										//synchronization tools used in macros
		using BASE::is_key_mask_stable;											//exposing to external classes can be helpul
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

		//a few helper methods to check that we haven't forgot to call something like collect_vertices etc.
		[[nodiscard]] bool is_vertices_collected() const noexcept {return is_vertices_collected_.load();}
		[[nodiscard]] bool is_active_elements_linked() const noexcept {return are_spans_same_data(active_elements,BASE::active_keys);}
		[[nodiscard]] bool is_encoded() const noexcept {return is_elements_encoded_.load();}
		[[nodiscard]] bool is_depth_sorted() const noexcept {return is_elements_depth_sorted_.load();}
		[[nodiscard]] bool is_color_sorted() const noexcept {return is_elements_color_sorted_.load();}
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
		
		~UnstructuredVoxelMesh() {
			std::lock_guard<std::mutex> lock(request_mutex);
			threads.wait_idle();
		}
		UnstructuredVoxelMesh(const UnstructuredVoxelMesh&) = delete;
		UnstructuredVoxelMesh& operator=(const UnstructuredVoxelMesh&) = delete;
		UnstructuredVoxelMesh& operator=(UnstructuredVoxelMesh&&) = delete;
		
		UnstructuredVoxelMesh(UnstructuredVoxelMesh&& other) noexcept :
			BASE(std::move(other)),
			max_depth{other.max_depth},
			max_possible_elements{other.max_possible_elements},
			box{other.box},
			diag{other.diag},
			inv_diag{other.inv_diag},
			tracked_vertices{std::move(other.tracked_vertices)},
			vertex_sorter{std::move(other.vertex_sorter)},
			is_vertices_collected_{other.is_vertices_collected_.load()},
			active_elements{BASE::reinterpret_key_span<Elem_t,uint64_t>(std::span<uint64_t>(BASE::active_keys))},
			request_refine_list{std::move(other.request_refine_list)},
			request_unrefine_list{std::move(other.request_unrefine_list)}
			{}
		


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

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t static constexpr read_depth_from_byte(uint8_t byte) noexcept {
			return (byte&DEPTH_BITS)>>1;
		}
		[[nodiscard]] uint8_t read_depth_field(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return read_depth_from_byte(get_mask(el));
		}
		[[nodiscard]] uint8_t read_depth_field_stable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return read_depth_from_byte(get_mask_stable(el));
		}
		[[nodiscard]] uint8_t read_depth_field_unstable(Elem_t el) const noexcept {
			GUTIL_ASSERT(el.is_valid()); return read_depth_from_byte(get_mask_unstable(el));
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
		static constexpr int n_colors() noexcept {return 54;}

		GUTIL_DECLARE_SIMD()
		static constexpr int element_key_color54_bin(uint64_t key) noexcept {
			return static_cast<int>(Elem_t::color_simd(key));
		}

		GUTIL_DECLARE_SIMD()
		static constexpr int element_key_depth_bin(uint64_t key) noexcept {
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
		UnstructuredVoxelMesh get_depth_slice_as_mesh(uint8_t min_d, uint8_t max_d) const noexcept {
			GV_BEGIN_STABLE

			GUTIL_ASSERT(min_d <= max_d && max_d<=max_depth);
			UnstructuredVoxelMesh result(box, max_d);

			//copy the mask
			GUTIL_ASSERT(key_mask.size() == max_possible_elements);
			GUTIL_ASSERT(result.key_mask.size() == result.max_possible_elements);
			GUTIL_ASSERT(result.max_possible_elements <= max_possible_elements);
			std::copy(key_mask.begin(), key_mask.begin()+result.max_possible_elements, result.key_mask.begin());

			//deactivate any elements below the requested depth
			const size_t min_depth_start = Elem_t::elements_below_depth(min_d);
			GUTIL_SIMD()
			for (size_t i=0; i<min_depth_start; ++i) {
				result.key_mask[i]&=~ACTIVE_BIT;
			}

			GV_END_STABLE
			return result;
		}

		void set_depth(uint8_t depth) noexcept {
			if (depth>max_depth) {GUTIL_ABORT("depth too large");}
			{
				GV_BEGIN_UNSTABLE
				BASE::clear();

				uint64_t start = Elem_t::elements_below_depth(depth);
				uint64_t end   = Elem_t::elements_below_depth(depth+1);
				uint8_t depth_mask = (depth<<1)&DEPTH_BITS;
				std::fill(key_mask.begin()+start, key_mask.begin()+end, ACTIVE_BIT|depth_mask);
				std::fill(key_mask.begin(), key_mask.begin()+start, depth_mask);
				GV_END_UNSTABLE
			}
			collect_elements();
			GUTIL_ASSERT(is_current() && is_depth_field_correct());
		}

		[[nodiscard]] std::span<const Elem_t> get_depth(uint8_t depth) const noexcept {
			GUTIL_ASSERT(is_depth_sorted());
			GUTIL_ASSERT(depth<=max_depth);
			return BASE::reinterpret_key_span<Elem_t,uint64_t>(sorter.get_bin((int) depth));
		}

		[[nodiscard]] std::span<Elem_t> get_depth(uint8_t depth) noexcept {
			GUTIL_ASSERT(is_depth_sorted());
			GUTIL_ASSERT(depth<=max_depth);
			return BASE::reinterpret_key_span<Elem_t,uint64_t>(sorter.get_bin((int) depth));
		}

		[[nodiscard]] std::span<const Elem_t> get_color(int color) const noexcept {
			GUTIL_ASSERT(is_color_sorted());
			GUTIL_ASSERT(color<=54);
			return BASE::reinterpret_key_span<Elem_t,uint64_t>(sorter.get_bin(color));
		}

		[[nodiscard]] std::span<Elem_t> get_color(int color) noexcept {
			GUTIL_ASSERT(is_color_sorted());
			GUTIL_ASSERT(color<=54);
			return BASE::reinterpret_key_span<Elem_t,uint64_t>(sorter.get_bin(color));
		}

		void collect_elements() noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GV_ASSERT_KEY_MASK_STABLE_STATE
			GUTIL_PROFILE("collecting active elements");
			BASE::collect_active_keys<Elem_t>();
			active_elements = BASE::reinterpret_key_span<Elem_t,uint64_t>(BASE::active_keys);
			sort_elements_by_depth();
			GUTIL_ASSERT(is_current());
			is_vertices_collected_.store(false);
			is_elements_encoded_.store(true);
		}

		void sort_elements_by_depth() noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GV_ASSERT_KEY_MASK_STABLE_STATE

			GUTIL_PROFILE("sorting elements by depth");
			BASE::sort_active_keys((int) max_depth+1, &UnstructuredVoxelMesh::element_key_depth_bin);
			is_elements_color_sorted_.store(false);
			is_elements_depth_sorted_.store(true);
		}

		void sort_elements_by_color() noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GV_ASSERT_KEY_MASK_STABLE_STATE

			const bool old_encoding = is_encoded();
			set_encoded(false);
			GUTIL_ASSERT(!is_encoded());

			GUTIL_PROFILE("sorting elements by color");
			BASE::sort_active_keys(54, &UnstructuredVoxelMesh::element_key_color54_bin);
			is_elements_color_sorted_.store(true);
			is_elements_depth_sorted_.store(false);

			set_encoded(old_encoding);
		}

		void collect_vertices() noexcept {
			GV_BEGIN_STABLE  	//The base class+elements must be stable, we are only altering the vertices
			GUTIL_PROFILE("Collecting active vertices");
			GUTIL_ASSERT(is_current());
			is_vertices_collected_.store(false);

			// go through active elements and collect their vertices
			// reduce the vertex index (eg., (3,4,2,6) -> (2,2,1,3)) for deduplication by geometric coordinates
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

				auto it = gutil::sort_and_unique(verts);
				verts.erase(it, verts.end());
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

			//ensure no duplicates from different threads
			auto it = gutil::sort_and_unique(tracked_vertices, threads);
			threads.wait_idle();
			tracked_vertices.erase(it,tracked_vertices.end());

			GUTIL_ASSERT(std::unique(tracked_vertices.begin(), tracked_vertices.end())==tracked_vertices.end());

			vertex_sorter = gutil::BinSort<Vert_t>(tracked_vertices, max_depth+1);
			GUTIL_ASSERT(vertex_sorter.n_bins() == max_depth+1);
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

		//change the encoding of the active elements
		void set_encoded(bool val) noexcept {
			
			#ifndef NDEBUG
				if (is_encoded() == val) {
					for (size_t i=0; i<std::min(size_t{100}, active_elements.size()); ++i) {
						
						if (active_elements[i].is_encoded()!=val) {
							std::cout << print_bytes(active_elements[i]) << std::endl;
						}
						GUTIL_ASSERT(active_elements[i].is_encoded() == val);
					}
				}
			#endif
			
			if (is_encoded() == val) {return;}


			GV_BEGIN_ACTIVE_UNSTABLE
			GUTIL_PROFILE("Setting encoding to ", val ? "morton" : "cartesian");

			if (val) { //cartesian->morton
				dispatch_parallel_active_keys(
					[val](std::span<uint64_t> keys) {
					GUTIL_SIMD()
					for (size_t i=0; i<keys.size(); ++i) {
						keys[i] = Elem_t::encode_simd(keys[i]);
						GUTIL_ASSERT(Elem_t{keys[i]}.is_encoded()==val);
						GUTIL_ASSERT(Elem_t{keys[i]}.is_valid());
					}
				});
			}
			else {	//morton->cartesian
				dispatch_parallel_active_keys(
					[val](std::span<uint64_t> keys) {
					GUTIL_SIMD()
					for (size_t i=0; i<keys.size(); ++i) {
						keys[i] = Elem_t::decode_simd(keys[i]);
						GUTIL_ASSERT(Elem_t{keys[i]}.is_encoded()==val);
						GUTIL_ASSERT(Elem_t{keys[i]}.is_valid());
					}
				});
			}
			threads.wait_idle();
			is_elements_encoded_.store(val);
			GV_END_ACTIVE_UNSTABLE
		}


		//select the subset of the active elements that satisfy some predicate
		template<typename Predicate> requires(std::is_invocable_r_v<bool,Predicate,Elem_t>)
		std::vector<Elem_t> select_elements(Predicate&& pred) const noexcept {
			//collect elements in parallel per-thread
			std::vector<std::vector<Elem_t>> thread_elems(threads.n_threads());
			BASE::template dispatch_parallel_active_keys_const<Elem_t>([pred, &thread_elems](std::span<const Elem_t> list, int tid) {
				for (auto it=list.begin(); it!=list.end(); ++it)
				if (pred(*it)) {thread_elems[tid].push_back(*it);}
			});
			threads.wait_idle();

			//join the per-thread results
			size_t count=0;
			for (auto& list : thread_elems) {count+=list.size();}

			std::vector<Elem_t> result; result.reserve(count);
			for (auto& list : thread_elems) {
				result.insert(result.end(), std::make_move_iterator(list.begin()), std::make_move_iterator(list.end()));
				list.clear();
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
						if (is_active(Elem_t{elem})) {
							return false;
						}
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
			const uint8_t min_d = min_active_depth();
			const uint8_t max_d = max_active_depth();
			while(vtx.depth()>min_d) {vtx = vtx.parent();}
			GUTIL_ASSERT(vtx.is_valid() && vtx.depth()<=min_d);

			Keys::VoxelElement<Period> els[8];
			while (vtx.depth()>=max_d) {
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
		[[nodiscard]] GeoPoint_t geo_coord(Vert_t vtx) const noexcept {return box.low + (box.high-box.low)*vtx.template normalized_coordinate<Scalar_t>();}
		[[nodiscard]] GeoPoint_t geo_center(Elem_t el) const noexcept {return box.low + (box.high-box.low)*el.template normalized_center<Scalar_t>();}
		[[nodiscard]] GeoPoint_t el_size(Elem_t el) const noexcept {
			return gutil::ldexp(Scalar_t{1}, -(int)(el.depth())) * diag;
		}
		[[nodiscard]] GeoPoint_t el_size_inv(Elem_t el) const noexcept {
			return gutil::ldexp(Scalar_t{1}, (int)(el.depth())) * inv_diag;
		}

		[[nodiscard]] size_t n_elements() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			return active_elements.size();
		}

		[[nodiscard]] size_t n_vertices() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			return tracked_vertices.size();
		}

		[[nodiscard]] size_t vertex_index(Vert_t vtx) const noexcept {
			//the unstructured layers must be up to date and the vertices collected
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			GUTIL_ASSERT(is_vertices_collected());
			GUTIL_ASSERT(vtx.is_valid());
			GUTIL_ASSERT(vertex_sorter.n_bins() == max_depth+1);
			vtx.reduced_key_simd_in_place();
			int bn = static_cast<int>(vtx.depth());
			std::span<const Vert_t> list = vertex_sorter.get_bin(bn);
			auto it = std::lower_bound(list.begin(), list.end(), vtx);
			return (it==list.end() || *it!=vtx) ? size_t(-1) : vertex_sorter.bin_start(bn) + static_cast<size_t>(std::distance(list.begin(), it));
		}



		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods for manipulating the mesh
		/////////////////////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool has_pending_refine_requests() const noexcept {
			for (auto& list : request_refine_list) {
				if (list.size()>0) {return true;}
			}
			return false;
		}

		[[nodiscard]] bool has_pending_unrefine_requests() const noexcept {
			for (auto& list : request_unrefine_list) {
				if (list.size()>0) {return true;}
			}
			return false;
		}

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
			GUTIL_ASSERT(is_depth_sorted());
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
			GUTIL_ASSERT(is_depth_sorted());
			std::lock_guard<std::mutex> lock(request_mutex);
			auto job = [pred,this](std::span<const uint64_t> list, uint8_t dd) {
				std::span<const Elem_t> e_list = BASE::reinterpret_key_span<Elem_t,uint64_t>(list);
				for (Elem_t el : e_list) {
					el = el.encode();
					Elem_t sib{el.siblings_simd()};
					bool can_unrefine = true;
					for (int i=0; i<8; ++i, ++sib) {
						if (read_depth_field_stable(sib) > dd) {can_unrefine=false; break;}
						if (is_active_stable(sib) && !pred(sib)) {can_unrefine=false; break;}
					}
					if (can_unrefine) {
						request_unrefine_list[dd-1].push_back(el.parent());
					}
				}
			};

			for (uint8_t dd=1; dd<=max_depth; ++dd) {
				threads.submit(job, sorter.get_bin(dd), dd);
			}
			threads.wait_idle();
		}

		[[nodiscard]] bool is_depth_field_correct() const noexcept;
		void propagate_depth_field(Elem_t el) noexcept;
		

		//if we require the 2-1 rule to be respected periodically, we need to pass the period
		template<uint8_t Period=0, typename Predicate = std::nullptr_t> requires(Period<8)
		void process_refine(Predicate&& pred = nullptr) noexcept;

		template<uint8_t Period=0> requires(Period<8)
		void process_unrefine() noexcept;

		template<uint8_t Period=0> requires(Period<8)
		[[nodiscard]] std::vector<Elem_t> neighbors(Elem_t el_) const noexcept {
			using P_Elem_t = Keys::VoxelElement<Period>;
			P_Elem_t el = static_cast<P_Elem_t>(el_);


			GUTIL_ASSERT(el.is_valid());
			GUTIL_ASSERT(is_active_no_check(Elem_t{el.key}));	

			const uint64_t dd = el.depth();
			const int s_el_i = static_cast<int>(el.i());	//note i/j/k is at most 2^15 - 1
			const int s_el_j = static_cast<int>(el.j());
			const int s_el_k = static_cast<int>(el.k());
			
			[[maybe_unused]] int s_max_idx;
			if constexpr (Period!=0) {
				s_max_idx = static_cast<int>(1<<dd)-1;
			}
			

			// if (!is_active_no_check(el)) { return {}; }

			//a coarse cell at depth dd surrounded by cells at depth dd+1 will have 54 neighbors
			std::vector<Elem_t> list;
			list.reserve(54);

			for (P_Elem_t nbr : el.neighbors()) {
				if (!nbr.is_valid()) { continue; }

				if ( is_active_no_check(Elem_t{nbr.key}) ) { list.emplace_back(nbr.key); }
				else if ( dd>0 && is_active_no_check(Elem_t{nbr.parent().key})) { list.emplace_back(nbr.parent().key); }
				else if ( dd<max_depth ) {
					
					//to only get the correct fine neighbors, we need to be careful
					//for each axis, if el's index is even (low), then the low depth-dd-neighbor's children
					//must have an odd (high) axis index. Similar rules apply to corner and other neighbors.
					
					//track if the nbr child needs low/high/any bits for each axis
					//note that el.i/j/k and nbr.i/j/k are at most one apart
					//if we need to get periodic neighbors, this must be handled here as well.
					//suppose we are periodic in the x-axis:
					// 	if el.i()=n_el-1, then we need the neighbor nbr.i()=0 for the high neighbor 
					//	if el.i()=0, then we need the neighbor nbr.i()=n_el-1 for the low neighbor
					
					//compute indicators from comparing the element and neighbor indices in each axis
					// di>0 means we need the neighbor's high index children (the neighbor is lower than the query element)
					// di<0 means we need the neighbor's low index children
					// di=0 means this axis does not restrict the candidate children.
					const int s_nbr_i = static_cast<int>(nbr.i());
					const int s_nbr_j = static_cast<int>(nbr.j());
					const int s_nbr_k = static_cast<int>(nbr.k());
					int di = s_el_i - s_nbr_i;
					int dj = s_el_j - s_nbr_j;
					int dk = s_el_k - s_nbr_k;

					if constexpr (Period&0b001) {
						if (s_el_i == 0) {di=-1;}	//need the low side of this axis
						else if (s_el_i==s_max_idx) {di=1;}
					}

					if constexpr (Period&0b010) {
						if (s_el_j == 0) {dj=-1;}	//need the low side of this axis
						else if (s_el_j==s_max_idx) {dj=1;}
					}

					if constexpr (Period&0b100) {
						if (s_el_k == 0) {dk=-1;}	//need the low side of this axis
						else if (s_el_k==s_max_idx) {dk=1;}
					}

					// int64_t di = static_cast<int64_t>(nbr.i()) - static_cast<int64_t>(el.i());
					// int64_t dj = static_cast<int64_t>(nbr.j()) - static_cast<int64_t>(el.j());
					// int64_t dk = static_cast<int64_t>(nbr.k()) - static_cast<int64_t>(el.k());

					for (P_Elem_t c : nbr.children()) {
						if (!c.exists() || !is_active_no_check(Elem_t{c.key})) {continue;}

						//check if each axis index is ok (any/low/high)
						if ( (di>0 && !static_cast<bool>(c.i()&1)) || (di<0 && static_cast<bool>(c.i()&1)) ) {continue;}
						if ( (dj>0 && !static_cast<bool>(c.j()&1)) || (dj<0 && static_cast<bool>(c.j()&1)) ) {continue;}
						if ( (dk>0 && !static_cast<bool>(c.k()&1)) || (dk<0 && static_cast<bool>(c.k()&1)) ) {continue;}
						list.emplace_back(c.key);
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
		/// Iterators and other element access
		/////////////////////////////////////////////////////////////////////////////////////////////////
		template<uint8_t Period=0> requires(Period<8)
		std::span<const Keys::VoxelElement<Period>> elements_as_span() const noexcept {
			return BASE::reinterpret_key_span<Keys::VoxelElement<Period>, uint64_t>(BASE::active_keys);
		}


		auto element_begin() 			 const {
			using c_iter = std::span<const Elem_t>::iterator;
			return c_iter{active_elements.data()};
			// return active_elements.cbegin();
		}
		auto element_end()   			 const {
			using c_iter = std::span<const Elem_t>::iterator;
			return c_iter{active_elements.data() + active_elements.size()};
			// return active_elements.cend();
		}
		auto element_begin(uint8_t dd)  const {
			GUTIL_ASSERT(dd<=max_depth && BASE::is_sorted());
			return sorter.begin(static_cast<int>(dd));
		}
		auto element_end(uint8_t dd)    const {
			GUTIL_ASSERT(dd<=max_depth && BASE::is_sorted());
			return sorter.end(static_cast<int>(dd));
		}

		auto element_begin() 			 {
			return active_elements.begin();
		}
		auto element_end()   			 {
			return active_elements.end();
		}
		auto element_begin(uint8_t dd)  {
			GUTIL_ASSERT(dd<=max_depth && BASE::is_sorted());
			return sorter.begin(static_cast<int>(dd));
		}
		auto element_end(uint8_t dd)    {
			GUTIL_ASSERT(dd<=max_depth && BASE::is_sorted());
			return sorter.end(static_cast<int>(dd));
		}
		

		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Iterators for vertex access
		/////////////////////////////////////////////////////////////////////////////////////////////////
		auto vertex_begin() 			 const {
			return tracked_vertices.cbegin();
		}
		auto vertex_end()   			 const {
			return tracked_vertices.cend();
		}
		auto vertex_begin(uint64_t dd)  const {
			GUTIL_ASSERT(dd<=max_depth && is_vertices_collected());
			return vertex_sorter.begin(static_cast<int>(dd));
		}
		auto vertex_end(uint64_t dd)    const {
			GUTIL_ASSERT(dd<=max_depth && is_vertices_collected());
			return vertex_sorter.end(static_cast<int>(dd));
		}

		auto vertex_begin() 			 {
			return tracked_vertices.begin();
		}
		auto vertex_end()   			 {
			return tracked_vertices.end();
		}
		auto vertex_begin(uint64_t dd)  {
			GUTIL_ASSERT(dd<=max_depth && is_vertices_collected());
			return vertex_sorter.begin(static_cast<int>(dd));
		}
		auto vertex_end(uint64_t dd)    {
			GUTIL_ASSERT(dd<=max_depth && is_vertices_collected());
			return vertex_sorter.end(static_cast<int>(dd));
		}
	};//UnstructuredVoxelMesh


	//////////////////////////////////////////////////////////////////////////////////////////////////////
	/// Implementations
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T> 
	template<uint8_t Period> requires(Period<8)
	void UnstructuredVoxelMesh<T>::process_unrefine() noexcept {
		GUTIL_ASSERT(is_current());
		GUTIL_ASSERT(is_depth_sorted());

		if (!has_pending_unrefine_requests()) {
			GUTIL_LOG("there are no unrefine requests");
			return;
		}

		is_elements_color_sorted_.store(false);
		is_elements_depth_sorted_.store(false);

		GUTIL_PROFILE("process_unrefine : ", n_elements(), " current elements");
		{
			GV_BEGIN_UNSTABLE
			is_vertices_collected_.store(false);
			is_depth_explicitly_correct_.store(false);	//set true by checking all elements in debug
			std::lock_guard<std::mutex> lock(request_mutex);	//don't allow incoming requests
			{
				//ensure the unrefine lists don't contain duplicates
				//additionally, add elements to unrefine so that the 2-1 refinement rule is respected
				//note that elements marked as 'unrefine' are elements that should be activated and
				//if they have no active descendant, the request is removed.
				auto job = [&](uint8_t dd) {
					auto it = gutil::sort_and_unique(request_unrefine_list[dd]);
					request_unrefine_list[dd].erase(it, request_unrefine_list[dd].end());
					//this is only a valid unrefinement target if it is not active and its depth field is greater
					//than its own depth. The latter guarantees the former as the elements are disjoint.
					//additionally, rather than adding cells to unrefine, we only refine cells that will still
					//satisfy the 2-1 rule.
					std::erase_if(request_unrefine_list[dd], [&,dd](Elem_t el) {
						if (read_depth_field_unstable(el) <= dd) {return true;}
						for (Elem_t c : el.children()) {
							if (!is_active_unstable(c)) { continue; }
							for (Elem_t nbr : neighbors<Period>(c)) {
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
						const int tid = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
						auto& list = updated[tid][dd];
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
		GUTIL_ASSERT(is_current() && is_depth_field_correct());
	}

	template<typename T>
	template<uint8_t Period, typename Predicate> requires(Period<8)
	void UnstructuredVoxelMesh<T>::process_refine(Predicate&& pred) noexcept {
		GUTIL_ASSERT(is_current());
		GUTIL_ASSERT(is_depth_sorted());

		if (!has_pending_refine_requests()) {
			GUTIL_LOG("there are no refine requests");
			return;
		}

		is_elements_color_sorted_.store(false);
		is_elements_depth_sorted_.store(false);

		GUTIL_PROFILE("process_refine : ", n_elements(), " current elements");
		{
			GV_BEGIN_UNSTABLE
			is_vertices_collected_.store(false);
			is_depth_explicitly_correct_.store(false);	//set true by checking all elements in debug
			std::lock_guard<std::mutex> lock(request_mutex);	//don't allow incoming requests

			//ensure the refine lists don't contain duplicates
			//additionally, add elements to refine so that the 2-1 refinement rule is respected
			//note that elements at the max_depth cannot be refined and elements at depth 0
			//cannot have neighbors that are 'too coarse'
			for (uint8_t dd=max_depth-1; dd>=1; --dd) {
				auto it = gutil::sort_and_unique(request_refine_list[dd]);
				request_refine_list[dd].erase(it, request_refine_list[dd].end());
				for (Elem_t el : request_refine_list[dd]) {
					for (Elem_t nbr : neighbors<Period>(el)) {
						if (nbr.depth_u8() == dd-1) {
							request_refine_list[dd-1].push_back(nbr);
						}
					}
				}
			}
			auto it = gutil::sort_and_unique(request_refine_list[0]);
			request_refine_list[0].erase(it, request_refine_list[0].end());

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
					const int tid = GUTIL_OMP_TERNARY(omp_get_thread_num(),0);

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
		GUTIL_ASSERT(is_current() && is_depth_field_correct());
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

	///////////////////////////////////////////////////////////////////////
	/// Debugging: verify the depth field satisfies its own documented invariant:
	///   - if E is active,                        D(E) == E.depth()
	///   - if E is inactive but has an active     D(E) == max(D(C)) over E's children C
	///     descendant,
	///   - if E is inactive with no active        D(E) == 0
	///     descendants,
	/// Walks bottom-up (deepest first) so a coarse-level mismatch can be
	/// attributed to its own children being wrong first, if that's the root
	/// cause. Logs every mismatch found; returns false if any were found.
	///////////////////////////////////////////////////////////////////////
	template<typename T>
	[[nodiscard]] bool UnstructuredVoxelMesh<T>::is_depth_field_correct() const noexcept {
		GV_ASSERT_KEY_MASK_STABLE_STATE
		if (is_depth_explicitly_correct_.load()) {return true;}
		
		GV_BEGIN_STABLE
		
		size_t count = 0;
		for (uint8_t dd=max_depth+1; dd>0; --dd) {
			const uint8_t depth = dd-1;
			const uint64_t start = Elem_t::elements_below_depth(depth);
			const uint64_t end   = Elem_t::elements_below_depth(depth+1);

			GUTIL_OMP(parallel for reduction(+:count) schedule(static, 1024))
			for (uint64_t idx=start; idx<end; ++idx) {
				//read the actual depth and validate that the mask bits and
				//standard element interface agree.
				const uint8_t byte = key_mask[idx];
				const uint8_t actual = read_depth_from_byte(byte);
				Elem_t el = Elem_t::MakeFromIndex(idx);
				if(read_depth_field(el)!=actual) {
					GUTIL_ERROR("something went wrong. read depth ", actual, " from mask ", print_bytes(byte),
									" at mask index ", idx, ". The corresponding element ", el, " read deapth ",
									read_depth_field(el), " the computed linear index for the element is ", el.linear_index());
					GUTIL_ABORT("elements and mask are out of sync. check that the element to/from linear index is computed correctly");
				}

				uint8_t expected;
				if (is_active_stable(el)) {
					expected = depth;
				}
				else {
					expected = 0;
					for (Elem_t c : el.children()) {
						if (c.exists() && c.depth()<=max_depth) {
							expected = std::max(expected, read_depth_field_stable(c));
						}
					}
				}

				if (actual != expected) {
					GUTIL_ERROR(el, " depth field mismatch: stored=", actual, " expected=", expected);
					++count;
				}
			}
		}
		is_depth_explicitly_correct_.store(count==0);
		GV_END_STABLE
		return count==0;
	}



	template<typename T>
	std::ostream& operator<<(std::ostream& os, const UnstructuredVoxelMesh<T>& mesh) {
		size_t n_used_vert = mesh.n_vertices() * sizeof(typename UnstructuredVoxelMesh<T>::Vert_t);
		size_t n_vert_reserved = mesh.tracked_vertices.capacity() * sizeof(typename UnstructuredVoxelMesh<T>::Vert_t);

		os << "UnstructuredVoxelMesh with maximum depth of " << (int) mesh.max_depth << "\n";
		os << mesh.summary();
		os << gutil::format(mesh.n_vertices(),16) << " tracked vertices (" + format_byte_count(n_used_vert) + " / "
		    							+ format_byte_count(n_vert_reserved) + " used/reserved\n";
		os << gutil::format(mesh.n_elements(),16) << " active elements (keys)\n";

		if (mesh.is_depth_sorted()) {
			for (uint8_t dd=0; dd<=mesh.max_depth; ++dd) {
				if (mesh.get_depth(dd).size()>0) {
					std::cout << "\tdepth " << (int)dd <<" : " << std::to_string(mesh.get_depth(dd).size()) << "\n";
				}
			}
		}


		return os;
	}




}//GV
	









