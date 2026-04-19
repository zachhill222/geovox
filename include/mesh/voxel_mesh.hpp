#pragma once

#include "mesh/keys/voxel_key.hpp"
#include "util/concepts.hpp"
#include "util/point.hpp"

#include <cstdint>
#include <type_traits>
#include <bitset>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//concepts to constrain feature types to exactly match the/a
	//corresponding mesh feature
	template<typename F, typename M>
	concept MeshElementType = VoxelKeyType<F> && std::same_as<F,typename M::VoxelElement>;

	template<typename F, typename M>
	concept MeshVertexType = VoxelKeyType<F> && std::same_as<F,typename M::VoxelVertex>;

	template<typename F, typename M>
	concept MeshFaceType = VoxelKeyType<F> && std::same_as<F,typename M::VoxelFace>;

	template<typename F, typename M>
	concept MeshFeatureType = MeshElementType<F,M> || MeshVertexType<F,M> || MeshFaceType<F,M>;

	template<typename M>
	concept VoxelMeshType = MeshElementType<typename M::VoxelElement, M> && 
							MeshVertexType<typename M::VoxelVertex, M> && 
							MeshFaceType<typename M::VoxelFace, M>;

	//This class is for a hierarchical voxel mesh.
	//This structure allows us to very efficiently store elements, vertices, faces, etc.
	//The base mesh is 1x1x1 so that every vertex is (in reference coordinates) a dyadic rational number
	//Elements, vertices, and faces all have special index/key structs for their storage and logical relations.
	//All information for every element, vertex, and face is compressed into a 64-bit unsigned integer
	//With the element/vertex/face keys, we must have a maximum depth of 15. If more is needed (unlikely) we can stitch together
	//multiples of these meshes.

	template<int MAX_DEPTH_=10, bool MORTON_ORDER=false> requires (MAX_DEPTH_>=0)
	class VoxelMesh
	{
	public:
		//mesh features are never periodic
		using VoxelElement = VoxelElementKey<MAX_DEPTH_+1,0,MORTON_ORDER>;
		using VoxelVertex  = VoxelVertexKey<MAX_DEPTH_+1,0,MORTON_ORDER>;
		using VoxelFace    = VoxelFaceKey<MAX_DEPTH_+1,0,MORTON_ORDER>;
		using Mesh_t       = VoxelMesh<MAX_DEPTH_,MORTON_ORDER>; //this mesh type

		static constexpr uint64_t MAX_DEPTH = MAX_DEPTH_;
		static constexpr uint64_t TOTAL_POSSIBLE_ELEMENTS = total_possible<VoxelElement>(MAX_DEPTH);

		#ifdef _OPENMP
			static constexpr bool OPENMP = true;
		#else
			static constexpr bool OPENMP = false;
		#endif

	protected:
		std::bitset<TOTAL_POSSIBLE_ELEMENTS>* active_elem = new std::bitset<TOTAL_POSSIBLE_ELEMENTS>(0);

		//alow other methods with a const ref to this class to request element to be activated
		//or deactivated. some other class will then process the request.
		//these requests are just the 'const' versions of activate(el) and deactivate(el).
		//they are thread safe in OpenMP.
		mutable std::vector<std::vector<VoxelElement>> request_active;
		mutable std::vector<std::vector<VoxelElement>> request_deactive;

	public:
		using GeoPoint_t = Point<3,double>; //points in space
		
		const GeoPoint_t low;
		const GeoPoint_t high;
		const GeoPoint_t diag;

		VoxelMesh(const GeoPoint_t low_, const GeoPoint_t high_) :
			low{elmin(low_, high_)},
			high{elmax(low_, high_)},
			diag{high-low}
		{
			#ifdef _OPENMP
				request_active.resize(omp_get_max_threads());
				request_deactive.resize(omp_get_max_threads());
			#else
				request_active.resize(1);
				request_deactive.resize(1);
			#endif
		}

		virtual ~VoxelMesh() {delete active_elem;}

		//simple querries and operations
		inline void reset() {active_elem->reset();}
		inline size_t n_elements() const {return active_elem->count();}

		inline void activate(const VoxelElement el) {assert(el.is_valid()); active_elem->set(el.linear_index());}
		inline void activate(const VoxelElement el) const {
			assert(el.is_valid());
			#ifdef _OPENMP
				request_active[omp_get_thread_num()].push_back(el);
			#else
				request_active[0].push_back(el);
			#endif
		}

		inline void deactivate(const VoxelElement el) {assert(el.is_valid()); active_elem->reset(el.linear_index());}
		inline void deactivate(const VoxelElement el) const {
			assert(el.is_valid());
			#ifdef _OPENMP
				request_deactive[omp_get_thread_num()].push_back(el);
			#else
				request_deactive[0].push_back(el);
			#endif
		}

		

		//test if a feature is active.
		//active elements are recorded int the active_elem bitset
		//vertices and faces are active if they belong to an active element
		//note there may be many vertices at the same geometric location but existing at different levels
		//you may need to look at parent/child vertices to get the expected result
		inline bool is_active(const VoxelElement el) const {assert(el.is_valid()); return active_elem->test(el.linear_index());}
		
		template<typename Key_t> requires (MeshFaceType<Key_t,Mesh_t> || MeshVertexType<Key_t,Mesh_t>)
		bool is_active(const Key_t key) const {
			assert(key.is_valid());
			for (const VoxelElement el : key.elements()) {
				if (el.exists() and is_active(el)) {return true;}
			}
			return false;
		}

		//find the vertex with the lowest index that is active and overlaps (geometrically)
		//with the specified vertex
		VoxelVertex first_active(VoxelVertex vtx) const {
			assert(vtx.is_valid());
			//get lowest vertex
			vtx = vtx.reduced_key();
			while (vtx.exists()) {
				if (is_active(vtx)) {return vtx;}
				vtx = vtx.child();
			}
			return vtx;
		}

		//test if a vertex is a hanging node
		//an active vertex is hanging if there is another active vertex at
		//the same geometric location
		bool is_hanging(const VoxelVertex vtx) const {
			assert(is_active(vtx));
			//search the keys at this location from the lowest depth to greatest
			VoxelVertex other = vtx.reduced_key();
			while (other.exists()) {
				if (other!=vtx && is_active(vtx)) {return true;}
			}
			return false;
		}

		inline void set(const VoxelElement el, const bool flag = true) {assert(el.is_valid()); active_elem->set(el.linear_index(), flag);}


		//process requested activations
		void process_request_active() {
			#ifndef _OPENMP
			for (auto& request : request_active) {
				for (VoxelElement el : request) {
					activate(el);
					//check if all the children exist and are active
					//if so, deactivate the element
					bool all_children_active = true;
					for (VoxelElement c : el.children()) {
						if (c.exists() and !is_active(c)) {
							all_children_active = false;
							break;
						}
					}

					if (all_children_active) {
						deactivate(el);
					}
				}
				request.clear();
			}
			#else
			#pragma omp parallel
			{
				auto& request = request_active[omp_get_thread_num()];
				for (VoxelElement el : request) {
					activate(el);
					//check if all the children exist and are active
					//if so, deactivate the element
					bool all_children_active = true;
					for (VoxelElement c : el.children()) {
						if (c.exists() and !is_active(c)) {
							all_children_active = false;
							break;
						}
					}

					if (all_children_active) {
						deactivate(el);
					}
				}
				request.clear();
			}
			#endif
			make_disjoint();
		}

		//process requested deactivations
		void process_request_deactive() {
			for (auto& request : request_deactive) {
				for (VoxelElement el : request) {
					if (el.exists()) {
						deactivate(el);}
				}
				request.clear();
			}
			make_disjoint();
		}

		//layer operations
		template<typename Predicate = std::nullptr_t>
		void set_depth(const uint64_t depth, Predicate&& pred = nullptr) {
			//set to true by default. use the predicate if it is passed
			auto action = [&pred, this](VoxelElement el) {
				if constexpr (!NULLPTR_T<Predicate>) {
					active_elem->set(el.linear_index(), pred(el));
				}
				else {
					active_elem->set(el.linear_index());
				}
			};

			reset();
			//note that the predicate to activate an element is not the same
			//as the predicate to skip an element in the for_each loop
			for_each_depth<VoxelElement>(depth, action);
		}

		
		//dispatch to the *_impl iteration methods for consistency
		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		inline void for_each_depth(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) {
			for_each_depth_impl<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		inline void for_each_depth(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const {
			for_each_depth_impl<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		inline void for_each(Action&& action, bool omp=false, Predicate&& pred = nullptr) {
			for_each_impl<Key_t>(std::forward<Action>(action), omp, std::forward<Predicate>(pred));
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		inline void for_each(Action&& action, bool omp=false, Predicate&& pred = nullptr) const {
			for_each_impl<Key_t>(std::forward<Action>(action), omp, std::forward<Predicate>(pred));
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		inline void for_each_depth_omp(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) {
			for_each_depth_omp_impl<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
		}

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		inline void for_each_depth_omp(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const {
			for_each_depth_omp_impl<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
		}

		
		//hierarchy
		template<typename Predicate = std::nullptr_t>
		void refine(const VoxelElement el, Predicate&& pred = nullptr) {
			deactivate(el);
			for (const auto child : el.children()) {
				if constexpr (!NULLPTR_T<Predicate>) {
					set(child, pred(child));
				}
				else {
					activate(child);
				}
			}
		}

		template<typename Predicate = std::nullptr_t>
		void coarsen(const VoxelElement el, Predicate&& pred = nullptr) {
			if constexpr (!NULLPTR_T<Predicate>) {
				set(el, pred(el));
			}
			else {
				activate(el);
			}

			for (const auto child : el.children()) {deactivate(child);}
		}

		bool has_active_child(const VoxelElement el) const {
			assert(el.is_valid());
			for (const VoxelElement child : el.children()) {
				if (child.exists() and is_active(child)) {
					return true;
				}
			}
			return false;
		}

		template<typename Predicate = std::nullptr_t>
		bool has_active_descendent(const VoxelElement el, const uint64_t layers, Predicate&& pred = nullptr) const {
			assert(el.is_valid());

			bool result = false;
			for (const auto child : el.children()) {
				if (!child.exists()) {continue;}
				if constexpr (!NULLPTR_T<Predicate>) {
					if (!pred(child)) {continue;}
				}

				if (is_active(child)) {return true;}
				if (layers==0) {continue;}
				result = result || has_active_descendent(child, layers-1, std::forward<Predicate>(pred));
			}
			return result;
		}


		//tree operations
		template<typename Predicate = std::nullptr_t>
		void refine_to_depth(const VoxelElement el, const uint64_t depth, Predicate&& pred = nullptr) {
			if (el.depth() < depth) {
				refine(el, std::forward<Predicate>(pred));
			}
			for (VoxelElement c : el.children()) {
				if (!c.exists()) {continue;}
				//it is possible that some children satisfy the predicate
				//even if the parent doesn't. this is especially likely if
				//the predicate is testing if the voxel is in some specified region (e.g., a true geometry)
				refine_to_depth(c, depth, std::forward<Predicate>(pred));
			}
		}

		//if any element is active and has any active children,
		//then deactivate that element. activate any in-active children
		//if the predicate on the inactive child returns true.
		//the default predicate always returns true
		template<typename Predicate = std::nullptr_t>
		void make_disjoint(Predicate&& pred = nullptr) {
			auto action = [&](VoxelElement el) {
				if (is_active(el)) {
					if (has_active_child(el)) {
						deactivate(el);
						for (const VoxelElement child : el.children()) {
							if (!child.exists()) {continue;}
							if constexpr (!NULLPTR_T<Predicate>) {
								if (pred(child)) {activate(child);}
							}
							else {activate(child);}
						}
					}
				}
			};

			//TODO: this can be made more efficient by not cascading
			for_each<VoxelElement>(action, OPENMP);
		}

		//geometry operations
		inline GeoPoint_t ref2geo(const VoxelVertex vtx) const {
			return low + diag*vtx.normalized_coordinate();
		}

		uint64_t write_unstructured_vtk(std::ostream& os) const {
			os << "# vtk DataFile Version 2.0\n"
			   << "Hierarchical mesh as unstructured mesh\n"
			   << "ASCII\n\n"
			   << "DATASET UNSTRUCTURED_GRID\n";

			//Loop through the active elements
			//write the element buffer and track the largest index
			//of the vertices to write
			//additionally, write the cell buffer
			std::ostringstream cell_buffer;
			const uint64_t n_elems = n_elements();
			cell_buffer << "CELLS " << n_elems << " " << 9*n_elems << "\n";

			uint64_t max_vtx_index=0;
			auto action_c = [&](VoxelElement el) {
				if (is_active(el)) {
					cell_buffer << "8 ";
					const auto verts = el.vertices();
					for (int v=0; v<8; ++v) {
						// const VoxelVertex vtx = el.vertex(v).reduced_key();
						const VoxelVertex vtx = verts[v];
						assert(vtx.is_valid());
						const uint64_t idx = vtx.linear_index();
						max_vtx_index = std::max(idx, max_vtx_index);
						cell_buffer << idx << " ";
					}
					cell_buffer << "\n";
				}
			};

			for_each<VoxelElement>(action_c, false);
			cell_buffer << "\n";

			const uint64_t n_vertices = max_vtx_index+1; //writing a few extra vertices is ok

			//write the vertex points up to the requested index
		   	std::ostringstream point_buffer;
			point_buffer << "POINTS " << n_vertices << " float\n";

			VoxelVertex vtx(0,0);
			for (uint64_t idx=0; idx<n_vertices; ++idx, ++vtx) {
				point_buffer << ref2geo(vtx) << "\n";
			}
			point_buffer << "\n";

			//Element type
			cell_buffer << "CELL_TYPES " << n_elems << "\n";
			for (size_t i=0; i<n_elems; ++i) {cell_buffer << "11 ";}
			cell_buffer << "\n\n";
			
			//write to file
			os << point_buffer.str();
			os << cell_buffer.str();

			//return the number of vertices for other methods to use
			return n_vertices;
		}

		template<typename CellDataLookup = std::nullptr_t>
		void append_unstructured_cell_data_vtk(
			std::ostream& os,
			const std::string& data_header, 
			CellDataLookup&& lookup
			) const {
			
			std::ostringstream buffer;
			buffer << data_header << "\n";

			auto pred   = [this](const VoxelElement el) {return this->is_active(el);};
			auto action = [&buffer, &lookup](const VoxelElement el) {buffer << lookup(el) << "\n";};
			for_each<VoxelElement>(action, false, pred);
			buffer << "\n";
			os << buffer.str();
		}

		template<typename CellDataLookup = std::nullptr_t>
		void append_unstructured_point_data_vtk(
			std::ostream& os,
			const std::string& data_header,
			const uint64_t n_vertices,
			CellDataLookup&& lookup
			) const {

			os << data_header << "\n";

			std::ostringstream point_buffer;
			VoxelVertex vtx(0,0);
			for (uint64_t idx=0; idx<n_vertices; ++idx, ++vtx) {
				point_buffer << lookup(vtx) << "\n";
			}
			point_buffer << "\n";

			os << point_buffer.str();
		}

		void save_unstructured_mesh(const std::string filename = "voxel_mesh_unstructured.vtk") const {
			std::ofstream file(filename);
			if (!file.is_open()) {
				throw std::runtime_error("VoxelMesh::save_unstructured_mesh - could not open file: " + filename);
			}

			write_unstructured_vtk(file);
			file << "CELL_DATA " << n_elements() << "\n";

			auto lookup_dijk   = [](VoxelElement el) {return Point<4,uint64_t>{el.depth(), el.i(), el.j(), el.k()};};
			auto lookup_linear = [](VoxelElement el) {return el.depth_linear_index();};
			auto lookup_color  = [](VoxelElement el) {return el.color();};

			append_unstructured_cell_data_vtk(file, "SCALARS color int 1\nLOOKUP_TABLE default", lookup_color);
			append_unstructured_cell_data_vtk(file, "SCALARS dijk int 4\nLOOKUP_TABLE default",  lookup_dijk);
			append_unstructured_cell_data_vtk(file, "SCALARS linear int 1\nLOOKUP_TABLE default",lookup_linear);
		}

	private:
		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t> && !OPENMP)
		void for_each_depth_omp_impl(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const {
			std::cerr << "VoxelMesh::for_each_depth_omp (const) called without OpenMP enabled\n";
			for_each_depth<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
		}

		#ifdef _OPENMP
		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t> && OPENMP)
		void for_each_depth_omp_impl(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const {
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
					if constexpr (std::is_invocable_v<Action, Key_t, int>) {
						action(key, tid);
					}
					else {
						action(key);
					}
				}
			}
		}
		#endif

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		void for_each_depth_impl(const uint64_t depth, Action&& action, Predicate&& pred = nullptr) const {
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

		template<typename Key_t, typename Action, typename Predicate = std::nullptr_t> requires (MeshFeatureType<Key_t,Mesh_t>)
		void for_each_impl(Action&& action, bool omp=false, Predicate&& pred = nullptr) const {
			if (omp) {
				for (uint64_t depth=0; depth<MAX_DEPTH+1; ++depth) {
					for_each_depth_omp<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
				}
			}
			else {
				for (uint64_t depth=0; depth<MAX_DEPTH+1; ++depth) {
					for_each_depth<Key_t>(depth, std::forward<Action>(action), std::forward<Predicate>(pred));
				}
			}
		}
	};
}