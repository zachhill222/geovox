#pragma once

#include "gutil.hpp"

#include "mesh/keys/voxel_key.hpp"
#include "mesh/vtk_file_io.hpp"
#include "mesh/voxel_mesh.hpp"
#include "util/concepts.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>
#include <array>
#include <span>
#include <bitset>

#include <iostream>
#include <sstream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV {
	

	//This class is for a structured voxel mesh at a single depth.
	//This structure allows us to very efficiently store elements, vertices, faces, etc.
	//The base mesh is 1x1x1 so that every vertex is (in reference coordinates) a dyadic rational number
	//Elements, vertices, and faces all have special index/key structs for their storage and logical relations.
	//All information for every element, vertex, and face is compressed into a 64-bit unsigned integer
	//With the element/vertex/face keys, we must have a maximum depth of 15. If more is needed (unlikely) we can stitch together
	//multiples of these meshes.

	template<uint64_t Depth>
	class StructuredVoxelMesh {
	public:
		//mesh features are never periodic
		using VoxelElement = VoxelElementKey<15,0>;
		using VoxelVertex  = VoxelVertexKey<15,0>;
		using VoxelFace    = VoxelFaceKey<15,0>;
		using Mesh_t       = VoxelMesh<15>; //this mesh type

		static constexpr uint64_t DEPTH = Depth;
		static constexpr uint64_t TOTAL_POSSIBLE_ELEMENTS = 
				VoxelElement::depth_linear_start(Depth+1) - VoxelElement::depth_linear_start(Depth);

		#ifdef _OPENMP
			static constexpr bool OPENMP = true;
		#else
			static constexpr bool OPENMP = false;
		#endif

	protected:
		std::bitset<TOTAL_POSSIBLE_ELEMENTS>* active_elem = new std::bitset<TOTAL_POSSIBLE_ELEMENTS>(1);

	public:
		using GeoPoint_t = gutil::Point<3,double>; //points in space
		
		const GeoPoint_t low;
		const GeoPoint_t high;
		const GeoPoint_t diag;

		StructuredVoxelMesh(const GeoPoint_t low_, const GeoPoint_t high_) :
			low{elmin(low_, high_)},
			high{elmax(low_, high_)},
			diag{high-low} {}

		virtual ~StructuredVoxelMesh() {delete active_elem;}

		//simple querries and operations
		inline void reset() {active_elem->reset();}
		inline size_t n_elements() const {return active_elem->count();}
		inline size_t count_elements() const {return active_elem->count();}

		inline void activate(const VoxelElement el) noexcept {
			assert(el.depth()==DEPTH);
			assert(el.is_valid());
			active_elem->set(el.linear_index());
		}

		inline void deactivate(const VoxelElement el) noexcept {
			assert(el.depth()==DEPTH);
			assert(el.is_valid());
			active_elem->reset(el.linear_index());
		}

		//test if a feature is active.
		//active elements are recorded int the active_elem bitset
		//vertices and faces are active if they belong to an active element
		//note there may be many vertices at the same geometric location but existing at different levels
		//you may need to look at parent/child vertices to get the expected result
		[[nodiscard]] inline bool is_active(const VoxelElement el) const noexcept {
			assert(el.depth()==DEPTH);
			assert(el.is_valid()); 
			return active_elem->test(el.linear_index());
		}
		
		template<typename Key_t> requires (MeshFaceType<Key_t,Mesh_t> || MeshVertexType<Key_t,Mesh_t>)
		bool is_active(const Key_t key) const {
			assert(key.depth()==DEPTH);
			assert(key.is_valid());
			for (const VoxelElement el : key.elements()) {
				if (el.exists() and is_active(el)) {return true;}
			}
			return false;
		}

		inline void set(const VoxelElement el, const bool flag = true) {
			assert(el.depth()==DEPTH);
			assert(el.is_valid());
			active_elem->set(el.linear_index(), flag);
		}

		template<typename Predicate = std::nullptr_t>
		[[nodiscard]] StructuredVoxelMesh<DEPTH+1> refine(Predicate&& pred = nullptr) requires (DEPTH<MAX_DEPTH) {
			StructuredVoxelMesh<DEPTH+1> mesh(low,high);

			//set the children of active elements to active and the
			//children of inactive elements to inactive
			if constexpr (IS_NULLPTR_T<Predicate>) {
				#pragma omp parallel for
				for (size_t i=0; i<TOTAL_POSSIBLE_ELEMENTS; ++i) {
					const VoxelElement el(DEPTH, i);
					const bool flag = is_active(el);
					for (VoxelElement child : el.children()) {
						mesh.set(child, flag);
					}
				}
			}
			else {
				#pragma omp parallel for
				for (size_t i=0; i<TOTAL_POSSIBLE_ELEMENTS; ++i) {
					const VoxelElement el(DEPTH, i);
					for (VoxelElement child : el.children()) {
						mesh.set(child, pred(child));
					}
				}
			}
		}

		template<typename Predicate = std::nullptr_t>
		[[nodiscard]] StructuredVoxelMesh<DEPTH+1> coarsen(Predicate&& pred = nullptr) requires (DEPTH>0) {
			StructuredVoxelMesh<DEPTH-1> mesh(low,high);

			//set the parent of active elements to active if it has more active than inactive children
			if constexpr (IS_NULLPTR_T<Predicate>) {
				#pragma omp parallel for
				for (size_t i=0; i<mesh.TOTAL_POSSIBLE_ELEMENTS; ++i) {
					const VoxelElement el(DEPTH-1, i);
					int count = 0;
					for (VoxelElement child : el.children()) {
						count += this->is_active(child) ? 1 : 0;
					}
					mesh.set(el, count>=4);
				}
			}
			else {
				#pragma omp parallel for
				for (size_t i=0; i<mesh.TOTAL_POSSIBLE_ELEMENTS; ++i) {
					const VoxelElement el(DEPTH-1, i);
					mesh.set(el, pred(el));
				}
			}
		}

		
		//geometry operations
		template<typename V> requires (VoxelEquivFeature<V,VoxelVertex>)
		inline GeoPoint_t ref2geo(const V vtx) const {
			return low + diag*vtx.normalized_coordinate();
		}

		
		


	};
}