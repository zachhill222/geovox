#pragma once

#include <cassert>
#include <iostream>

#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>

#include "geometry/point.hpp"
#include "geometry/box.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////////
/// This file provides helper classes for the BasicParallelOctree class. Specifically,
/// it provides classes for the tree nodes and a thread-safe queue that multiple
/// threads can write to while the worker thread reads from it.
////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace gutil {
	/////////////////////////////////////////////////
	/// Octree node structure
	/////////////////////////////////////////////////
	template<int DIM=3, int N_DATA=16, typename T=float>
	struct OctreeParallelNode {
		static constexpr int N_CHILDREN = 1 << DIM;  // 2^DIM
		using Point_t = Point<DIM,T>;
		using Box_t   = Box<DIM,T>;
		using Node_t  = OctreeParallelNode<DIM,N_DATA,T>;

		// Tree structure
		Node_t* parent = nullptr;
		Node_t* children[N_CHILDREN] {nullptr};
		int sibling_number = -1;  // parent->children[this->sibling_number] == this
		int depth = 0;
		const Box_t bbox;
		
		// Data indices stored in this node
		size_t* data_idx = nullptr;
		int cursor = 0;
		
		// Synchronization
		std::atomic<bool> is_leaf{true};
		mutable std::shared_mutex mutex;

		// Create child node
		OctreeParallelNode(Node_t* parent, int sibling_number):
			parent(parent),
			sibling_number(sibling_number), 
			depth(parent->depth + 1), 
			bbox(parent->bbox.center(), parent->bbox.voxelvertex(sibling_number))
			{}

		// Create root node with specified bounding box
		explicit OctreeParallelNode(const Box_t& bbox, int depth=0) :
			depth(depth),
			bbox(bbox)
			{}

		// Destructor
		~OctreeParallelNode() {
			for (int c=0; c<N_CHILDREN; c++) {
				delete children[c];
			}
			delete[] data_idx;
		}

		// Non-copyable, non-movable
		OctreeParallelNode(const OctreeParallelNode&) = delete;
		OctreeParallelNode& operator=(const OctreeParallelNode&) = delete;
		OctreeParallelNode(OctreeParallelNode&&) = delete;
		OctreeParallelNode& operator=(OctreeParallelNode&&) = delete;
	};

	/////////////////////////////////////////////////
	/// Node data management helpers
	/////////////////////////////////////////////////
	
	template<int DIM, int N_DATA, typename T>
	void resetDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node) {
		assert(node);
		assert(node->cursor <= N_DATA);

		delete[] node->data_idx;
		node->data_idx = new size_t[N_DATA];
		node->cursor = 0;
	}

	template<int DIM, int N_DATA, typename T>
	void clearDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node) {
		assert(node);
		assert(node->cursor <= N_DATA);

		delete[] node->data_idx;
		node->data_idx = nullptr;
		node->cursor = 0;
	}

	/// Append data index to node
	/// Returns: 1 if added, 0 if already present, -1 if no room
	template<int DIM, int N_DATA, typename T>
	int appendDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node, const size_t idx) {
		assert(node);
		assert(isLeaf(node));
		assert(node->cursor <= N_DATA);

		if (node->data_idx == nullptr) {
			resetDataIdx(node);
		}

		// Check if already present
		for (int i = 0; i < node->cursor; i++) {
			if (node->data_idx[i] == idx) {
				return 0;
			}
		}

		// Check capacity
		if (node->cursor >= N_DATA) {
			return -1;
		}

		// Add index
		node->data_idx[node->cursor] = idx;
		node->cursor++;
		return 1;
	}

	/// Remove data index from node
	template<int DIM, int N_DATA, typename T>
	void removeDataIdx(OctreeParallelNode<DIM,N_DATA,T>* node, const size_t idx) {
		if (node == nullptr) {return;}
		if (node->data_idx == nullptr) {return;}
		
		for (int i = 0; i < node->cursor; i++) {
			if (node->data_idx[i] == idx) {
				// Swap with last element and decrement cursor
				node->data_idx[i] = node->data_idx[node->cursor - 1];
				node->cursor--;
				return;
			}
		}
	}

	/// Check if a node contains a specific index
	template<int DIM, int N_DATA, typename T>
	bool containsIndex(const OctreeParallelNode<DIM,N_DATA,T>* node, const size_t idx) {
		if (node==nullptr) {return false;}
		if (node->data_idx==nullptr) {return false;}
		for (int i = 0; i < node->cursor; i++) {
			if (node->data_idx[i] == idx) {return true;}
		}
		return false;
	}

	/// Check if node is a leaf
	template<int DIM, int N_DATA, typename T>
	inline bool isLeaf(const OctreeParallelNode<DIM,N_DATA,T>* node) {
		assert(node);
		return node->is_leaf.load(std::memory_order_acquire);
	}

	/// Print for debug
	template<int DIM, int N_DATA, typename T>
	std::ostream& operator<<(std::ostream& os, const OctreeParallelNode<DIM,N_DATA,T>* node) {
		if (node==nullptr) {os << "NULL\n"; return os;}

		os << "thread id: " << std::this_thread::get_id() << "\n"
		   << "bbox: " << node->bbox << "\n"
		   << "sibling_number: " << node->sibling_number << "\n"
		   << "depth: " << node->depth << "\n"
		   << "cursor: " << node->cursor << "/" << N_DATA << "\n"
		   << "is_leaf: " << node->is_leaf.load(std::memory_order_acquire) << "\n";

		assert(node->cursor <= N_DATA);
		return os;
	}


}