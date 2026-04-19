#pragma once

#include <cassert>
#include <vector>

#include <cstring>

#include <mutex>
#include <shared_mutex>
#include <thread>
#include <chrono>
#include <optional>

#include "geometry/point.hpp"
#include "geometry/box.hpp"
#include "octree/octree_util.hpp"
#include "octree/thread_queue.hpp"

#ifndef GUTIL_MAX_OCTREE_DEPTH
	#define GUTIL_MAX_OCTREE_DEPTH 16
#endif



////////////////////////////////////////////////////////////////////////////////////////////////////////
/// BasicParallelOctree - Thread-safe spatial data structure
///
/// This octree supports concurrent insertions from multiple OpenMP threads via push_back_async().
/// Space must be pre-allocated via resize() before async insertions. Use shrink_to_fit() to reclaim
/// unused space afterwards.
///
/// Key features:
/// - Immediate data storage: push_back_async() stores data immediately and returns its index
/// - Deferred tree updates: A dedicated worker thread updates the octree structure asynchronously
/// - flush() waits until all pending tree updates complete
///
/// Thread safety:
/// - Calling push_back() should only be done from a single thread
/// - Multiple threads can call push_back_async() concurrently
/// - If the same new data is inserted by multiple threads, each gets a different index but only
///   one is correct. This will result in a runtime error.
///   If data already exists in the tree, the correct index is safely returned.
////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace gutil {
	/////////////////////////////////////////////////
	/// BasicParallelOctree - Main container class
	///
	/// @tparam Data_t           Type of data to store
	/// @tparam SINGLE_DATA      If true, data goes in first valid leaf only.
	///                            If false, data goes in all overlapping leaves.
	/// @tparam DIM              Spatial dimension (typically 3)
	/// @tparam N_DATA           Max data indices per leaf node
	/// @tparam T                Type that emulates the real line (e.g., float) for bounding boxes
	/////////////////////////////////////////////////
	template<
		typename DATA_T,
		bool SINGLE_DATA,
		int DIM=3,
		int N_DATA=16,
		typename T=float>
	class BasicParallelOctree {
		static_assert(DIM==3 or DIM==2, "The octree must be in 2 or 3 dimensions");
		static_assert(N_DATA > 0, "N_DATA must be positive");

	public:
		// Type aliases
		using Node_t  = OctreeParallelNode<DIM,N_DATA,T>;
		using Point_t = Point<DIM,T>;
		using Box_t   = Box<DIM,T>;
		using Data_t  = DATA_T;

		struct Stats {
			size_t n_used_indices=0;
			size_t n_indices_capacity=0;
			size_t n_leafs=0;
			size_t n_nodes=0;
			double memory_reserved_bytes=0;
			double memory_used_bytes=0;
			int max_depth=0;
		};

		static constexpr int N_CHILDREN = Node_t::N_CHILDREN;
	
	protected:
		std::vector<Data_t> _data; //sometimes derived classes should be able to access
		// Tree structure

		Node_t* _root = nullptr;
		mutable std::shared_mutex _tree_mutex; //lock when doing large tree changes
	
	private:
		// Thread management
		mutable std::mutex _thread_manage_mutex; //lock when determining which threads should change tasks
		int n_flushing_threads{0};
		static constexpr int n_max_flushing_threads = 400; //self balancing seems best

		// Data storage
		std::atomic<size_t> _next_data_idx{0};

		struct QueueData_t {
			Node_t* node;
			size_t  idx;
		};

		MultipleInMultipleOut<QueueData_t> _queue;
		bool _bbox_has_changed = false;
		
	public:
		////////////////////////////////////////////////////////////
		// Construction and destruction
		////////////////////////////////////////////////////////////
		explicit constexpr BasicParallelOctree(const Box_t &bbox) noexcept;
		constexpr BasicParallelOctree() noexcept : BasicParallelOctree(Box_t{}) {}
		virtual ~BasicParallelOctree() noexcept {delete _root;};

		// Non-copyable, non-movable
		BasicParallelOctree(const BasicParallelOctree&) = delete;
		BasicParallelOctree& operator=(const BasicParallelOctree&) = delete;
		BasicParallelOctree(BasicParallelOctree&&) = delete;
		BasicParallelOctree& operator=(BasicParallelOctree&&) = delete;

		////////////////////////////////////////////////////////////
		// Container interface
		////////////////////////////////////////////////////////////
		void reserve(const size_t length) {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			_data.reserve(length);
		}

		void shrink_to_fit() {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			_data.resize(_next_data_idx.load(std::memory_order_acquire));
			_data.shrink_to_fit();
		}

		bool empty() const {
			std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
			return _data.empty();
		}

		size_t size() const {
			return _next_data_idx.load(std::memory_order_acquire);
		}

		size_t capacity() const {
			std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
			return _data.capacity();
		}

		void resize(const size_t length) {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			assert(length > _next_data_idx.load(std::memory_order_acquire));

			// Resize the storage container
			_data.resize(length);
		}

		void clear() {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);

			_data.clear();
			_next_data_idx.store(0);
			
			// clear the tree structure
			Node_t* new_root = new Node_t(_root->bbox);
			resetDataIdx(new_root);
			delete _root;
			_root = new_root;
		}

		Stats get_tree_stats() const {
			Stats result{};
			_recursive_node_properties(_root, result);
			result.max_depth -= _root->depth;
			return result;
		}

		////////////////////////////////////////////////////////////
		// Element access
		////////////////////////////////////////////////////////////
		inline constexpr const Data_t& operator[](const size_t idx) const noexcept {
			assert(idx < size());
			std::shared_lock<std::shared_mutex> lock(_tree_mutex);
			return _data[idx];
		}

		inline constexpr Data_t& operator[](const size_t idx) noexcept {
			assert(idx < size());
			std::shared_lock<std::shared_mutex> lock(_tree_mutex);
			return _data[idx];
		}

		constexpr const Data_t& at(const size_t idx) const {
			if(idx >= size()) {throw std::runtime_error("BasicParallelOctree: index out of range");}
			std::shared_lock<std::shared_mutex> lock(_tree_mutex);
			return _data[idx];
		}

		constexpr Data_t& at(const size_t idx) {
			if(idx >= size()) {throw std::runtime_error("BasicParallelOctree: index out of range");}
			std::shared_lock<std::shared_mutex> lock(_tree_mutex);
			return _data[idx];
		}

		////////////////////////////////////////////////////////////
		// Queries
		////////////////////////////////////////////////////////////
		/// Find all data indices that MAY be associated with the specified box
		/// This gets all data contained in nodes where the node bounding box intersects the
		/// specified bounding box
		std::vector<size_t> get_data_in_box(const Box_t& bbox) const;

		/// Find all data indices that MAY be associated with the specified point
		/// This returns all data indices contained in leaf nodes that intersect a "sphere" of radius R
		/// (sphere in the infinity norm, so it's actually a box) centered at the specified point.
		/// R is chosen such that the sphere encloses the closest leaf to the specified point that is non-empty.
		std::vector<size_t> get_data_near_point(const Point_t& point) const;

		/// Find existing data in tree
		/// Returns index if found, (size_t)-1 if not found
		/// Only call while the the tree is stable (i.e., push_back_async() or flush() are not running)
		size_t find(const Data_t& val) const {
			std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
			const size_t result = _recursive_find_index<false>(_root, val); //only called while the tree is stable
			return result;
		}

		/// Find closest data to the specified point
		size_t find_closest(const Point_t& point) const {
			std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
			if (empty()) {return (size_t) -1;}

			T bestdist = dist2data(point, _data[0]);
			size_t idx = 0;

			_recursive_find_closest_data(_root, point, idx, bestdist);
			return idx;
		}

		////////////////////////////////////////////////////////////
		// Iterators
		////////////////////////////////////////////////////////////
		auto begin()        { return _data.begin(); }
		auto begin()  const { return _data.cbegin();}
		auto cbegin() const { return _data.cbegin();}
		auto end()          { return _data.begin() += size();}
		auto end()    const { return _data.begin() += size();}
		auto cend()   const { return _data.begin() += size();}

		const DATA_T& back() const {return _data[size()-1];}

		////////////////////////////////////////////////////////////
		// Interact with the bounding box
		////////////////////////////////////////////////////////////
		inline const Box_t& bbox() const {return _root->bbox;}
		inline Box_t bbox_tight() const;
		inline bool bbox_has_changed() {return _bbox_has_changed;}

		void resize_to_fit_data(const Data_t& val) {
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			while (!isValid(_root->bbox, val)) {
				_recursive_expand_bbox(T{2} * _root->bbox);
			}

			_bbox_has_changed=true;
		}

		void set_bbox(const Box_t& new_bbox) {
			assert(new_bbox.contains(_root->bbox));
			std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
			_recursive_expand_bbox(new_bbox);

			_bbox_has_changed=true;
		}
		
		////////////////////////////////////////////////////////////
		// Insertion operations
		////////////////////////////////////////////////////////////
		
		/// Single thread copy and move
		size_t push_back(const Data_t& val) {
			Data_t copy_val(val);
			return push_back(std::move(copy_val));
		}

		/// Add data to end of _data. Can be called asynchronously, but does not use the queue.
		size_t push_back(Data_t&& val);

		/// Asynchronous push back. To use this method, we must have:
		/// 	1) every data value inserted in a parallel batch is unique
		///	 	2) resize() must be called ahead of time so that no _data allocations
		///              are required
		///     3) (optional) set the bounding box (via set_bbox()) to encapulate all
		/// 			 new data values
		size_t push_back_async(Data_t&& val);

		/// Wait for all pending async insertions to complete
		void flush() {
			while (!_queue.empty()) {
				if (!_switch_to_flush()) {
					std::this_thread::yield();
				}
			}
		}

		/// Change data at the specified index
		void replace(Data_t&& new_val, const size_t idx) {
			assert(idx<size());
			if (!isValid(_root->bbox, new_val)) {
				resize_to_fit_data(new_val);
			}

			std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);

			_recursive_remove_index(_root, idx);
			_data[idx] = std::move(new_val);
			Node_t* start_node = _recursive_find_best_node(_root, _data[idx]);
			[[maybe_unused]] int flag = _recursive_insert_data<true>(start_node, _data[idx], idx);
			assert(flag==1);
		}

		void replace(const Data_t& new_val, const size_t idx) {
			Data_t new_val_copy(new_val);
			replace(std::move(new_val_copy), idx);
		}

		/// ensure the entire tree is valid. useful after changing the bounding box.
		void rebuild_tree()
		{
			std::lock_guard<std::shared_mutex> lock(_tree_mutex);
			//TODO: this should be in parallel
			for (size_t idx=0; idx<size(); idx++) {
				int flag = _recursive_insert_data<true>(_root, _data[idx], idx);
				if (flag == -1) {
					throw std::runtime_error("BasicParallelOctree::rebuild_tree() - data at index " + std::to_string(idx) + " is invalid");
				}

				if constexpr (SINGLE_DATA) {
					if (flag==1) {
						//the data should have already existed
						_recursive_remove_index<true>(_root, idx);
						_recursive_insert_data<true>(_root, _data[idx], idx);
					}
				}
			}

			_bbox_has_changed = false;
		}

	private:
		/// Determine if data belongs in the given bounding box (must be overridden)
		virtual constexpr bool isValid(const Box_t &bbox, const Data_t &val) const = 0;

		/// Compute distance (or a proxy such as distance squared) from a point to a Data_t
		virtual constexpr T dist2data(const Point_t &point, const Data_t &val) const = 0;

		////////////////////////////////////////////////////////////
		// Recursive helper functions
		////////////////////////////////////////////////////////////
		/// Find best node to start insertion/search
		Node_t* _recursive_find_best_node(const Node_t* node, const Data_t& val) const;

		/// Insert data into tree
		template<bool WAIT>
		int _recursive_insert_data(Node_t* node, const Data_t& val, const size_t idx);

		/// Divide a leaf node into children
		void _divide(Node_t* node);

		/// Find index of data in tree
		template<bool LOCKED>
		size_t _recursive_find_index(const Node_t* node, const Data_t& val) const;

		/// Find all data indices in nodes that intersect the specified box
		void _recursive_data_in_box(const Node_t* node, const Box_t& bbox,
			std::vector<size_t>& data_indices) const;

		const Node_t* _recursive_closest_data_leaf(const Node_t* node, const Point_t& point, T& dist2) const;

		void _recursive_find_closest_data(const Node_t* node, const Point_t& point, size_t& idx, T& dist) const;


		/// Expand root bounding box to contain new region
		void _recursive_expand_bbox(const Box_t& new_bbox);

		/// Switch a thread from inserting to flushing if possible
		/// Return false if the thread didn't switch.
		bool _switch_to_flush();

		/// Remove index idx from all leaf nodes
		/// _data[idx] must be valid (and the data that will be removed from the tree)
		template<bool SEARCH_ALL=false>
		void _recursive_remove_index(Node_t* node, const size_t idx);

		/////////////////////////////////////////////////
		/// Convenience and debug methods
		/////////////////////////////////////////////////
		void _recursive_node_properties(const Node_t* node, Stats& stats) const;

		void _recursive_compute_data_leaf_bbox(const Node_t* node, Box_t& bbox) const;
	};


	//////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////    METHOD IMPLEMNTATION    /////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	constexpr BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::BasicParallelOctree(
		const Box<DIM,T>& bbox) noexcept
	{
		//increase bounding box to closest power of 2 to better avoid floating point arithmetic errors
		Point_t low(bbox.low());
		Point_t high(bbox.high());

		for (int i = 0; i < DIM; i++) {
			int n  = 1+static_cast<int>(std::log2(std::fabs(low[i])));
			if (low[i]<0.0) {low[i] = -static_cast<T>(std::pow(2.0, n));}
			else {low[i] = static_cast<T>(std::exp2(n));}
			
			n  = 1+static_cast<int>(std::log2(std::fabs(high[i])));
			if (high[i]<0.0) {high[i] = -static_cast<T>(std::pow(2.0, n));}
			else {high[i] = static_cast<T>(std::exp2(n));}
		}
		
		//set _root with rounded bounding box
		_root = new Node_t(Box_t{low, high});
		{
			std::lock_guard<std::shared_mutex> lock(_root->mutex);
			resetDataIdx(_root);
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	std::vector<size_t> BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::get_data_in_box(
		const Box_t& bbox) const
	{
		std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);

		std::vector<size_t> data_indices;
		_recursive_data_in_box(_root, bbox, data_indices);

		//ensure that data_indices is sorted and has no duplicates
		std::sort(data_indices.begin(), data_indices.end());
		auto last = std::unique(data_indices.begin(), data_indices.end());
		data_indices.erase(last, data_indices.end());

		//return data indices found (sorted, no duplicates, all valid)
		return data_indices;
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	std::vector<size_t> BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::get_data_near_point(
		const Point_t& point) const
	{
		//set maximal distance
		T dist2{0};
		for (int i=0; i<N_CHILDREN; i++) {
			T dist2vtx = squaredNorm(_root->bbox.voxelvertex(i) - point);
			if (dist2vtx > dist2) {dist2 = dist2vtx;}
		}

		//find closets leaf with data
		const Node_t* leaf = _recursive_closest_data_leaf(_root, point, dist2);

		//construct box that contains the found leaf
		Point_t H = leaf->bbox.voxelvertex(0) - point;
		for (int i=1; i<N_CHILDREN; i++) {
			H = elmax(H, leaf->bbox.voxelvertex(i) - point);
		}

		return get_data_in_box(Box_t{point-H, point+H});
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	size_t BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::push_back(DATA_T&& val)
	{
		if (!isValid(_root->bbox, val))
		{
			//enforces unique tree lock
			resize_to_fit_data(val);
		}

		std::lock_guard<std::shared_mutex> tree_lock(_tree_mutex);
		Node_t* start_node = _recursive_find_best_node(_root, val);

		//try to find the existing data
		size_t idx = (size_t) -1;
		idx = _recursive_find_index<false>(start_node, val); //single threaded, no need to lock
		if (idx!=(size_t) -1) {return idx;}
		
		//data must be inserted
		idx = _next_data_idx.fetch_add(1, std::memory_order_acq_rel);
		assert(idx == _data.size());
		_data.push_back(std::move(val));

		//insert the data into the tree
		int flag = _recursive_insert_data<true>(start_node, val, idx);

		if (flag==-1)
		{
			throw std::runtime_error("BasicParallelOctree: couldn't insert data at index " + std::to_string(idx));
		}


		return idx;
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	size_t BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::push_back_async(DATA_T&& val)
	{
		if (!isValid(_root->bbox, val))
		{
			//enforces unique tree lock
			resize_to_fit_data(val);
		}

		std::shared_lock<std::shared_mutex> tree_lock(_tree_mutex);
		Node_t* start_node = _recursive_find_best_node(_root, val);

		//try to find the existing data
		size_t idx = (size_t) -1;
		idx = _recursive_find_index<true>(start_node, val); //must aquire read lock
		if (idx!=(size_t) -1) {return idx;}

		//data must be inserted
		idx = _next_data_idx.fetch_add(1, std::memory_order_acq_rel);
		assert(idx < _data.size());
		assert(_data[idx] == Data_t{});
		_data[idx] = std::move(val);

		//insert the data into the tree
		int flag = 1;
		{
			flag = _recursive_insert_data<false>(start_node, val, idx);
			if (flag == -1)
			{
				[[maybe_unused]] bool success = _queue.push({start_node, idx});
				assert(success);
				_switch_to_flush(); //change to an inserter thread until the queue is empty
			}
		}
		return idx;
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////    CORE METHOD IMPLEMNTATION    ///////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	OctreeParallelNode<DIM,N_DATA,T>* 
		BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_find_best_node(
			const OctreeParallelNode<DIM,N_DATA,T>* node,
			const Data_t &val) const
	{
		if (node==nullptr) {return nullptr;}
		if (!isValid(node->bbox, val)) {return nullptr;}

		bool is_leaf_node = false;
		{
			std::shared_lock<std::shared_mutex> lock(node->mutex);
			is_leaf_node = isLeaf(node);
		}

		if (!is_leaf_node)
		{
			// Traverse to child containing data
			if constexpr (SINGLE_DATA)
			{
				for (int c=0; c<N_CHILDREN; c++) {
					assert(node->children[c]);
					if (isValid(node->children[c]->bbox, val)) {
						return _recursive_find_best_node(node->children[c], val);
					}
				}
			}
			else
			{
				int n_valid_children = 0;
				int last_valid_child = -1;
				for (int c=0; c<N_CHILDREN; c++) {
					assert(node->children[c]);
					if (isValid(node->children[c]->bbox, val)) {
						n_valid_children++;
						last_valid_child = c;
					}
				}

				if (n_valid_children>1) {
					//we are at a branch node
					return const_cast<Node_t*>(node);
				}
				else if (n_valid_children == 1) {
					//only one valid child, find a better node
					return _recursive_find_best_node(node->children[last_valid_child], val);
				}
				else {
					throw std::runtime_error("BasicParallelOctree::_recursive_find_best_node - couldn't find valid child in a valid node");
				}
			}
		}

		// Couldn't find a valid leaf
		return const_cast<Node_t*>(node);
	}


	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	template<bool LOCKED>
	size_t BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::
		_recursive_find_index(
			const OctreeParallelNode<DIM,N_DATA,T>* node,
			const DATA_T &val) const
	{
		if (node==nullptr) {return (size_t) -1;}
		if (!isValid(node->bbox, val)) {return (size_t) -1;}

		
		//make a shared lock all the way down
		std::optional<std::shared_lock<std::shared_mutex>> lock;
		if constexpr (LOCKED) {lock.emplace(node->mutex);}

		if (!isLeaf(node))
		{
			assert(node->data_idx==nullptr);
			assert(node->cursor==0);
			for (int c=0; c<N_CHILDREN; c++)
			{
				assert(node->children[c]);
				if (isValid(node->children[c]->bbox, val))
				{
					const size_t result = _recursive_find_index<LOCKED>(node->children[c], val);
					if (result<size()) {return result;}
				}
			}
		}
		else
		{
			if (node->data_idx==nullptr) {return (size_t) -1;}

			for (int i=0; i<node->cursor; i++)
			{
				const size_t d_idx = node->data_idx[i];
				if (_data[d_idx]==val) {return d_idx;}
			}
		}

		return (size_t) -1;
	}


	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	template<bool WAIT>
	int BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_insert_data(
			OctreeParallelNode<DIM,N_DATA,T>* node,
			const Data_t &val,
			const size_t idx)
	{
		//return 1 on success and -1 on fail, return 0 on data already exists
		if (node == nullptr) {return -1;}
		if (!isValid(node->bbox, val)) {return -1;}

		//make shared lock on the way down
		std::shared_lock<std::shared_mutex> read_lock(node->mutex);

		
		if (!isLeaf(node))
		{
			assert(node->data_idx==nullptr);
			assert(node->cursor==0);

			int flag=-1; //failure flag
			for (int c=0; c<N_CHILDREN; c++)
			{
				assert(node->children[c]);
				if (isValid(node->children[c]->bbox, val))
				{
					flag = std::max(flag, _recursive_insert_data<WAIT>(node->children[c], val, idx));
					if constexpr (SINGLE_DATA) {break;}
				}
			}
			return flag;
		}
		else
		{
			//the node is PROBABLY a leaf node
			read_lock.unlock();
			std::unique_lock<std::shared_mutex> write_lock(node->mutex, std::try_to_lock);
			if (!write_lock.owns_lock())
			{
				if constexpr (WAIT) {write_lock.lock();}
				else {return -1;}
			}

			//some other thread divided this node since we checked last
			if (!isLeaf(node)) {
				write_lock.unlock();
				return _recursive_insert_data<WAIT>(node, val, idx);
			}

			//we have the unique lock and must append the data
			int flag = appendDataIdx(node, idx);
			if (flag>=0) {return flag;}

			//the data insertion was unsuccessful
			assert(node->cursor==N_DATA);
			_divide(node);

			//insert into children while we have the unique lock held
			flag=-1; //reset failure flag
			for (int c=0; c<N_CHILDREN; c++)
			{
				assert(node->children[c]);
				if (isValid(node->children[c]->bbox, val))
				{
					flag = std::max(flag, _recursive_insert_data<WAIT>(node->children[c], val, idx));
					if constexpr (SINGLE_DATA) {break;}
				}
			}
			return flag;
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_divide(
			OctreeParallelNode<DIM,N_DATA,T>* node)
	{
		//the unique lock must be engaged before calling
		if (node==nullptr) {return;}
		if (!isLeaf(node)) {return;}

		if (node->depth - _root->depth >= GUTIL_MAX_OCTREE_DEPTH) {
			throw std::runtime_error("BasicParallelOctree::_divide() - maximum depth (" + std::to_string(GUTIL_MAX_OCTREE_DEPTH) + ") exceeded");
		}

		node->is_leaf.store(false, std::memory_order_release);

		//create child nodes
		for (int c=0; c<N_CHILDREN; c++) {
			assert(node->children[c] == nullptr);
			node->children[c] = new Node_t(node, c);
		}

		//copy data down
		for (int d=0; d<node->cursor; d++)
		{
			const size_t d_idx = node->data_idx[d];
			const DATA_T& val  = _data[d_idx];
			for (int c=0; c<N_CHILDREN; c++)
			{
				if (isValid(node->children[c]->bbox, val))
				{
					[[maybe_unused]] int flag = appendDataIdx(node->children[c], d_idx);
					assert(flag==1);
					if constexpr (SINGLE_DATA) {break;}
				}
			}
		}

		// Clear parent node
		clearDataIdx(node);
		assert(node->data_idx==nullptr);
		assert(node->cursor==0);
	}


	


	//////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////    OTHER RECURSIVE METHOD IMPLEMNTATION    /////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////
	/// Expand root bounding box to contain new region
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_expand_bbox(const Box<DIM,T>& new_bbox) {
		if (_root->bbox.contains(new_bbox)) {
			return;
		}

		// Double the bounding box and find best placement
		Box_t expanded_root_bbox = T{2} * _root->bbox;
		int max_vertices = -1;
		int best_sibling_number = -1;

		for (int c = 0; c < N_CHILDREN; c++) {
			Point_t offset = _root->bbox.voxelvertex(c) - expanded_root_bbox.voxelvertex(c);
			Box_t test_box = expanded_root_bbox + offset;
			
			int n_verts = 0;
			for (int i = 0; i < N_CHILDREN; i++) {
				if (test_box.contains(new_bbox.voxelvertex(i))) {
					n_verts++;
				}
			}

			if (n_verts > max_vertices) {
				best_sibling_number = c;
				max_vertices = n_verts;
			}
		}

		// Create new root
		Point_t offset = _root->bbox.voxelvertex(best_sibling_number) 
		               - expanded_root_bbox.voxelvertex(best_sibling_number);
		Box_t new_root_bbox = expanded_root_bbox + offset;

		Node_t* old_root = _root;
		_root = new Node_t(new_root_bbox, old_root->depth - 1);
		{
			std::lock_guard<std::shared_mutex> lock(_root->mutex);
			_divide(_root);
		}
		
		
		delete _root->children[best_sibling_number];
		_root->children[best_sibling_number] = old_root;
		old_root->parent = _root;

		assert(_root->bbox.voxelvertex(best_sibling_number) == 
		       old_root->bbox.voxelvertex(best_sibling_number));
		
		// Recursively expand if needed
		if (max_vertices < N_CHILDREN) {
			_recursive_expand_bbox(new_bbox);
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_data_in_box(
		const OctreeParallelNode<DIM,N_DATA,T>* node,
		const Box<DIM,T>& bbox,
		std::vector<size_t>& data_indices) const
	{
		if (node==nullptr) {return;}
		if(!bbox.intersects(node->bbox)) {return;}
		
		if (isLeaf(node))
		{
			std::shared_lock<std::shared_mutex> lock(node->mutex);
			for (int d=0; d<node->cursor; d++) {
				const size_t d_idx = node->data_idx[d]; 
				if (isValid(bbox, _data[d_idx])) {data_indices.push_back(d_idx);}
			}
		}
		else
		{
			//recurse into children
			for (int c = 0; c < N_CHILDREN; c++) {
				_recursive_data_in_box(node->children[c], bbox, data_indices);
			}
		}

		//note that data_indices may contain duplicate entries at this point,
		//but all will be valid data
	}


	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_find_closest_data(
		const OctreeParallelNode<DIM,N_DATA,T>* node,
		const Point<DIM,T>& point,
		size_t& bestidx,
		T& bestdist) const
	{
		assert(node);

		//engage shared lock on the way down
		std::shared_lock<std::shared_mutex> read_lock(node->mutex);

		if (!isLeaf(node))
		{
			//sort children to recurse into closest first
			std::array<std::pair<T,int>, this->N_CHILDREN> children_dist_pair;
			for (int c=0; c<this->N_CHILDREN; c++) {
				children_dist_pair[c] = {distance_squared(node->children[c]->bbox, point), c};
			}
			std::sort(children_dist_pair.begin(), children_dist_pair.end()); //sorts lexigraphically, distance is first

			//find closest child
			for (const auto& [dist2child, c] : children_dist_pair) {
				if (dist2child < bestdist) {
					_recursive_find_closest_data(node->children[c], point, bestidx, bestdist);
				}
			}
		}
		else
		{
			//check if we are the closest leaf with data
			for (int i=0; i<node->cursor; i++) {
				const Data_t& VAL = this->_data[node->data_idx[i]];
				T temp_dist = dist2data(point, VAL);
				if (temp_dist < bestdist) {
					bestdist = temp_dist;
					bestidx  = node->data_idx[i];
				}
			}
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	const OctreeParallelNode<DIM,N_DATA,T>* BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_closest_data_leaf(
		const OctreeParallelNode<DIM,N_DATA,T>* node,
		const Point<DIM,T>& point,
		T& dist2) const
	{
		assert(node);

		//engage shared lock on the way down
		std::shared_lock<std::shared_mutex> read_lock(node->mutex);

		if (!isLeaf(node))
		{
			//find closest child
			int c_idx=0;
			T dist2child = distance_squared(node->children[0]->bbox, point);
			for (int c=1; c<N_CHILDREN; c++) {
				T temp_dist2child = distance_squared(node->children[c]->bbox, point);
				if (temp_dist2child < dist2child) {
					dist2child = temp_dist2child;
					c_idx = c;
				}
			}

			//if the closest child is further than dist2, don't recurse
			if (dist2 < dist2child) {return nullptr;}

			//recurse into closest child
			return _recursive_closest_data_leaf(node->children[c_idx], point, dist2);
		}
		else
		{
			//exit if we have no data
			if (node->cursor==0) {return nullptr;}
			assert(node->data_idx);

			//check if we are the closest leaf with data
			T dist2leaf = distance_squared(node->bbox, point);
			if (dist2leaf < dist2) {
				dist2 = dist2leaf;
				return node;
			}

			//we were not the closest leaf
			return nullptr;
		}
	}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	template<bool SEARCH_ALL>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_remove_index(
		OctreeParallelNode<DIM,N_DATA,T>* node,
		const size_t idx)
	{

		assert(node);
		assert(isValid(node->bbox, _data[idx]));

		//engage shared lock on the way down
		std::shared_lock<std::shared_mutex> read_lock(node->mutex);

		if (!isLeaf(node))
		{
			assert(node->data_idx==nullptr);
			assert(node->cursor==0);

			for (int c=0; c<N_CHILDREN; c++)
			{
				assert(node->children[c]);
				if constexpr (SEARCH_ALL) {
					_recursive_remove_index<SEARCH_ALL>(node->children[c], idx);
				}
				else if (isValid(node->children[c]->bbox, _data[idx])) {
					_recursive_remove_index<SEARCH_ALL>(node->children[c], idx);
					if constexpr (SINGLE_DATA) {break;}
				}
			}
		}
		else
		{
			read_lock.unlock();
			std::lock_guard<std::shared_mutex> write_lock(node->mutex);

			//make sure we are still in a leaf node
			if (!isLeaf(node)) {return _recursive_remove_index<SEARCH_ALL>(node, idx);}
			removeDataIdx(node, idx);
		}
	}


	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_node_properties(
		const OctreeParallelNode<DIM,N_DATA,T>* node,
		BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::Stats& result) const {
			if (node == nullptr) {return;}
			std::shared_lock<std::shared_mutex> lock(node->mutex);

			result.n_nodes++;
			if (isLeaf(node)) {result.n_leafs++;}
			result.memory_used_bytes += sizeof(decltype(*node));
			result.memory_reserved_bytes += sizeof(decltype(*node));

			if (node->data_idx != nullptr) {
				result.n_used_indices += node->cursor;
				result.n_indices_capacity += N_DATA;

				result.memory_used_bytes += node->cursor * sizeof(size_t);
				result.memory_reserved_bytes += N_DATA * sizeof(size_t);
			}

			result.max_depth = std::max(result.max_depth, node->depth);

			for (int c = 0; c < OctreeParallelNode<DIM,N_DATA,T>::N_CHILDREN; c++) {
				_recursive_node_properties(node->children[c], result);
			}
		}

	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	void BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_recursive_compute_data_leaf_bbox(
		const OctreeParallelNode<DIM,N_DATA,T>* node,
		Box_t& bbox) const
	{
		assert(node);

		//engage shared lock on the way down
		std::shared_lock<std::shared_mutex> read_lock(node->mutex);

		if (!isLeaf(node))
		{
			//recurse into any child that extends past the current bbox
			for (int c=0; c<N_CHILDREN; c++) {
				assert(node->children[c]);
				if (!bbox.contains(node->children[c]->bbox)) {
					_recursive_compute_data_leaf_bbox(node->children[c], bbox);
				}
			}
		}
		else
		{
			//exit if we have no data
			if (node->cursor==0) {return;}
			assert(node->data_idx);

			//check if we are the closest leaf with data
			bbox = combine(bbox,node->bbox);
		}
	}



	///////////////////////////////////////////////////////////////////////////
	//////////////////////// OTHER PRIVATE METHODS ////////////////////////////
	///////////////////////////////////////////////////////////////////////////
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA,typename T>
	bool BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::_switch_to_flush()
	{
		{
			std::lock_guard<std::mutex> lock(_thread_manage_mutex);
			if (n_flushing_threads >= n_max_flushing_threads) {return false;}
			n_flushing_threads++;
			if (n_flushing_threads>n_max_flushing_threads)
			{
				throw std::runtime_error("BasicParallelOctree::_switch_to_flush() - flushing thread count exceeds maximum");
			}
		}
		
		QueueData_t data;
		while (_queue.pop(data)) {
			assert(data.idx < (size_t) -1);
			assert(data.node);
			
			//insert data until successful
			int flag = _recursive_insert_data<true>(data.node, _data[data.idx], data.idx);

			if (flag == -1)
			{
				throw std::runtime_error("BasicParallelOctree::_switch_to_flush() - couldn't insert data at index " + std::to_string(data.idx));
			}

			if (flag == 0)
			{
				throw std::runtime_error("BasicParallelOctree::_switch_to_flush() - data at index " + std::to_string(data.idx) + " was already inserted");
			}
		}

		{
			std::lock_guard<std::mutex> lock(_thread_manage_mutex);
			n_flushing_threads--;
			if (n_flushing_threads<0)
			{
				throw std::runtime_error("BasicParallelOctree::_switch_to_flush() - flushing thread count is negative");
			}
		}

		return true;
	}







	////////////////////////////////////////////////////////////////////////////
	/////////////////////// OTHER UTILITY METHODS //////////////////////////////
	////////////////////////////////////////////////////////////////////////////
	template<typename DATA_T, bool SINGLE_DATA, int DIM, int N_DATA, typename T>
	inline Box<DIM,T> BasicParallelOctree<DATA_T,SINGLE_DATA,DIM,N_DATA,T>::bbox_tight() const
	{
		Point_t center = _root->bbox.center();
		Box_t _bbox(center,center);
		_recursive_compute_data_leaf_bbox(_root, _bbox);
		return _bbox;
	}
}