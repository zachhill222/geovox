#pragma once

#include <cassert>
#include <deque>


#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>


////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Single producer single consumer queue
////////////////////////////////////////////////////////////////////////////////////////////////////////
namespace gutil {
	/////////////////////////////////////////////////
	/// A queue data structure for use in multithreaded applications.
	/// This requires a single thread to be the inserter and a (possibly different)
	/// thread for pulling out data.
	/// Making CAPACITY too large can cause stack overflow, especially when
	/// each thread gets its own queue. Sometimes the queue will get full,
	/// so data should be inserted with while(!try_push(data)). The number of
	/// unssuccessful pushes are tracked in buffer_bumps. The buffer/queue can
	/// be moved to the heap if it needs to be larger, but I have found that
	/// having a large buffer doesn't necessarily lead to performance gains.
	///
	/// @tparam DATA_T   The type of data to be stored 
	/// @tparam CAPACITY The maximum number of elements that can be stored in the
	///                  queue/buffer at a time. Should be a power of 2 for faster
	///                  mod computation. If this is changed to be re-sizable,
	///                  the mod computation may need to be implemented by hand
	///                  because the compiler won't know it is a power of 2.
	///
	/// @tparam USE_STACK  The buffer can be allocated on either the heap or the stack.
	/////////////////////////////////////////////////
	template<typename DATA_T, size_t CAPACITY=2048, bool USE_STACK=true>
	struct SingleInSingleOut {
		//thread A may only need head while thread B may only need tail.
		//setting alignas(64) keeps one thread from loading both but only using one.
		alignas(64) std::atomic<size_t> head{0};
		alignas(64) std::atomic<size_t> tail{0};
		std::atomic<size_t> count{0};
		size_t queue_id;

		#if USE_STACK
			DATA_T queue[CAPACITY];
			virtual ~SingleInSingleOut() {}
		#else
			DATA_T* queue = new DATA_T[CAPACITY];
			virtual ~SingleInSingleOut() {delete[] queue;}
		#endif

		std::atomic<size_t> buffer_bumps{0};

		bool try_push(DATA_T&& item) {
			size_t current_tail = tail.load(std::memory_order_acquire);
			size_t next_tail = (current_tail + 1) % CAPACITY;

			if (next_tail == head.load(std::memory_order_acquire)) {
				buffer_bumps.fetch_add(1, std::memory_order_relaxed); //buffer_bumps is mostly of profiling
				return false;  // Queue full
			}

			queue[current_tail] = std::move(item);
			tail.store(next_tail);
			count++;
			return true;
		}

		bool try_pop(DATA_T& item) {
			size_t current_head = head.load(std::memory_order_acquire);

			if (current_head == tail.load(std::memory_order_acquire)) {
				return false;  // Queue empty
			}

			item = queue[current_head];
			head.store((current_head + 1) % CAPACITY, std::memory_order_release);
			count.fetch_add(-1, std::memory_order_release);
			return true;
		}

		template<typename DataCondition>
		bool try_pop(DATA_T& item) {
			size_t current_head = head.load(std::memory_order_acquire);

			if (current_head == tail.load(std::memory_order_acquire)) {
				return false;  // Queue empty
			} else if (!DataCondition(queue[current_head])) {
				return false; // next data is not valid
			}

			head.store((current_head + 1) % CAPACITY, std::memory_order_release);
			count.fetch_add(-1, std::memory_order_release);
			return true;
		}

		bool empty(std::memory_order order = std::memory_order_seq_cst) const {
			return head.load(order) == tail.load(order);
		}
	};


	/////////////////////////////////////////////////
	/// A queue data structure for use in multithreaded applications.
	/// This allows multiple threads for both putting data in and pulling
	/// data out. This is a simple wrapper around std::deque.
	///
	/// @tparam DATA_T   The type of data to be stored
	/////////////////////////////////////////////////
	template<typename DATA_T>
	class MultipleInMultipleOut {
	public:
		MultipleInMultipleOut() : dq{} {}

		bool empty() const
		{
			std::shared_lock<std::shared_mutex> lock(mutex);
			return dq.empty();
		}

		size_t capacity() const
		{
			std::shared_lock<std::shared_mutex> lock(mutex);
			return dq.size();
		}

		void clear()
		{
			std::lock_guard<std::shared_mutex> lock(mutex);
			dq.clear();
		}

		size_t size() const
		{
			std::shared_lock<std::shared_mutex> lock(mutex);
			return dq.size();
		}

		//non blocking pop
		template<typename Predicate>
		bool try_pop(DATA_T& item, const Predicate& pred) {
			std::unique_lock<std::shared_mutex> lock(mutex, std::try_to_lock);
			if (!lock.owns_lock()) {return false;}
			else {return pop_unlocked(item, pred);}
		}

		bool try_pop(DATA_T& item) {
			std::unique_lock<std::shared_mutex> lock(mutex, std::try_to_lock);
			if (!lock.owns_lock()) {return false;}
			else {return pop_unlocked(item);}
		}

		//non blocking push
		bool try_push(DATA_T&& item) {
			std::unique_lock<std::shared_mutex> lock(mutex, std::try_to_lock);
			if (!lock.owns_lock()) {return false;}
			else {return push_unlocked(std::move(item));}
		}

		//blocking pop
		template<typename Predicate>
		bool pop(DATA_T& item, const Predicate& pred) {
			std::lock_guard<std::shared_mutex> lock(mutex);
			return pop_unlocked(item, pred);
		}

		bool pop(DATA_T& item) {
			std::lock_guard<std::shared_mutex> lock(mutex);
			return pop_unlocked(item);
		}

		//blocking push
		bool push(DATA_T&& item) {
			std::lock_guard<std::shared_mutex> lock(mutex);
			return push_unlocked(std::move(item));
		}

		inline bool push(const DATA_T& item) {
			DATA_T new_item{item};
			return push(std::move(new_item));
		}

	private:
		mutable std::shared_mutex mutex;
		std::deque<DATA_T> dq;

		bool push_unlocked(DATA_T&& item) {
			dq.push_back(std::move(item));
			return true;
		}

		template<typename Predicate>
		bool pop_unlocked(DATA_T& item, const Predicate& pred) {
			if (dq.empty()) {return false;}
			if (!pred(dq.front())) {return false;}
			item = std::move(dq.front());
			dq.pop_front();
			return true;
		}

		bool pop_unlocked(DATA_T& item) {
			if (dq.empty()) {return false;}
			item = std::move(dq.front());
			dq.pop_front();
			return true;
		}
	};
}