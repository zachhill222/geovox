#pragma once

#include "gutil.hpp"
#include "simd_keys/mesh/mesh_key_implementation.hpp"
#include "util/byte_print.hpp"

#include <cstdint>
#include <vector>
#include <span>
#include <type_traits>


namespace GV {
namespace Keys {
		

	//////////////////////////////////////////////////////////////////////////////
	/// A container for tracking keys mesh feature (or dof) keys.
	/// A one byte is reserved per possible key up to some MAX_DEPTH. The first
	/// bit of these masks is reserved for an active flag. The other 7 are for use 
	/// by derived classes. Additonally, a compressed list is stored for better 
	/// actions on only the active keys. The raw uint64_t for active keys are stored 
	/// for better simd actions. 
	///
	/// For the intended use case, the maximum number of keys is likely known at
	/// compile time but too large for the stack anyways.
	///////////////////////////////////////////////////////////////////////////////
	struct HybridKeyTracker {


		/////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////////////////////
		static constexpr uint8_t ACTIVE_BIT 			= 0b00000001;	//the mask that all classes must agree is the active marker
		static constexpr uint8_t FREE_BITS 	            = 0b11111110;	//each class may use these themselves


		/////////////////////////////////////////////////////////////////////////
		/// Storage. Store a vector<uint8_t> for O(1) active queries.
		/// Additionally, store a compressed list of active dofs for tracking
		/// global DOF numbers. It is essential for fast quadrature that we may
		/// look up all active DOFs whos support OVERLAPS a given active element.
		///
		/// Using uint8_t instead of bool guarantees thread safe access of different elements
		/// and allows one bit to be used for an "is active" flag and another bit for
		/// "has been refined" flag, which is useful for hierarchical methods. Additionally,
		/// it gives us 6 more bits that could be used for other purposes.
		///
		/// It is often necessary to look up a key by its value. To do this, we use
		/// a "BinSort" to divide keys into some number of categories and then
		/// sort within each bin by key comparision. By default, the key bins correspond
		/// to the pairity of their cartesian indices, but others can be provided.
		/// 
		/// For example, if the keys are colored, the color can be used as the bin number.
		/// The sorted list can be used for a 'cononical' global numbering.
		/////////////////////////////////////////////////////////////////////////
	protected:
		mutable std::vector<uint8_t> 	key_mask{};				//	marked as mutable so that classes with const references can have a reserved bit to use.
		std::vector<uint64_t>			active_keys{};			//	a compressed list of the active keys
		mutable gutil::ThreadPool 		threads{};				//	max hardware concurency by default
		gutil::BinSort<uint64_t>		sorter{};				//	a class to sort and look up keys by their value.
		bool							is_sorted_{false};		//  a flag to track when the active keys are up to date
		bool							is_collected_{false};	//  a flag to track when the active keys are ready to be sorted (managed by the derived class)
	public:
		/////////////////////////////////////////////////////////////////////////
		/// Constructors. These objects are generally quite large, so copying
		/// should be avoided when possible.
		/////////////////////////////////////////////////////////////////////////
		HybridKeyTracker() {}
		HybridKeyTracker(size_t total_possible_keys) : key_mask(total_possible_keys) {};
		HybridKeyTracker(size_t total_possible_keys, size_t n_threads) : 
									key_mask(total_possible_keys), threads{n_threads} {};
		
		HybridKeyTracker(const HybridKeyTracker&) = delete;				// we cannot stop the other tracker's thread pool here

		HybridKeyTracker& operator=(const HybridKeyTracker& other) {
			if (this != &other) {
				if (other.key_mask.size() > (size_t{1} << 20)) {
					GUTIL_LOG("Copying a HybridKeyTracker with ", other.key_mask.size(), " bytes of mask storage and ",
						8*other.active_keys.size(), " bytes of active key storage");
				}
				threads.wait_idle();
				other.threads.wait_idle();
				key_mask 	    = other.key_mask;
				active_keys     = other.active_keys;
				sorter			= other.sorter;
			}
			return *this;
		}

		HybridKeyTracker(HybridKeyTracker&& other) :
			key_mask{std::move(other.key_mask)},
			active_keys{std::move(other.active_keys)},
			threads{other.threads.n_threads()},
			sorter{other.sorter} {}

		[[nodiscard]] HybridKeyTracker& operator=(HybridKeyTracker&& other) noexcept {
			if (this != &other) {
				threads.wait_idle();
				other.threads.wait_idle();
				key_mask 	    = std::move(other.key_mask);
				active_keys     = std::move(other.active_keys);
				sorter			= other.sorter;
			}
			return *this;
		}

		/////////////////////////////////////////////////////////////////////////
		/// A few methods for initialziation and working with multithreading.
		///
		/// Note that is_collected_ is the derived classes's responsibility to update.
		/////////////////////////////////////////////////////////////////////////
		void wait_idle() const noexcept 								{ threads.wait_idle();}
		[[nodiscard]] size_t n_threads() noexcept 						{return threads.n_threads();}

		[[nodiscard]] constexpr size_t n_active_keys() const noexcept 	{ return active_keys.size();}
		[[nodiscard]] constexpr size_t n_possible_keys() const noexcept { return key_mask.size();}
		[[nodiscard]] constexpr bool is_sorted() const noexcept 		{ return is_sorted_;}
		[[nodiscard]] constexpr bool is_collected() const noexcept		{ return is_collected_;}

		[[nodiscard]] constexpr bool is_current() const noexcept {
			if (!is_collected_) {
				GUTIL_ERROR("the keys are not collected");
			}
			if (!is_sorted_) {
				GUTIL_ERROR("the keys are not sorted");
			}
			return is_sorted_ && is_collected_;
		}

		void mark_stale() noexcept {
			is_sorted_    = false;
			is_collected_ = false;
		}

		void clear() noexcept {
			std::fill(key_mask.begin(), key_mask.end(), 0);
			active_keys.clear();
			mark_stale();
		}

		void init_key_mask(size_t total_possible_keys) noexcept {
			key_mask.clear();
			key_mask.resize(total_possible_keys,0);
			key_mask.shrink_to_fit();

			active_keys.clear();
			sorter.clear();
			mark_stale();
		}


		/////////////////////////////////////////////////////////////////////////
		/// A few methods to help with viewing the active list as a particular type
		/////////////////////////////////////////////////////////////////////////
		template<typename KeyTypeOut, typename KeyTypeIn> requires(sizeof(KeyTypeIn)==8 && sizeof(KeyTypeOut)==8)
		[[nodiscard]] static std::span<KeyTypeOut> reinterpret_key_span(std::span<KeyTypeIn> list) noexcept {
			if constexpr (std::same_as<KeyTypeIn, KeyTypeOut>) {return list;}
			else {
				std::span<KeyTypeOut> view {reinterpret_cast<KeyTypeOut*>(list.data()), list.size()};
				#ifndef NDEBUG
					//make sure that the span is correctly interprets the raw values
					GUTIL_ASSERT(list.size() == view.size());
					GUTIL_ASSERT(reinterpret_cast<uintptr_t>(list.data()) == 
									reinterpret_cast<uintptr_t>(view.data()) && "begin address does not match");
					GUTIL_ASSERT(reinterpret_cast<uintptr_t>(list.data()+list.size()) == 
									reinterpret_cast<uintptr_t>(view.data()+view.size()) && "end address does not match");
					for (size_t i=0; i<std::min(size_t{100},view.size()); ++i) {
						uint64_t raw  = static_cast<uint64_t>(list[i]);
						GUTIL_ASSERT(raw == static_cast<uint64_t>(view[i]) && "raw bytes changed");
						GUTIL_ASSERT(KeyTypeIn{raw} == list[i] && "reconstructing in type failed");
						GUTIL_ASSERT(KeyTypeOut{raw} == view[i] && "reconstructing out type failed");
					}
				#endif
				return view;
			}
		}

		template<typename KeyTypeOut, typename KeyTypeIn> requires(sizeof(KeyTypeIn)==8 && sizeof(KeyTypeOut)==8)
		[[nodiscard]] static std::span<const KeyTypeOut> reinterpret_key_span(std::span<const KeyTypeIn> list) noexcept {
			if constexpr (std::same_as<KeyTypeIn, KeyTypeOut>) {return list;}
			else {
				std::span<const KeyTypeOut> view {reinterpret_cast<const KeyTypeOut*>(list.data()), list.size()};
				#ifndef NDEBUG
					//make sure that the span is correctly interprets the raw values
					GUTIL_ASSERT(list.size() == view.size());
					GUTIL_ASSERT(reinterpret_cast<uintptr_t>(list.data()) == 
									reinterpret_cast<uintptr_t>(view.data()) && "begin address does not match");
					GUTIL_ASSERT(reinterpret_cast<uintptr_t>(list.data()+list.size()) == 
									reinterpret_cast<uintptr_t>(view.data()+view.size()) && "end address does not match");
					for (size_t i=0; i<std::min(size_t{100},view.size()); ++i) {
						uint64_t raw  = static_cast<uint64_t>(list[i]);
						GUTIL_ASSERT(raw == static_cast<uint64_t>(view[i]) && "raw bytes changed");
						GUTIL_ASSERT(KeyTypeIn{raw} == list[i] && "reconstructing in type failed");
						GUTIL_ASSERT(KeyTypeOut{raw} == view[i] && "reconstructing out type failed");
					}
				#endif
				return view;
			}
		}

		/////////////////////////////////////////////////////////////////////////
		/// A few methods read and write to the masks.
		/// The ACTIVE_BIT is the only specific single bit query, but derived
		///	classes can implement their own. Additionally, each key has it's own
		/// way to get a linear index for it. Each derived class may wish to 
		/// change the query (i.e., get_mask(key.linear_index()))
		/////////////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t get_mask(uint64_t idx) const noexcept {
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t& get_mask_ref(uint64_t idx) noexcept {
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t& get_mask_ref_pseudo_const(uint64_t idx) const noexcept {
			//note key_mask is mutable, but we want to know when we use this property
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		[[nodiscard]] std::span<const uint8_t> get_mask(uint64_t start, uint64_t end) const noexcept {
			//get a range of masks
			GUTIL_ASSERT(start<end && end<=key_mask.size());
			return std::span<const uint8_t>(key_mask.begin()+start, key_mask.begin()+end);
		}

		[[nodiscard]] std::span<uint8_t> get_mask_ref(uint64_t start, uint64_t end) noexcept {
			//get a range of masks
			GUTIL_ASSERT(start<end && end<=key_mask.size());
			return std::span<uint8_t>(key_mask.begin()+start, key_mask.begin()+end);
		}

		[[nodiscard]] std::span<uint8_t> get_mask_ref_pseudo_const(uint64_t start, uint64_t end) const noexcept {
			//get a range of masks
			GUTIL_ASSERT(start<end && end<=key_mask.size());
			return std::span<uint8_t>(key_mask.begin()+start, key_mask.begin()+end);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_active(uint64_t idx) const noexcept {
			//read the active bit
			return get_mask(idx) & ACTIVE_BIT;
		}

		GUTIL_DECLARE_SIMD()
		void set_active(uint64_t idx, bool val) noexcept {
			//set the active bit
			val ? get_mask_ref(idx) |= ACTIVE_BIT  :  get_mask_ref(idx) &= ~ACTIVE_BIT;
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] bool set_active_compare_exchange(uint64_t idx, bool val) noexcept {
			//set the active bit and return true if it was changed
			uint8_t& byte  = get_mask_ref(idx);
			const bool old = byte & ACTIVE_BIT;
			val ? byte |= ACTIVE_BIT  :  byte &=~ACTIVE_BIT;
			return old!=val;
		}


		/////////////////////////////////////////////////////////////////////////
		/// A few methods to do bulk bitwise operations on the entire key set.
		/// These can be use for example, to set a reserved bit to mark the start of an bulk
		/// operation.
		/////////////////////////////////////////////////////////////////////////
		void conditional_bitwise_and_all_masks(uint8_t condition_mask, uint8_t mask) noexcept {
			dispatch_parallel_key_mask( 
				[](std::span<uint8_t> mask_span, uint8_t condition_mask, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						if(mask_span[i]&condition_mask) {mask_span[i]&=mask;}
					}
				}, condition_mask, mask);
			threads.wait_idle();
		}

		void conditional_bitwise_or_all_masks(uint8_t condition_mask, uint8_t mask) noexcept {
			dispatch_parallel_key_mask( 
				[](std::span<uint8_t> mask_span, uint8_t condition_mask, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						if(mask_span[i]&condition_mask) {mask_span[i]|=mask;}
					}
				}, condition_mask, mask);
			threads.wait_idle();
		}

		void unconditional_bitwise_and_all_masks(uint8_t mask) noexcept {
			dispatch_parallel_key_mask( 
				[](std::span<uint8_t> mask_span, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						mask_span[i]&=mask;
					}
				}, mask);
			threads.wait_idle();
		}

		void unconditional_bitwise_or_all_masks(uint8_t mask) noexcept {
			dispatch_parallel_key_mask( 
				[](std::span<uint8_t> mask_span, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						mask_span[i]|=mask;
					}
				}, mask);
			threads.wait_idle();
		}


		/////////////////////////////////////////////////////////////////////////
		/// A few methods to dispatch threaded processes to either each mask or each active key.
		///
		/// Sometimes it is useful for the user to know which thread number (from 0 to n_threads-1)
		/// is executing the Task. Valid action signatures are
		///		a) void(span<uint64_t>, size_t, args...)
		///		b) void(span<uint64_t>, args...)
		///		c) void(span<const uint64_t>, size_t, args...)
		///		d) void(span<const uint64_t>, args...)
		///
		/// when acting over active keys and
		///		a) void(span<uint64_t>, size_t, args...)
		///		b) void(span<uint64_t>, args...)
		///		c) void(span<const uint64_t>, size_t, args...)
		///		d) void(span<const uint64_t>, args...)
		///
		/// when acting over masks. The size_t argument is the thread id and is a
		/// unique number between 0 and n_threads-1.
		///
		/// Note that these methods only dispatch to the thread pool. Be sure to call
		/// wait_idle() before using the results.
		///
		/// Note that you are free to use OpenMP within the submitted Task. However, you may
		/// wish to reduce the number of threads in this thread pool by using set_thread_pool_size().
		/// Ideally, you can use the full number of hardware threads and use SIMD within each task.
		/////////////////////////////////////////////////////////////////////////
		template<typename Task, typename...Args>
		void dispatch_parallel_active_keys(Task&& action, Args&&... args) noexcept {
			GUTIL_ASSERT(is_current() && "the active keys are stale");
			GUTIL_ASSERT(active_keys.size()>0 && "HybridKeyTracker - no keys found. Did you forget to collect them?");
			GUTIL_ASSERT(threads.n_threads() > 0);

			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			const size_t n_keys 		= active_keys.size();
			const size_t n_workers 		= threads.n_threads();
			const size_t dof_per_thread = n_keys/n_workers;

			for (size_t tid=0; tid<n_workers; ++tid) {
				const size_t start = tid * dof_per_thread;
				const size_t end = (tid==n_workers-1) ? n_keys : start + dof_per_thread;

				if constexpr (std::is_invocable_r_v<void, Task, std::span<uint64_t>, int, Args...>) {
					std::span<uint64_t> list(active_keys.begin()+start, active_keys.begin()+end);
					threads.submit(action, list, tid, std::forward<Args>(args)...);
				}
				else if constexpr (std::is_invocable_r_v<void, Task, std::span<uint64_t>, Args...>) {
					std::span<uint64_t> list(active_keys.begin()+start, active_keys.begin()+end);
					threads.submit(action, list, std::forward<Args>(args)...);
				}
				else {
					GUTIL_ERROR("the Task signature should be one of: void(span<uint64_t>, size_t, args...) or void(span<uint64_t>, args...)");
					GUTIL_ABORT("Arguments did not match what was expected");
				}
			}
		}

		template<typename Task, typename...Args>
		void dispatch_parallel_active_keys(Task&& action, Args&&... args) const noexcept {
			GUTIL_ASSERT(is_current() && "the active keys are stale");
			GUTIL_ASSERT(active_keys.size()>0 && "HybridKeyTracker - no keys found. Did you forget to collect them?");
			GUTIL_ASSERT(threads.n_threads() > 0);

			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			const size_t n_keys 		= active_keys.size();
			const size_t n_workers 		= threads.n_threads();
			const size_t dof_per_thread = n_keys/n_workers;

			for (size_t tid=0; tid<n_workers; ++tid) {
				const size_t start = tid * dof_per_thread;
				const size_t end = (tid==n_workers-1) ? n_keys : start + dof_per_thread;

				if constexpr (std::is_invocable_r_v<void, Task, std::span<uint64_t>, int, Args...>) {
					std::span<uint64_t> list(active_keys.begin()+start, active_keys.begin()+end);
					threads.submit(action, list, tid, std::forward<Args>(args)...);
				}
				else if constexpr (std::is_invocable_r_v<void, Task, std::span<uint64_t>, Args...>) {
					std::span<uint64_t> list(active_keys.begin()+start, active_keys.begin()+end);
					threads.submit(action, list, std::forward<Args>(args)...);
				}
				else {
					GUTIL_ERROR("the Task signature should be one of: void(span<const uint64_t>, size_t, args...) or void(span<const uint64_t>, args...)");
					GUTIL_ABORT("Arguments did not match what was expected");
				}
			}
		}

		template<typename Task, typename...Args>
		void dispatch_parallel_key_mask(Task&& action, Args&&... args) noexcept {
			GUTIL_ASSERT(key_mask.size()>0 && "HybridKeyTracker - key_mask is not initialized");
			GUTIL_ASSERT(threads.n_threads() > 0);

			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			const size_t n_threads = threads.n_threads();
			const size_t bytes_per_thread = key_mask.size()/n_threads;

			for (size_t tid=0; tid<n_threads; ++tid) {
				const size_t start = tid * bytes_per_thread;
				const size_t end = (tid==n_threads-1) ? key_mask.size() : start + bytes_per_thread;

				if constexpr (std::is_invocable_r_v<void, Task, std::span<uint8_t>, int, Args...>) {
					std::span<uint8_t> list(key_mask.begin()+start, key_mask.begin()+end);
					threads.submit(action, list, tid, std::forward<Args>(args)...);
				}
				else if constexpr (std::is_invocable_r_v<void, Task, std::span<uint8_t>, Args...>) {
					std::span<uint8_t> list(key_mask.begin()+start, key_mask.begin()+end);
					threads.submit(action, list, std::forward<Args>(args)...);
				}
				else {
					GUTIL_ERROR("the Task signature should be one of: void(span<uint8_t>, size_t, args...) or void(span<uint8_t>, args...)");
					GUTIL_ABORT("Arguments did not match what was expected");
				}
			}
		}

		template<typename Task, typename...Args>
		void dispatch_parallel_key_mask(Task&& action, Args&&... args) const noexcept {
			GUTIL_ASSERT(key_mask.size()>0 && "HybridKeyTracker - key_mask is not initialized");
			GUTIL_ASSERT(threads.n_threads() > 0);

			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			const size_t n_threads = threads.n_threads();
			const size_t bytes_per_thread = key_mask.size()/n_threads;

			for (size_t tid=0; tid<n_threads; ++tid) {
				const size_t start = tid * bytes_per_thread;
				const size_t end = (tid==n_threads-1) ? key_mask.size() : start + bytes_per_thread;
				
				if constexpr (std::is_invocable_r_v<void, Task, std::span<const uint8_t>, int, Args...>) {
					std::span<const uint8_t> list(key_mask.begin()+start, key_mask.begin()+end);
					threads.submit(action, list, tid, std::forward<Args>(args)...);
				}
				else if constexpr (std::is_invocable_r_v<void, Task, std::span<const uint8_t>, Args...>) {
					std::span<const uint8_t> list(key_mask.begin()+start, key_mask.begin()+end);
					threads.submit(action, list, std::forward<Args>(args)...);
				}
				else {
					GUTIL_ERROR("the Task signature should be one of: void(span<const uint8_t>, size_t, args...) or void(span<const uint8_t>, args...)");
					GUTIL_ABORT("Arguments did not match what was expected");
				}
			}
		}


		/////////////////////////////////////////////////////////////////////////
		/// A few methods to work with the active keys
		/////////////////////////////////////////////////////////////////////////
		template<typename KeyType>
		void collect_active_keys() noexcept {
			GUTIL_TIMER("Collecting active keys (", key_mask.size(), " to check)");

			const size_t n_threads = threads.n_threads();
			const size_t n_keys_per_worker = (n_threads==0) ? key_mask.size() : key_mask.size()/n_threads;
			std::vector<std::vector<uint64_t>> thread_keys(n_threads);
			
			auto job = [n_keys_per_worker, &thread_keys](std::span<const uint8_t> masks, size_t tid) {
				size_t key_index = tid*n_keys_per_worker;	//we need to track the location of the mask we are examing
				for (size_t i=0; i<masks.size(); ++i, ++key_index) {
					if (masks[i]&ACTIVE_BIT) {thread_keys[tid].push_back(static_cast<uint64_t>(KeyType::MakeFromIndex(key_index)));}
				}
			};

			dispatch_parallel_key_mask(job);
			active_keys.clear();
			
			threads.wait_idle();
			for (size_t tid=0; tid<n_threads; ++tid) {
				active_keys.insert(active_keys.end(), std::make_move_iterator(thread_keys[tid].begin()),
													  std::make_move_iterator(thread_keys[tid].end()));
			}

			is_collected_ = true;
		}


		/////////////////////////////////////////////////////////////////////////
		/// A few methods to handle sorting keys.
		///
		/// Note that the bin type needs to be unsigned.
		/////////////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] int static constexpr default_key_bin(uint64_t key) noexcept {
			return Mesh3D::IndexPairity_SIMD(key);
		}

		template<typename BinFun = decltype(&HybridKeyTracker::default_key_bin)>
		void sort_active_keys(int N=8, BinFun&& fun = &HybridKeyTracker::default_key_bin) noexcept {
			GUTIL_ASSERT(is_collected_ && "The keys were not collected. Call collect_active_keys<KeyType>() to collect them.");
			GUTIL_TIMER("sorting ", active_keys.size(), " keys into ", N, " bins");
			sorter = sort_keys(std::span<uint64_t>{active_keys}, N, std::forward<BinFun>(fun));
			is_sorted_ = true;
		}

		template<typename BinFun> requires( std::is_invocable_r_v<int, BinFun, uint64_t>)
		[[maybe_unused]] gutil::BinSort<uint64_t> sort_keys(std::span<uint64_t> list, int N, BinFun&& fun) const noexcept {
			//link the current active keys to the sorter
			gutil::BinSort<uint64_t> local_sorter{list, N};
			GUTIL_ASSERT(local_sorter.n_bins() == N);
			
			//do the primary bin sort
			local_sorter.dispatch_sort(std::forward<BinFun>(fun), &threads);
			threads.wait_idle();

			//sort within bins
			for (int i=0; i<local_sorter.n_bins(); ++i) {
				threads.submit( [](std::span<uint64_t> list){ std::sort(list.begin(), list.end()); }, local_sorter.get_bin(i));
			}
			threads.wait_idle();

			//return the sorter so that the caller has views into each bin
			return local_sorter;
		}

		template<typename BinFun> requires( std::is_invocable_r_v<int, BinFun, uint64_t>)
		[[nodiscard]] size_t lookup_key(const uint64_t key, BinFun&& fun = &HybridKeyTracker::default_key_bin) const noexcept {
			GUTIL_ASSERT(is_sorted());

			const int bin = fun(key);
			GUTIL_ASSERT(bin < sorter.n_bins());

			auto list = sorter.get_bin(bin);
			auto it = std::lower_bound(list.begin(), list.end(), key);
			return (it==list.end() || *it!=key) ? size_t(-1) : sorter.bin_start(bin) + std::distance(list.begin(), it);
		}

		template<typename BinFun> requires( std::is_invocable_r_v<int, BinFun, uint64_t>)
		std::vector<size_t> lookup_keys_and_sort(std::span<uint64_t> keys, int N, BinFun&& fun) const noexcept {
			std::vector<size_t> result(keys.size());
			lookup_keys_and_sort(keys, std::span<size_t>{result}, N, std::forward<BinFun>(fun));
			return result;
		}

		template<typename BinFun> requires( std::is_invocable_r_v<int, BinFun, uint64_t>)
		void lookup_keys_and_sort(std::span<uint64_t> keys, std::span<size_t> global_index, int N, BinFun&& fun) const noexcept {
			GUTIL_ASSERT(is_sorted());
			GUTIL_ASSERT(keys.size() == global_index.size());
			GUTIL_ASSERT(N==sorter.n_bins());

			auto local_sorter = sort_keys(keys, N, std::forward<BinFun>(fun));
			std::vector<size_t> result(keys.size());

			//go through each bin and find the global index
			size_t n = 0;
			for (int i=0; i<N; ++i) {
				auto local_list  = local_sorter.get_bin(i);
				auto global_list = sorter.get_bin(i);

				for (size_t j=0; j<local_sorter.bin_size(i); ++j, ++n) {
					uint64_t query_key = local_list[j];
					GUTIL_ASSERT(query_key==keys[n]);

					auto it = std::lower_bound(global_list.begin(), global_list.end(), query_key);
					if ( it==global_list.end() || *it!=query_key) { global_index[n] = size_t(-1); }
					else {
						global_index[n] = sorter.bin_start(i) + 
								static_cast<size_t>(std::distance(global_list.begin(), it));
					}
				}
			}

			#ifndef NDEBUG
			for (size_t n=0; n<keys.size(); ++n) {
				GUTIL_ASSERT(keys[n] == active_keys[global_index[n]]);
			}
			#endif
		}
		










	};



}}



