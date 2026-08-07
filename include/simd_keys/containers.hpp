#pragma once

#include "gutil.hpp"
#include "simd_keys/mesh/mesh_key_implementation.hpp"
#include "util/byte_print.hpp"

#include <cstdint>
#include <vector>
#include <span>
#include <type_traits>
#include <shared_mutex>
#include <atomic>
#include <bit>

///////////////////////////////////////////////////////////////////////////////////
/// Macros for synchronizing operations within a key container
///
/// To use in a derived class, don't forget using BASE::is_mask_unstable etc.
///////////////////////////////////////////////////////////////////////////////////
#define GV_ASSERT_KEY_MASK_UNSTABLE_STATE		GUTIL_ASSERT(is_key_mask_unstable());
#define GV_ASSERT_KEY_MASK_STABLE_STATE			GUTIL_ASSERT(is_key_mask_stable());

#define GV_ASSERT_ACTIVE_KEYS_UNSTABLE_STATE	GUTIL_ASSERT(is_active_keys_unstable());
#define GV_ASSERT_ACTIVE_KEYS_STABLE_STATE		GUTIL_ASSERT(is_active_keys_stable());

//////////////////////////////////////////////////////////////////////
/// BEGIN/END macros for entering a synchronized region.
/// _UNSTABLE variants take the exclusive lock (mutating); _STABLE variants
/// take the shared lock (read-only, multiple readers may coexist).
/// Each BEGIN declares a lock variable held for the rest of the enclosing
/// scope; the matching END clears the diagnostic flag (the lock itself is
/// released via RAII at scope exit, regardless of whether END is called).
/// 
/// We assert that immediately before any BEGIN, the state is stable,
/// and before any END, the state was in the stated scope name.
/// For example, when exiting a stable block, check that the stable flag
/// is still active. This may catch bugs if the lock went out of scope
/// on accident.
///
/// These locks should only go on "large" operations and the "atomic"
/// operations should use GV_ASSERT_****_STABLE/UNSTABLE to check that 
/// they are called appropriately.
//////////////////////////////////////////////////////////////////////
#define GV_BEGIN_MASK_UNSTABLE						\
	GV_ASSERT_KEY_MASK_STABLE_STATE					\
	auto gv_key_mask_lock = begin_key_mask_unstable();

#define GV_END_MASK_UNSTABLE						\
	GV_ASSERT_KEY_MASK_UNSTABLE_STATE				\
	end_key_mask_unstable();

#define GV_BEGIN_MASK_STABLE						\
	GV_ASSERT_KEY_MASK_STABLE_STATE					\
	auto gv_key_mask_lock = begin_key_mask_stable();

#define GV_END_MASK_STABLE							\
	GV_ASSERT_KEY_MASK_STABLE_STATE 				\
	end_key_mask_stable();

#define GV_BEGIN_ACTIVE_UNSTABLE					\
	GV_ASSERT_ACTIVE_KEYS_STABLE_STATE				\
	auto gv_active_keys_lock = begin_active_keys_unstable();

#define GV_END_ACTIVE_UNSTABLE						\
	GV_ASSERT_ACTIVE_KEYS_UNSTABLE_STATE 			\
	end_active_keys_unstable();

#define GV_BEGIN_ACTIVE_STABLE						\
	GV_ASSERT_ACTIVE_KEYS_STABLE_STATE 				\
	auto gv_active_keys_lock = begin_active_keys_stable();

#define GV_END_ACTIVE_STABLE						\
	GV_ASSERT_ACTIVE_KEYS_STABLE_STATE 				\
	end_active_keys_stable();

#define GV_BEGIN_UNSTABLE							\
	GV_BEGIN_MASK_UNSTABLE							\
	GV_BEGIN_ACTIVE_UNSTABLE 

#define GV_END_UNSTABLE								\
	GV_END_MASK_UNSTABLE							\
	GV_END_ACTIVE_UNSTABLE 

#define GV_BEGIN_STABLE								\
	GV_BEGIN_MASK_STABLE							\
	GV_BEGIN_ACTIVE_STABLE 

#define GV_END_STABLE								\
	GV_END_MASK_STABLE								\
	GV_END_ACTIVE_STABLE


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
		using unique_lock_t = std::unique_lock<std::shared_mutex>;
		using shared_lock_t = std::shared_lock<std::shared_mutex>;
		using lock_guard_t  = std::lock_guard<std::shared_mutex>;


		/////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////////////////////
		static constexpr uint8_t  	ACTIVE_BIT 			= 0b00000001;			//the mask that all classes must agree is the active marker
		static constexpr uint8_t 	FREE_BITS 	        = 0b11111110;			//each class may use these themselves
		static constexpr uint64_t 	KEY_MAX_DEPTH		= Mesh3D::MAX_DEPTH;	//the maximum depth that the key type can support

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
		std::atomic<bool>				is_sorted_{false};		//  a flag to track when the active keys are up to date
		std::atomic<bool>				is_collected_{false};	//  a flag to track when the active keys are ready to be sorted (managed by the derived class)
		
		//////////////////////////////////////////////////////////////////////////
		/// Synchronization is primarily just handled by checking if a field
		/// is marked as mutating or stable. Rather than have everything aquire a
		/// mutex, we assume that some responsible caller will perform the pattern:
		///
		///		GV_ASSERT_KEY_MASK_STABLE_STATE
		///		{
		///			auto lock = start_mask_unstable();
		///		
		///			*alter masks but not active keys*
		///
		///			end_mask_unstable();
		///		}
		///
		/// With a similar pattern for ACTIVE_KEYS and stable regions.
		/// If a method is supposed to only ever run when the field is stable,
		/// use the assert and the get_mask_stable methods.
		///
		/// Note that the asserts at the beginning of the responsible calling function
		/// are not strictly essential (because the lock is aquired), but the asserts
		/// in the read/write 'atomic' methods are essential.
		//////////////////////////////////////////////////////////////////////////
		mutable std::atomic<bool>		is_mutating_key_mask{false};	//	track if there are reads/writes to the global mask vector
		mutable std::shared_mutex		key_mask_mutex;						//	mutex for key_mask
		mutable std::atomic<bool>		is_mutating_active_keys{false};	//	track if there are reads/writes to the active key vector
		mutable std::shared_mutex		active_keys_mutex;					//	mutex for active_keys

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

		~HybridKeyTracker() {
			threads.wait_idle();
			lock_guard_t lock1(key_mask_mutex);
			lock_guard_t lock2(active_keys_mutex); 
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
		[[nodiscard]] bool is_sorted() const noexcept 					{ return is_sorted_.load();}
		[[nodiscard]] bool is_collected() const noexcept				{ return is_collected_.load();}

		[[nodiscard]] bool is_current() const noexcept {
			const bool col = is_collected();
			const bool srt = is_sorted();
			if (!col) {
				GUTIL_ERROR("the keys are not collected");
			}
			if (!srt) {
				GUTIL_ERROR("the keys are not sorted");
			}
			return col && srt;
		}

		void mark_stale() noexcept {
			is_sorted_.store(false);
			is_collected_.store(false);
		}

		void clear() noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GV_ASSERT_ACTIVE_KEYS_UNSTABLE_STATE

			std::fill(key_mask.begin(), key_mask.end(), 0);
			active_keys.clear();
			mark_stale();
		}

		void init_key_mask(size_t total_possible_keys) noexcept {
			GV_BEGIN_UNSTABLE

			key_mask.clear();
			key_mask.resize(total_possible_keys,0);
			key_mask.shrink_to_fit();

			active_keys.clear();
			sorter.clear();
			mark_stale();

			GV_END_UNSTABLE
		}

		/////////////////////////////////////////////////////////////////////////
		/// Synchronization methods (mask)
		///
		/// Note that external methods may call these.
		/////////////////////////////////////////////////////////////////////////
		public:
		[[nodiscard]] bool is_key_mask_unstable() const noexcept {
			return is_mutating_key_mask.load();
		}

		[[nodiscard]] bool is_key_mask_stable() const noexcept {
			return !is_mutating_key_mask.load();
		}

		[[nodiscard]] shared_lock_t begin_key_mask_stable() const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			is_mutating_key_mask.store(false);
			return shared_lock_t{key_mask_mutex};
		}

		void end_key_mask_stable() const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
		}
		
		[[nodiscard]] unique_lock_t begin_key_mask_unstable() const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			is_mutating_key_mask.store(true);
			return unique_lock_t{key_mask_mutex};
		}

		void end_key_mask_unstable() const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			is_mutating_key_mask.store(false);
		}


		/////////////////////////////////////////////////////////////////////////
		/// Synchronization methods (active)
		/////////////////////////////////////////////////////////////////////////
		public:
		[[nodiscard]] bool is_active_keys_unstable() const noexcept {
			return is_mutating_active_keys.load();
		}

		[[nodiscard]] bool is_active_keys_stable() const noexcept {
			return !is_mutating_active_keys.load();
		}

		[[nodiscard]] shared_lock_t begin_active_keys_stable() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			GUTIL_ASSERT(is_current());
			is_mutating_active_keys.store(false);
			return shared_lock_t{active_keys_mutex};
		}

		void end_active_keys_stable() const noexcept {
			GUTIL_ASSERT(is_current());
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
		}

		[[nodiscard]] unique_lock_t begin_active_keys_unstable() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			is_mutating_active_keys.store(true);
			return unique_lock_t{active_keys_mutex};
		}

		void end_active_keys_unstable() const noexcept {
			GV_ASSERT_ACTIVE_KEYS_UNSTABLE_STATE
			is_mutating_active_keys.store(false);
		}


		/////////////////////////////////////////////////////////////////////////
		/// Primary 'atomic' methods to interact with the mask
		/// The ACTIVE_BIT is the only specific single bit query, but derived
		///	classes can implement their own. Additionally, each key has it's own
		/// way to get a linear index for it. Each derived class may wish to 
		/// change the query (i.e., get_mask(key.linear_index()))
		///
		/// Note that the mask is mutable and we may wish to expose some number
		/// of its bits to external classes, so this interface is all marked const.
		/////////////////////////////////////////////////////////////////////////
		public:
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t get_mask_no_check(uint64_t idx) const noexcept {
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t get_mask(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t get_mask_stable(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t get_mask_unstable(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t& get_mask_ref_no_check(uint64_t idx) const noexcept {
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t& get_mask_ref(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] uint8_t& get_mask_ref_pseudo_const(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			//note key_mask is mutable, but we want to know when we use this property
			GUTIL_ASSERT(idx<key_mask.size());
			return key_mask[idx];
		}

		[[nodiscard]] std::span<const uint8_t> get_mask_span(uint64_t start, uint64_t end) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			GUTIL_ASSERT(start<end && end<=key_mask.size());
			return std::span<const uint8_t>(key_mask.begin()+start, key_mask.begin()+end);
		}

		[[nodiscard]] std::span<const uint8_t> get_mask_span_stable(uint64_t start, uint64_t end) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			//get a range of masks
			GUTIL_ASSERT(start<end && end<=key_mask.size());
			return std::span<const uint8_t>(key_mask.begin()+start, key_mask.begin()+end);
		}

		[[nodiscard]] std::span<uint8_t> get_mask_span_ref(uint64_t start, uint64_t end) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			//get a range of masks
			GUTIL_ASSERT(start<end && end<=key_mask.size());
			return std::span<uint8_t>(key_mask.begin()+start, key_mask.begin()+end);
		}

		[[nodiscard]] std::span<uint8_t> get_mask_span_ref_pseudo_const(uint64_t start, uint64_t end) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			//get a range of masks
			GUTIL_ASSERT(start<end && end<=key_mask.size());
			return std::span<uint8_t>(key_mask.begin()+start, key_mask.begin()+end);
		}


		//////////////////////////////////////////////////////////////////////////////
		/// Utility methods for interacting with the mask.
		//////////////////////////////////////////////////////////////////////////////
		public:
		GUTIL_DECLARE_SIMD()
		template<uint8_t BIT_MASK> requires (std::popcount(BIT_MASK)==1)
		[[nodiscard]] bool check_bit(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			return get_mask(idx) & BIT_MASK;
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t BIT_MASK> requires (std::popcount(BIT_MASK)==1)
		[[nodiscard]] bool check_bit_no_check(uint64_t idx) const noexcept {
			return get_mask_no_check(idx) & BIT_MASK;
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t BIT_MASK> requires (std::popcount(BIT_MASK)==1)
		[[nodiscard]] bool check_bit_stable(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_STABLE_STATE
			return get_mask_stable(idx) & BIT_MASK;
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t BIT_MASK> requires (std::popcount(BIT_MASK)==1)
		[[nodiscard]] bool check_bit_unstable(uint64_t idx) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			return get_mask_unstable(idx) & BIT_MASK;
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t BIT_MASK> requires (std::popcount(BIT_MASK)==1)
		void set_bit(uint64_t idx, bool val) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			val ? get_mask_ref(idx) |= BIT_MASK  :  get_mask_ref(idx) &= ~BIT_MASK;
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t BIT_MASK> requires (std::popcount(BIT_MASK)==1)
		[[nodiscard]] bool set_bit_check_changed(uint64_t idx, bool val) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			//set the specified bit and return true if it was changed
			uint8_t& byte  = get_mask_ref(idx);
			const bool old = byte & BIT_MASK;
			val ? byte |= BIT_MASK  :  byte &=~BIT_MASK;
			return old!=val;
		}

		public:
		///////////////////////////////////////////////////////////////////////////////
		/// Querying active is someting that anyone should be able to do
		///////////////////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_active(uint64_t idx) const noexcept {
			return check_bit<ACTIVE_BIT>(idx);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_active_no_check(uint64_t idx) const noexcept {
			return check_bit_no_check<ACTIVE_BIT>(idx);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_active_stable(uint64_t idx) const noexcept {
			return check_bit_stable<ACTIVE_BIT>(idx);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool is_active_unstable(uint64_t idx) const noexcept {
			return check_bit_unstable<ACTIVE_BIT>(idx);
		}
		
		GUTIL_DECLARE_SIMD()
		void set_active(uint64_t idx, bool val) noexcept {
			set_bit<ACTIVE_BIT>(idx,val);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] bool set_active_check_changed(uint64_t idx, bool val) noexcept {
			return set_bit_check_changed<ACTIVE_BIT>(idx, val);
		}



		/////////////////////////////////////////////////////////////////////////
		/// A few methods to do bulk bitwise operations on the entire key set.
		/// These can be use for example, to set a reserved bit to mark the start of an bulk
		/// operation.
		/////////////////////////////////////////////////////////////////////////
		void conditional_bitwise_and_all_masks(uint8_t condition_mask, uint8_t mask) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			dispatch_parallel_key_mask_const( 
				[](std::span<uint8_t> mask_span, uint8_t condition_mask, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						if(mask_span[i]&condition_mask) {mask_span[i]&=mask;}
					}
				}, condition_mask, mask);
			threads.wait_idle();
		}

		void conditional_bitwise_or_all_masks(uint8_t condition_mask, uint8_t mask) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			dispatch_parallel_key_mask_const( 
				[](std::span<uint8_t> mask_span, uint8_t condition_mask, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						if(mask_span[i]&condition_mask) {mask_span[i]|=mask;}
					}
				}, condition_mask, mask);
			threads.wait_idle();
		}

		void unconditional_bitwise_and_all_masks(uint8_t mask) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			dispatch_parallel_key_mask_const( 
				[](std::span<uint8_t> mask_span, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						mask_span[i]&=mask;
					}
				}, mask);
			threads.wait_idle();
		}

		void unconditional_bitwise_or_all_masks(uint8_t mask) const noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			dispatch_parallel_key_mask_const( 
				[](std::span<uint8_t> mask_span, uint8_t mask) {
					GUTIL_SIMD()
					for (size_t i=0; i<mask_span.size(); ++i) {
						mask_span[i]|=mask;
					}
				}, mask);
			threads.wait_idle();
		}



		/////////////////////////////////////////////////////////////////////////
		/// A few methods to help with viewing the active list as a particular type
		/////////////////////////////////////////////////////////////////////////
		template<typename Container_A, typename Container_B>
		[[nodiscard]] static bool are_spans_same_data(const Container_A& A, const Container_B& B) noexcept {
			return gutil::containers_are_same_bytes(A,B);
		}

		template<typename KeyTypeOut, typename Container> requires(sizeof(KeyTypeOut)==sizeof(uint64_t) && alignof(KeyTypeOut)==alignof(uint64_t))
		[[nodiscard]] static std::span<KeyTypeOut> reinterpret_key_span(Container& list) noexcept {
			return gutil::reinterpret_as_span<KeyTypeOut, Container>(list);
		}

		template<typename KeyTypeOut, typename Container> requires(sizeof(KeyTypeOut)==sizeof(uint64_t) && alignof(KeyTypeOut)==alignof(uint64_t))
		[[nodiscard]] static std::span<const KeyTypeOut> reinterpret_key_span(const Container& list) noexcept {
			return gutil::reinterpret_as_span<KeyTypeOut, Container>(list);
		}

		template<typename KeyTypeOut, typename KeyTypeIn> requires(sizeof(KeyTypeOut)==sizeof(KeyTypeIn) && alignof(KeyTypeOut)==alignof(KeyTypeIn))
		[[nodiscard]] static std::span<KeyTypeOut> reinterpret_key_span( std::span<KeyTypeIn> list) noexcept {
			return gutil::reinterpret_as_span<KeyTypeOut, KeyTypeIn>(list);
		}

		template<typename KeyTypeOut, typename KeyTypeIn> requires(sizeof(KeyTypeOut)==sizeof(KeyTypeIn) && alignof(KeyTypeOut)==alignof(KeyTypeIn))
		[[nodiscard]] static std::span<const KeyTypeOut> reinterpret_key_span( std::span<const KeyTypeIn> list) noexcept {
			return gutil::reinterpret_as_span<KeyTypeOut, KeyTypeIn>(list);
		}
		
		//allow explicit in/out type statements for better type safety in the generic container path
		template<typename KeyTypeOut, typename KeyTypeIn, typename Container> requires(std::same_as<typename Container::value_type, KeyTypeIn>) 
		[[nodiscard]] static std::span<const KeyTypeOut> reinterpret_key_span(const Container& list) noexcept {
			return gutil::reinterpret_as_span<KeyTypeOut, Container>(list);
		}

		//allow explicit in/out type statements for better type safety
		template<typename KeyTypeOut, typename KeyTypeIn, typename Container> requires(std::same_as<typename Container::value_type, KeyTypeIn>) 
		[[nodiscard]] static std::span<KeyTypeOut> reinterpret_key_span(Container& list) noexcept {
			return gutil::reinterpret_as_span<KeyTypeOut, Container>(list);
		}


		// template<typename Container_A, typename Container_B>
		// [[nodiscard]] static bool are_spans_same_data(const Container_A& A, const Container_B& B) noexcept {
		// 	using A_t = typename Container_A::value_type;
		// 	using B_t = typename Container_B::value_type;
		// 	if constexpr (sizeof(A_t) != sizeof(B_t)) {return false;}
		// 	if constexpr (alignof(A_t) != alignof(B_t)) {return false;}

		// 	if (A.size() != B.size()) {return false;}
		// 	if (reinterpret_cast<uintptr_t>(A.data()) != reinterpret_cast<uintptr_t>(B.data())) {return false;}
		// 	if (reinterpret_cast<uintptr_t>(A.data()+A.size()) != reinterpret_cast<uintptr_t>(B.data()+B.size())) {return false;}
		// 	return true; 
		// }

		// template<typename KeyTypeOut, typename Container_A> 
		// 		requires( 	sizeof(KeyTypeOut) == sizeof(typename Container_A::value_type) &&
		// 					alignof(KeyTypeOut) == alignof(typename Container_A::value_type) &&
		// 					std::contiguous_iterator<typename Container_A::iterator> )
		// [[nodiscard]] static std::span<KeyTypeOut> reinterpret_key_span(Container_A& list) noexcept {
		// 	using KeyTypeIn = typename Container_A::value_type;
		// 	std::span<KeyTypeOut> view = reinterpret_key_span<KeyTypeOut,KeyTypeIn>(std::span<KeyTypeIn>(list));
		// 	GUTIL_ASSERT(are_spans_same_data(view,list));
		// 	return view;
		// }

		// template<typename KeyTypeOut, typename Container_A> 
		// 		requires( 	sizeof(KeyTypeOut) == sizeof(typename Container_A::value_type) &&
		// 					alignof(KeyTypeOut) == alignof(typename Container_A::value_type) &&
		// 					std::contiguous_iterator<typename Container_A::iterator> )
		// [[nodiscard]] static std::span<const KeyTypeOut> reinterpret_key_span(const Container_A& list) noexcept {
		// 	using KeyTypeIn = typename Container_A::value_type;
		// 	std::span<KeyTypeOut> view = reinterpret_key_span<KeyTypeOut,KeyTypeIn>(std::span<const KeyTypeIn>(list));
		// 	GUTIL_ASSERT(are_spans_same_data(view,list));
		// 	return view;
		// }

		// template<typename KeyTypeOut, typename KeyTypeIn>
		// 	requires(sizeof(KeyTypeIn)==8 && sizeof(KeyTypeOut)==8 && alignof(KeyTypeOut)==alignof(KeyTypeIn))
		// [[nodiscard]] static std::span<KeyTypeOut> reinterpret_key_span(std::span<KeyTypeIn> list) noexcept {
		// 	if constexpr (std::same_as<KeyTypeIn, KeyTypeOut>) {return list;}
		// 	else {
		// 		std::span<KeyTypeOut> view {reinterpret_cast<KeyTypeOut*>(list.data()), list.size()};
		// 		#ifndef NDEBUG
		// 			//make sure that the span is correctly interprets the raw values
		// 			GUTIL_ASSERT(are_spans_same_data(view,list));
		// 			for (size_t i=0; i<std::min(size_t{100},view.size()); ++i) {
		// 				uint64_t raw  = static_cast<uint64_t>(list[i]);
		// 				GUTIL_ASSERT(raw == static_cast<uint64_t>(view[i]) && "raw bytes changed");
		// 				GUTIL_ASSERT(KeyTypeIn{raw} == list[i] && "reconstructing in type failed");
		// 				GUTIL_ASSERT(KeyTypeOut{raw} == view[i] && "reconstructing out type failed");
		// 			}
		// 		#endif
		// 		return view;
		// 	}
		// }




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
			
			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			//note that the gutil::ThreadPool with n_threads==0 will have the submitting thread run
			//the job.
			const size_t n_threads 		= threads.n_threads()==0 ? 1 : threads.n_threads();
			const size_t n_keys 		= active_keys.size();
			const size_t dof_per_thread = n_keys/n_threads;

			for (size_t tid=0; tid<n_threads; ++tid) {
				const size_t start = tid * dof_per_thread;
				const size_t end = (tid==n_threads-1) ? n_keys : start + dof_per_thread;

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
			threads.wait_idle();
		}

		template<typename Task, typename...Args>
		void dispatch_parallel_active_keys_const(Task&& action, Args&&... args) const noexcept {
			GUTIL_ASSERT(is_current() && "the active keys are stale");
			GUTIL_ASSERT(active_keys.size()>0 && "HybridKeyTracker - no keys found. Did you forget to collect them?");
			
			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			//note that the gutil::ThreadPool with n_threads==0 will have the submitting thread run
			//the job.
			const size_t n_threads		= threads.n_threads()==0 ? 1 : threads.n_threads();
			const size_t n_keys 		= active_keys.size();
			const size_t dof_per_thread = n_keys/n_threads;

			for (size_t tid=0; tid<n_threads; ++tid) {
				const size_t start = tid * dof_per_thread;
				const size_t end = (tid==n_threads-1) ? n_keys : start + dof_per_thread;

				if constexpr (std::is_invocable_r_v<void, Task, std::span<uint64_t>, int, Args...>) {
					std::span<const uint64_t> list(active_keys.begin()+start, active_keys.begin()+end);
					threads.submit(action, list, tid, std::forward<Args>(args)...);
				}
				else if constexpr (std::is_invocable_r_v<void, Task, std::span<uint64_t>, Args...>) {
					std::span<const uint64_t> list(active_keys.begin()+start, active_keys.begin()+end);
					threads.submit(action, list, std::forward<Args>(args)...);
				}
				else {
					GUTIL_ERROR("the Task signature should be one of: void(span<const uint64_t>, size_t, args...) or void(span<const uint64_t>, args...)");
					GUTIL_ABORT("Arguments did not match what was expected");
				}
			}
			threads.wait_idle();
		}

		template<typename Task, typename...Args>
		void dispatch_parallel_key_mask(Task&& action, Args&&... args) noexcept {
			GUTIL_ASSERT(key_mask.size()>0 && "HybridKeyTracker - key_mask is not initialized");
			
			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			//note that the gutil::ThreadPool with n_threads==0 will have the submitting thread run
			//the job.
			const size_t n_threads = threads.n_threads()==0 ? 1 : threads.n_threads();
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
			threads.wait_idle();
		}

		template<typename Task, typename...Args>
		void dispatch_parallel_key_mask_const(Task&& action, Args&&... args) const noexcept {
			GUTIL_ASSERT(key_mask.size()>0 && "HybridKeyTracker - key_mask is not initialized");
			
			//the first argument of action must be a span of DOFs
			//if the thread number is a required argument, it must be the second argument.
			//note that the gutil::ThreadPool with n_threads==0 will have the submitting thread run
			//the job.
			const size_t n_threads = threads.n_threads()==0 ? 1 : threads.n_threads();
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
				else if constexpr (std::is_invocable_r_v<void, Task, std::span<uint8_t>, int, Args...>) {
					std::span<uint8_t> list(key_mask.begin()+start, key_mask.begin()+end);
					threads.submit(action, list, tid, std::forward<Args>(args)...);
				}
				else if constexpr (std::is_invocable_r_v<void, Task, std::span<uint8_t>, Args...>) {
					std::span<uint8_t> list(key_mask.begin()+start, key_mask.begin()+end);
					threads.submit(action, list, std::forward<Args>(args)...);
				}
				else {
					GUTIL_ERROR("the Task signature should be one of: void(span<const uint8_t>, size_t, args...) or void(span<const uint8_t>, args...)");
					GUTIL_ABORT("Arguments did not match what was expected");
				}
			}
			threads.wait_idle();
		}


		/////////////////////////////////////////////////////////////////////////
		/// A few methods to work with the active keys
		/////////////////////////////////////////////////////////////////////////
		template<typename KeyType>
		void collect_active_keys() noexcept {
			GV_BEGIN_UNSTABLE

			GUTIL_TIMER("Collecting active keys (", key_mask.size(), " to check)");

			//note that the gutil::ThreadPool with n_threads==0 will have the submitting thread run
			//the job.

			const size_t n_threads = threads.n_threads()==0 ? 1 : threads.n_threads();
			const size_t n_keys_per_thread = key_mask.size()/n_threads;
			std::vector<std::vector<uint64_t>> thread_keys(n_threads);
			
			auto job = [n_keys_per_thread, &thread_keys](std::span<const uint8_t> masks, size_t tid) {
				size_t key_index = tid*n_keys_per_thread;	//we need to track the location of the mask we are examing
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

			is_collected_.store(true);

			GV_END_UNSTABLE
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

		template<typename BinFun = decltype(&HybridKeyTracker::default_key_bin), typename Less_t=std::nullptr_t>
		void sort_active_keys(int N=8, BinFun&& fun = &HybridKeyTracker::default_key_bin, Less_t&& less=nullptr) noexcept {
			GV_BEGIN_ACTIVE_UNSTABLE

			GUTIL_ASSERT(is_collected_ && "The keys were not collected. Call collect_active_keys<KeyType>() to collect them.");
			GUTIL_TIMER("sorting ", active_keys.size(), " keys into ", N, " bins");
			sorter = sort_keys(std::span<uint64_t>{active_keys}, N, std::forward<BinFun>(fun), &threads, std::forward<Less_t>(less));
			is_sorted_.store(true);

			GV_END_ACTIVE_UNSTABLE
		}

		template<typename BinFun, typename T, typename Less_t=std::nullptr_t> requires( std::is_invocable_r_v<int, BinFun, T>)
		[[maybe_unused]] gutil::BinSort<T> static sort_keys(std::span<T> list, int N, BinFun&& fun, 
														gutil::ThreadPool* tp=nullptr, Less_t&& less=nullptr) noexcept {
			constexpr bool USER_LESS = std::is_invocable_r_v<bool, Less_t, const T&, const T&>;
			static_assert(USER_LESS || std::same_as<Less_t,std::nullptr_t>);

			//link the current active keys to the sorter
			gutil::BinSort<T> local_sorter{list, N};
			GUTIL_ASSERT(local_sorter.n_bins() == N);
			
			if (tp) {
				local_sorter.dispatch_sort(std::forward<BinFun>(fun), tp);
				tp->wait_idle();

				//sort within bins
				for (int i=0; i<local_sorter.n_bins(); ++i) {
					if constexpr (USER_LESS) {
						tp->submit([less](auto a, auto b){std::sort(a, b, less);}, local_sorter.begin(i), local_sorter.end(i));
					} else {
						tp->submit([](auto a, auto b){std::sort(a, b);}, local_sorter.begin(i), local_sorter.end(i));
					}
				}
				tp->wait_idle();
			}
			else {
				local_sorter.sort(std::forward<BinFun>(fun));
				for (int i=0; i<local_sorter.n_bins(); ++i) {
					if constexpr (USER_LESS) {
						std::sort(local_sorter.begin(i), local_sorter.end(i), less);
					} else {
						std::sort(local_sorter.begin(i), local_sorter.end(i));
					}
				}
			}

			//return the sorter so that the caller has views into each bin
			return local_sorter;
		}

		template<typename BinFun> requires( std::is_invocable_r_v<int, BinFun, uint64_t>)
		[[nodiscard]] size_t lookup_key(const uint64_t key, BinFun&& fun = &HybridKeyTracker::default_key_bin) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			//this gets called many times so we do not aquire a lock
			GUTIL_ASSERT(is_sorted());

			const int bin = fun(key);
			GUTIL_ASSERT(bin < sorter.n_bins());

			std::span<const uint64_t> list = sorter.get_bin(bin);
			auto it = std::lower_bound(list.begin(), list.end(), key);
			return (it==list.end() || *it!=key) ? size_t(-1) : sorter.bin_start(bin) + std::distance(list.begin(), it);
		}

		template<typename BinFun> requires( std::is_invocable_r_v<int, BinFun, uint64_t>)
		std::vector<size_t> lookup_keys_and_sort(std::span<uint64_t> keys, int N, BinFun&& fun) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			//this gets called many times so we do not aquire a lock
			std::vector<size_t> result(keys.size());
			lookup_keys_and_sort(keys, std::span<size_t>{result}, N, std::forward<BinFun>(fun));
			return result;
		}

		template<typename BinFun> requires( std::is_invocable_r_v<int, BinFun, uint64_t>)
		void lookup_keys_and_sort(std::span<uint64_t> keys, std::span<size_t> global_index, int N, BinFun&& fun) const noexcept {
			GV_ASSERT_ACTIVE_KEYS_STABLE_STATE
			//this gets called many times so we do not aquire a lock

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



