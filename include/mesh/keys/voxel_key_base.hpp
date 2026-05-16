#pragma once

#include <cstdint>
#include <cassert>
#include <string>
#include <bit>

#include <bitset>
#include <iostream>
#include <iomanip>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//these classes store the logic of a (limited to 64-bit storage) an infinite hierarchical voxel mesh
	//this includes parent/child/adjacency relationships and is of course limited to the topology,
	//but converting normalized coordinates (we use [0,1] for more convenient index arithmetic) to
	//some geometric coordinate is trivial.
	//additionally, we leave the least significant several bits accessible to
	//other classes that can be freely changed without interfering with the logic.
	//the underlying data is stored in a uint64_t and can be accessed if needed (it probably isn't)
	//the operators &, &=, |, |=, <<, <<=, >>, >>=, ^, ^= have been overloaded to work only on the free bits
	//so that you can safely use them to compactly store extra data (e.g., a uint8_t index plus a flag)
	//these operations are masked to guarantee that the essential data is not affected
	//the operator[] has been overloaded to access any single free bit
	//the operator()(i,j) has been overloaded to access the free bits [i,j) cast to a uint64_t.
	
	//ON_W: other width - used by inherited classes, not touched by this or outside classes. NOT USED for comparison.
	//OC_W: other width - used by inherited classes, not touched by this or outside classes. IS USED for comparisoin.
	//I_W : index width - used by this and inherited classes, not touched by outside classes
	template<int ON_W_, int OC_W_, int I_W_> requires (ON_W_ + OC_W_ + 3*I_W_ < 64)
	struct VoxelKey
	{
		//compute the required number of bits to represent the number of vertices with the
		//given index width.
		//maximum depth that can be represented. the maximum valid depth may be smaller.
		static constexpr uint64_t MAX_INDEX = (uint64_t{1} << I_W_) -1;
		static constexpr uint64_t MAX_DEPTH = I_W_-1; //maximum depth that is compatible with the index
		static constexpr uint64_t D_W       = std::bit_width(MAX_DEPTH); //number of bits required to represent the max depth
		static constexpr uint64_t ON_W      = ON_W_;
		static constexpr uint64_t OC_W      = OC_W_;
		static constexpr uint64_t C_W       = 3; //color width

		//absorb any extra bits into the free bits
		static constexpr uint64_t F_W = 64 - (ON_W + OC_W + D_W + 3*I_W_);
		
		//width of remainder (non-color) index fields
		static constexpr uint64_t I_W = I_W_;
		static constexpr uint64_t J_W = I_W_; //index j width
		static constexpr uint64_t K_W = I_W_; //index k width

		//define offsets (data start)
		static constexpr uint64_t I_S  = 0;				//index i start
		static constexpr uint64_t J_S  = I_S  + I_W;	//index j start
		static constexpr uint64_t K_S  = J_S  + J_W;	//index k start
		static constexpr uint64_t OC_S = K_S  + K_W; 	//other (compare) start (e.g. axis normal)
		static constexpr uint64_t D_S  = OC_S + OC_W;	//depth start
		static constexpr uint64_t ON_S = D_S  + D_W;	//other (no-compare) start (e.g. periodic conditions)
		static constexpr uint64_t F_S  = ON_S + ON_W;	//free bit start
		static_assert(F_S+F_W == 64, "VoxelKey: incorrect field starts");

		//define masks
		static constexpr uint64_t I_M  = ((uint64_t{1} << I_W)  - 1) << I_S;  //index i mask (non-color bits)
		static constexpr uint64_t J_M  = ((uint64_t{1} << J_W)  - 1) << J_S;  //index j mask (non-color bits)
		static constexpr uint64_t K_M  = ((uint64_t{1} << K_W)  - 1) << K_S;  //index k mask (non-color bits)
		static constexpr uint64_t OC_M = ((uint64_t{1} << OC_W) - 1) << OC_S; //other (compare) mask
		static constexpr uint64_t D_M  = ((uint64_t{1} << D_W)  - 1) << D_S;  //depth mask
		static constexpr uint64_t ON_M = ((uint64_t{1} << ON_W) - 1) << ON_S; //other (no-compare) mask
		static constexpr uint64_t F_M  = ((uint64_t{1} << F_W)  - 1) << F_S;  //free mask

		//catch off-by-one shifting/width errors. each mask must be disjoint cover all 64 bits
		static_assert( (F_M | ON_M | OC_M | D_M | I_M | J_M | K_M) == (uint64_t) -1, "VoxelKey: incorrect field masks");
		static_assert( (F_M ^ ON_M ^ OC_M ^ D_M ^ I_M ^ J_M ^ K_M) == (uint64_t) -1, "VoxelKey: incorrect field masks");

		//convenient key to return
		static constexpr uint64_t DOES_NOT_EXIST = uint64_t(-1);

		//other useful masks
		static constexpr uint64_t COMPARE_MASK = OC_M | D_M | I_M | J_M | K_M;
		static constexpr uint64_t INDEX_MASK = I_M | J_M | K_M;
		static constexpr uint64_t DEPTH_INDEX_MASK = D_M | INDEX_MASK;

		//store the bits
		uint64_t _data_;

		//define constructors
		explicit constexpr VoxelKey() : _data_{DOES_NOT_EXIST} {}
		explicit constexpr VoxelKey(const uint64_t data) : _data_{data} {}
		constexpr VoxelKey( const uint64_t ii,
							const uint64_t jj,
							const uint64_t kk,
							const uint64_t oc, 
							const uint64_t dd, 
							const uint64_t on,
							const uint64_t ff) :
			_data_{
				//shift each field into place and mask
				((ff<<F_S)  & F_M)  |
				((on<<ON_S) & ON_M) |
				((oc<<OC_S) & OC_M) |
				((dd<<D_S)  & D_M)  |
				((ii<<I_S)  & I_M)  |
				((jj<<J_S)  & J_M)  |
				((kk<<K_S)  & K_M)
			}
		{}

		//access each field
		#pragma omp declare simd
		inline constexpr uint64_t free()    const {return (_data_&F_M) >>F_S;}
		#pragma omp declare simd
		inline constexpr uint64_t other_n() const {return (_data_&ON_M)>>ON_S;}
		#pragma omp declare simd
		inline constexpr uint64_t other_c() const {return (_data_&OC_M)>>OC_S;}
		#pragma omp declare simd
		inline constexpr uint64_t depth()   const {return (_data_&D_M) >>D_S;}
		#pragma omp declare simd
		inline constexpr uint64_t i() 	    const {return (_data_&I_M) >> I_S;}
		#pragma omp declare simd
		inline constexpr uint64_t j() 	    const {return (_data_&J_M) >> J_S;}
		#pragma omp declare simd
		inline constexpr uint64_t k() 	    const {return (_data_&K_M) >> K_S;}

		//access the un-shifted essential/free bits (good for comparison below)
		#pragma omp declare simd
		inline constexpr uint64_t essential_bits() const {return _data_&~F_M;}
		#pragma omp declare simd
		inline constexpr uint64_t free_bits() const {return _data_&F_M;}
		#pragma omp declare simd
		inline constexpr uint64_t compare_bits() const {return _data_&COMPARE_MASK;}

		//access i,j,k as axis 0, 1, 2 with a mod fail safe (axis -1 is axis 2, but more work)
		inline constexpr uint64_t index(const int a) const {return a==0 ? i() : a==1 ? j() : a==2 ? k() : index(a%3);}

		//set bit fields
		#pragma omp declare simd
		inline constexpr void set_free(const uint64_t bt) {
			_data_ &= ~F_M;
			_data_ |= ((bt<<F_S)&F_M);
		}
		#pragma omp declare simd
		inline constexpr void set_on(const uint64_t bt) {
			_data_ &= ~ON_M;
			_data_ |= ((bt<<ON_S)&ON_M);
		}
		#pragma omp declare simd
		inline constexpr void set_oc(const uint64_t bt) {
			_data_ &= ~OC_M;
			_data_ |= ((bt<<OC_S)&OC_M);
		}
		#pragma omp declare simd
		inline constexpr void set_depth(const uint64_t bt) {
			_data_ &= ~D_M;
			_data_ |= ((bt<<D_S)&D_M);
		}
		#pragma omp declare simd
		inline constexpr void set_i(uint64_t bt) {
			_data_ &= ~I_M; //clear i bits (all)
			_data_ |= ((bt<<I_S)&I_M);
		}
		#pragma omp declare simd
		inline constexpr void set_j(uint64_t bt) {
			_data_ &= ~J_M; //clear j bits (all)
			_data_ |= ((bt<<J_S)&J_M);
		}
		#pragma omp declare simd
		inline constexpr void set_k(uint64_t bt) {
			_data_ &= ~K_M; //clear k bits (all)
			_data_ |= ((bt<<K_S)&K_M);
		}


		//check if the key was set to DOES_NOT_EXIST
		#pragma omp declare simd
		inline constexpr bool exists() const {return _data_!=DOES_NOT_EXIST;}

		//in-place bit operations
		#pragma omp declare simd
		VoxelKey& operator<<=(const uint64_t other) {
			assert(other<F_W);
			const uint64_t fb = (_data_&F_M) << other;
			_data_ &= ~F_M;
			_data_ |= fb&F_M;
			return *this;
		}

		#pragma omp declare simd
		VoxelKey& operator>>=(const uint64_t other) {
			assert(other<F_W);
			const uint64_t fb = (_data_&F_M) >> other;
			_data_ &= ~F_M;
			_data_ |= (fb&F_M);
			return *this;
		}

		#pragma omp declare simd
		VoxelKey& operator^=(const uint64_t other) {
			_data_ ^= ((other<<F_S)&F_M); //include the shift in case the fields are re-arranged
			return *this;
		}

		#pragma omp declare simd
		VoxelKey& operator&=(const uint64_t other) {
			const uint64_t o = (other<<F_S)&F_M;
			_data_ &=  (o|~F_M); //include the shift in case the fields are re-arranged
			return *this;
		}

		#pragma omp declare simd
		VoxelKey& operator|=(const uint64_t other) {
			_data_ |= ((other<<F_S)&F_M); //include the shift in case the fields are re-arranged
			return *this;
		}

		//non-in-place bit operations (call the in-place version for consistency)
		#pragma omp declare simd
		VoxelKey operator<<(const uint64_t other) const {
			assert(other<F_W);
			VoxelKey result{_data_};
			result <<= other;
			return result;
		}

		#pragma omp declare simd
		VoxelKey operator>>(const uint64_t other) const {
			assert(other<F_W);
			VoxelKey result{_data_};
			result >>= other;
			return result;
		}

		#pragma omp declare simd
		VoxelKey operator^(const uint64_t other) const {
			VoxelKey result{_data_};
			result ^= other;
			return result;
		}

		#pragma omp declare simd
		VoxelKey operator&(const uint64_t other) const {
			VoxelKey result{_data_};
			result &= other;
			return result;
		}

		#pragma omp declare simd
		VoxelKey operator|(const uint64_t other) const {
			VoxelKey result{_data_};
			result |= other;
			return result;
		}

		//access bits
		#pragma omp declare simd
		constexpr bool operator[](const uint64_t idx) const {
			assert(idx<F_W);
			const uint64_t mask = uint64_t{1} << (idx+F_S);
			return static_cast<bool>(_data_ & mask & F_M);
		}

		#pragma omp declare simd
		constexpr uint64_t operator()(const uint64_t start, const uint64_t end) const {
			assert(start<=end);
			assert(end<=F_W);
			const uint64_t mask = ((uint64_t{1} << (end-start)) -1) << (start + F_S);
			return (_data_ & mask & F_M) >> (start + F_S);
		}

		//allow explicit conversion to uint64_t (although _data_ is directly accessible)
		explicit constexpr operator uint64_t() const {return _data_;}

		//comparison operators do not depend on the free bits
		inline constexpr bool operator==(VoxelKey other) const {return (_data_&COMPARE_MASK) == (other._data_&COMPARE_MASK);}
		inline constexpr bool operator!=(VoxelKey other) const {return (_data_&COMPARE_MASK) != (other._data_&COMPARE_MASK);}
		inline constexpr bool operator<(VoxelKey other)  const {return (_data_&COMPARE_MASK) <  (other._data_&COMPARE_MASK);}
		inline constexpr bool operator>(VoxelKey other)  const {return (_data_&COMPARE_MASK) >  (other._data_&COMPARE_MASK);}

		//hash function to respect comparisons
		struct Hash {
			template<typename T> //template to work correctly with derived types
			inline size_t operator()(const T& k) const {return std::hash<uint64_t>{}(k.compare_bits());}
		};
	};


	template<int ON_W, int OC_W, int I_W> requires (ON_W + OC_W + 3*I_W < 64)
	std::ostream& operator<<(std::ostream& os, const VoxelKey<ON_W, OC_W, I_W> k) {
	    using K = VoxelKey<ON_W, OC_W, I_W>;
	    std::bitset<64> bits(k._data_);

	    // print field labels MSB->LSB, skipping zero-width fields
	    os << std::setw(K::F_W) << "F" << " ";
	    if constexpr (K::ON_W > 0) os << std::setw(K::ON_W) << "ON" << " ";
	    os << std::setw(K::D_W) << "D" << " ";
	    if constexpr (K::OC_W > 0) os << std::setw(K::OC_W) << "OC" << " ";
	    os << std::setw(K::K_W) << "K" << " "
	       << std::setw(K::J_W) << "J" << " "
	       << std::setw(K::I_W) << "I" << "\n";

	    // print bits MSB->LSB with separators at non-degenerate field boundaries
	    for (int b=63; b>=0; --b) {
	        os << bits.test(b);
	        const int bs = static_cast<int>(b);
	        bool sep = bs == static_cast<int>(K::F_S)
	                || bs == static_cast<int>(K::D_S)
	                || bs == static_cast<int>(K::K_S)
	                || bs == static_cast<int>(K::J_S)
	                || bs == static_cast<int>(K::I_S);
	        if constexpr (K::ON_W > 0) sep = sep || bs == static_cast<int>(K::ON_S);
	        if constexpr (K::OC_W > 0) sep = sep || bs == static_cast<int>(K::OC_S);
	        if (sep) os << " ";
	    }

	    // print decoded fields
	    os << "\n"
	       << "\nraw:     " << k._data_
	       << "\nfree:    " << k.free()
	       << "\ndepth:   " << k.depth()
	       << "\ni:       " << k.i()
	       << "\nj:       " << k.j()
	       << "\nk:       " << k.k();
	    if constexpr (K::ON_W > 0) os << "\nother_n: " << k.other_n();
	    if constexpr (K::OC_W > 0) os << "\nother_c: " << k.other_c();
	    os << "\n";

	    return os;
	}

}

//inject a hash into the std namespace for any types that derive from this
namespace std
{
	template<typename T> requires std::derived_from<T, GV::VoxelKey<T::ON_W, T::OC_W, T::I_W>>
	struct hash<T>
	{
		inline size_t operator()(const T key) const {return std::hash<uint64_t>{}(key.compare_bits());}
	};
}