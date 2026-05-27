#pragma once

#include <cstdint>
#include <bit>
#include <iostream>

namespace GV
{
	//this key stores octree nodes as a 21 digit base-8 number stored in a uint64_t value. The last bit can be treated as a sentinel/root
	//and is always 1 on a valid node. Thus any odd uint64_t number is a valid key
	//The bit layout is as follows:
	//	(3-bit depth 21) (3-bit depth 20) ... (3-bit depth 1) (1-bit root/depth 0)
	//note that if the bit-width is W (minimum of 1), then we are at depth (W-1)/3
	struct OctreeDigitKey
	{
		//storage
		uint64_t _data_{0};

		//constants
		static constexpr uint64_t ROOT = 1; //also a mask for the root bit
		static constexpr uint64_t MAX_DEPTH = 21;
		
		//bit masks and shifts
		static constexpr uint64_t D_S(uint64_t dd) {assert(dd<=MAX_DEPTH && dd>0); return 1 + 3*(dd-1);}			//depth start
		static constexpr uint64_t D_M(uint64_t dd) {assert(dd<=MAX_DEPTH && dd>0); return uint64_t{7} << D_S(dd);}  //depth mask

		//simple queries
		inline constexpr uint64_t depth() const {return (std::bit_width(_data_)-1)/3;}
		inline constexpr bool is_valid()  const {return _data_ & ROOT;}
		inline constexpr bool is_root()	  const {return _data_ == ROOT;} 

		//get the digit (child number) for depth dd
		inline constexpr uint64_t digit(const uint64_t dd) const {return (_data_ & D_M(dd)) >> D_S(dd);}


		//comparisons
		bool constexpr operator==(const OctreeDigitKey other) const {return _data_ == other._data_;}
		bool constexpr operator!=(const OctreeDigitKey other) const {return _data_ != other._data_;}
		bool constexpr operator<=(const OctreeDigitKey other) const {return _data_ <= other._data_;}
		bool constexpr operator>=(const OctreeDigitKey other) const {return _data_ >= other._data_;}
		bool constexpr operator<( const OctreeDigitKey other) const {return _data_ <  other._data_;}
		bool constexpr operator>( const OctreeDigitKey other) const {return _data_ >  other._data_;}

		//trivial constructors
		OctreeDigitKey() : _data_{ROOT} {}
		explicit OctreeDigitKey(const uint64_t data) : _data_{data} {}

		//copy and move operations
		OctreeDigitKey(const OctreeDigitKey& other) : _data_{other._data_} {}
		OctreeDigitKey(OctreeDigitKey&& other) : _data_{other._data_} {}
		OctreeDigitKey& operator=(const OctreeDigitKey& other) {_data_=other._data_; return *this;}
		OctreeDigitKey& operator=(OctreeDigitKey&& other) {_data_=other._data_; return *this;}

		//hierarchy relations
		inline constexpr OctreeDigitKey parent() const {assert(depth()>0); return OctreeDigitKey{_data_ & ~D_M(depth())};}
		inline constexpr OctreeDigitKey child(const uint64_t c) const {assert(c<8); return OctreeDigitKey{_data_ & (c << D_S(depth()))};}
	};


	std::ostream& operator<<(std::ostream& os, const OctreeDigitKey key) {
		//represent the base-8 number
		//note we need to flip the digits
		
	}

}

//inject the key into the std hash
template<>
struct std::hash<GV::OctreeDigitKey>
{
	size_t operator()(const GV::OctreeDigitKey& k) const noexcept {return std::hash<uint64_t>(k._data_);}
};