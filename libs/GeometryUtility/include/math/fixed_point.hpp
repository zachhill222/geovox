#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <bitset>
#include <bit>
#include <cmath>
#include <iostream>

#include "math/float_manipulation.hpp"

namespace gutil
{
	///////////////////////////////////////////////////////////////
	/// This class provides exact arithmetic in fixed point precision.
	/// All numbers are stored as 
	///          mantissa * 2^exponent
	/// where exponent is an integer power known at compile time
	/// and mantissa is an integer that stores the current value.
	/// Similar to floating point numbers,
	///          exponent = bias - fraction_bits
	/// but unlike floating point numbers, this is constant for all
	/// numbers of this type. The fraction_bits are half the size of the
	/// integer (mantissa) type and the bias/offset is 0 by default
	///////////////////////////////////////////////////////////////

	template<std::signed_integral Mantissa_t = int64_t, int OFFSET=0>
	struct FixedPoint
	{
		///////////////////////////////////////////////////////////////
		/// Type defs
		///////////////////////////////////////////////////////////////
		/// allow Scalar concept to see this
		static constexpr bool IS_SCALAR = true;
		static constexpr bool IS_FLOAT = true; //this is a replacement class for floats in some contexts

		/// Underlying integer type
		using mantissa_type = Mantissa_t;

		/// Floating point type with the same number of bits
		using Float_t   = decltype([]() {
			if constexpr (sizeof(Mantissa_t) <= 4) {return float{};}
			else if constexpr (sizeof(Mantissa_t) <= 8) {return double{};}
			else {return (long double){};}
		}());

		/// Intermediate integer type with double the bits for exact multiplications
		using Intermediate_t = decltype([]() {
			if constexpr (sizeof(Mantissa_t) <= 2) {return int32_t{};}
			else if constexpr (sizeof(Mantissa_t) <= 4) {return int64_t{};}
			#ifdef __SIZEOF_INT128__ //should work on most compilers/machines. maybe not windows?
				else {return __int128{};}
			#else
				else {
					static_assert(sizeof(Mantissa_t) <= 8, "FixedPoint: 128-bit arithmetic not available (__SIZEOF_INT128__ not defined)");
					return int64_t{};
				}
			#endif
		}());

		/// Unsigned intermediate type for bit shifting
		using uIntermediate_t = decltype([]() {
			if constexpr (sizeof(Mantissa_t) <= 2) {return uint32_t{};}
			else if constexpr (sizeof(Mantissa_t) <= 4) {return uint64_t{};}
			#ifdef __SIZEOF_INT128__ //should work on most compilers/machines. maybe not windows?
				else {return (unsigned __int128){};}
			#else
				else {
					static_assert(sizeof(Mantissa_t) <= 8, "FixedPoint: 128-bit arithmetic not available (__SIZEOF_INT128__ not defined)");
					return uint64_t{};
				}
			#endif
		}());

		/// Conversion type to help cast to float or double
		using Conversion_t = BitView<sizeof(Mantissa_t)*8>;

		///////////////////////////////////////////////////////////////
		/// Current value and exponent
		///////////////////////////////////////////////////////////////
		Mantissa_t mantissa;
		static constexpr Mantissa_t FRACTION_BITS = Mantissa_t{sizeof(Mantissa_t) * 4}; //half the bits
		static constexpr Mantissa_t EXPONENT      = Mantissa_t{OFFSET} - FRACTION_BITS;

		static constexpr Float_t SCALE = []() {
			//set the exponent
			typename Conversion_t::uint_type bits = static_cast<typename Conversion_t::uint_type>(EXPONENT + Conversion_t::EXPONENT_BIAS);
			//shift into place (implicit leading 1)
			bits <<= Conversion_t::MANTISSA_BITS;
			return std::bit_cast<Float_t>(bits);
		}();

		static constexpr Float_t INV_SCALE = []() {
			//set the exponent
			typename Conversion_t::uint_type bits = static_cast<typename Conversion_t::uint_type>(-EXPONENT + Conversion_t::EXPONENT_BIAS);
			//shift into place (implicit leading 1)
			bits <<= Conversion_t::MANTISSA_BITS;
			return std::bit_cast<Float_t>(bits);
		}();

		static constexpr Float_t EPSILON   = SCALE; //the scale is also the  spacing between consecutive numbers
		static constexpr Float_t MAX_FLOAT = Float_t(static_cast<Float_t>(std::numeric_limits<Mantissa_t>::max()) * SCALE);

		//overflow if a positive intermediate calculation is larger than this
		//intermediate calculations are done in a larger unsigned type for easier bit shifting
		//this bound is the same if the final result is positive or negative
		static constexpr uIntermediate_t MAX_MANTISSA = static_cast<uIntermediate_t>(std::numeric_limits<Mantissa_t>::max());

		///////////////////////////////////////////////////////////////
		/// Constructors
		///////////////////////////////////////////////////////////////
		constexpr FixedPoint() noexcept : mantissa(0) {}
		constexpr FixedPoint(const std::floating_point auto x) noexcept : mantissa(static_cast<Mantissa_t>(std::llround(static_cast<Float_t>(x) * INV_SCALE))) {assert(std::fabs(x) <= MAX_FLOAT);} //scale and round
		constexpr FixedPoint(const std::integral auto x)       noexcept : FixedPoint(static_cast<Float_t>(x)) {}
		constexpr explicit FixedPoint(const Mantissa_t x, int) noexcept : mantissa(x) {} //tagged constructor to explicitly set the mantissa

		///////////////////////////////////////////////////////////////
		/// Conversions
		///////////////////////////////////////////////////////////////

		// Convert to float implicitly to allow this type as a drop-in replacement into existing code
		// Care should be taken where it is important that operations should be done in FixedPoint precision
		// instead of converting double, performing the operation, and then converting back.
		// e.g., FixedPoint a =  std::sqrt(FixedPoint b) will cast b to a float, compute the square root, and the cast back.
		constexpr operator Float_t() const noexcept
		{
			if (mantissa==0) {return Float_t(0);}

			Conversion_t conv;

			//set sign
			const bool negative = mantissa<0;
			conv.set_sign(negative);

			//get exponent after shifting decimal place to 1.float_mantissa * 2^float_exponent
			typename Conversion_t::uint_type abs_mantissa = negative ? -mantissa : mantissa;
			const int leading_bit_pos = Conversion_t::BITS - 1 - std::countl_zero(abs_mantissa);
			const typename Conversion_t::int_type float_exponent = EXPONENT + leading_bit_pos;
			conv.set_exponent_from_power_of_2(float_exponent);

			//get mantissa bits (the leading 1 is ignored when using conv.set_mantissa)
			if (leading_bit_pos >= Conversion_t::MANTISSA_BITS)
			{
				//shift right
				conv.set_mantissa(abs_mantissa >> (leading_bit_pos - Conversion_t::MANTISSA_BITS));
			}
			else 
			{
				//shift left
				conv.set_mantissa(abs_mantissa << (Conversion_t::MANTISSA_BITS - leading_bit_pos));
			}
			
			return static_cast<Float_t>(conv);
		}

		//sometimes this is more convenient. especially if the implicit cast is disabled during debugging.
		constexpr Float_t as_float() const {return static_cast<Float_t>(*this);}


		///////////////////////////////////////////////////////////////
		/// Arithmetic
		///////////////////////////////////////////////////////////////
		constexpr FixedPoint  operator+( const FixedPoint &other) const noexcept {return FixedPoint(mantissa + other.mantissa, 0);}
		constexpr FixedPoint& operator+=(const FixedPoint &other) noexcept {mantissa += other.mantissa; return *this;}
		constexpr FixedPoint  operator-( const FixedPoint &other) const noexcept {return FixedPoint(mantissa - other.mantissa, 0);}
		constexpr FixedPoint& operator-=(const FixedPoint &other) noexcept {mantissa -= other.mantissa; return *this;}
		
		constexpr FixedPoint  operator*(const FixedPoint &other) const noexcept
		{
			// (m1 * 2^EXP) * (m2 * 2^EXP) = m3 * 2^EXP where m3 = m1*m2*2^EXP
			// we can set m3 = (m1*m2 << EXP) if EXP>0.
			
			// Determine signs
			bool num_negative = mantissa < 0;
			bool den_negative = other.mantissa < 0;
			bool result_negative = num_negative != den_negative;

			// Work with absolute values
			uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = unum * uden;
			if constexpr (EXPONENT > 0) {uval <<= EXPONENT;}
			else {uval >>= -EXPONENT;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			// Get final result
			Mantissa_t m_val = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return FixedPoint(m_val, 0);
		}

		constexpr FixedPoint& operator*=(const FixedPoint &other) noexcept
		{
			// Determine signs
			bool num_negative = mantissa < 0;
			bool den_negative = other.mantissa < 0;
			bool result_negative = num_negative != den_negative;

			// Work with absolute values
			uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = unum * uden;
			if constexpr (EXPONENT > 0) {uval <<= EXPONENT;}
			else {uval >>= -EXPONENT;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			// Get final result
			mantissa = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return *this;
		}

		constexpr FixedPoint operator/(const FixedPoint &other) const noexcept
		{
			// (m1 / 2^EXP) * (m2 / 2^EXP) = m3 * 2^EXP where m3 = (m1 / m2) * 2^-EXP = ((m1 * 2^M) / m2 )* 2^(-M-EXP) 
			// where we choose M to take advantage of the full intermediate precision
			constexpr int M  = 4*sizeof(Intermediate_t);
			constexpr int final_shift = M + EXPONENT; //negative of what is in the comment
			static_assert(M == 8*sizeof(Mantissa_t));

			// Determine signs
			const bool num_negative    = mantissa < 0;
			const bool den_negative    = other.mantissa < 0;
			const bool result_negative = num_negative != den_negative;

			// Work with absolute values
			const uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			const uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = (unum << M) / uden;
			if constexpr (final_shift > 0) {uval >>= final_shift;}
			else {uval <<= -final_shift;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			//convert back to signed integer and convert to mantissa
			Mantissa_t m_val = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return FixedPoint(m_val, 0);
		}

		constexpr FixedPoint& operator/=(const FixedPoint &other) noexcept
		{
			constexpr int M  = 4*sizeof(Intermediate_t);
			constexpr int final_shift = M + EXPONENT; //negative of what is in the comment
			static_assert(M == 8*sizeof(Mantissa_t));

			// Determine signs
			const bool num_negative    = mantissa < 0;
			const bool den_negative    = other.mantissa < 0;
			const bool result_negative = num_negative != den_negative;

			// Work with absolute values
			const uIntermediate_t unum = num_negative ? static_cast<uIntermediate_t>(-mantissa) : static_cast<uIntermediate_t>(mantissa);
			const uIntermediate_t uden = den_negative ? static_cast<uIntermediate_t>(-other.mantissa) : static_cast<uIntermediate_t>(other.mantissa);

			// Perform division with unsigned values and apply shift
			uIntermediate_t uval = (unum << M) / uden;
			if constexpr (final_shift > 0) {uval >>= final_shift;}
			else {uval <<= -final_shift;}

			//check for overflow
			assert(uval<MAX_MANTISSA);

			//convert back to signed integer and convert to mantissa
			mantissa = result_negative ? -static_cast<Mantissa_t>(uval) : static_cast<Mantissa_t>(uval);
			return *this;
		}

		///////////////////////////////////////////////////////////////
		/// Extra Math
		///////////////////////////////////////////////////////////////
		constexpr FixedPoint  operator-() const noexcept {return FixedPoint(-mantissa,0);}
		constexpr FixedPoint  operator+( const std::floating_point auto &other) const noexcept {return (*this)+ FixedPoint(other);}
		constexpr FixedPoint& operator+=(const std::floating_point auto &other) noexcept {return (*this)+=FixedPoint(other);}
		constexpr FixedPoint  operator-( const std::floating_point auto &other) const noexcept {return (*this)- FixedPoint(other);}
		constexpr FixedPoint& operator-=(const std::floating_point auto &other) noexcept {return (*this)-=FixedPoint(other);}
		constexpr FixedPoint  operator*( const std::floating_point auto &other) const noexcept {return (*this)* FixedPoint(other);}
		constexpr FixedPoint& operator*=(const std::floating_point auto &other) noexcept {return (*this)*=FixedPoint(other);}
		constexpr FixedPoint  operator/( const std::floating_point auto &other) const noexcept {return (*this)/ FixedPoint(other);}
		constexpr FixedPoint& operator/=(const std::floating_point auto &other) noexcept {return (*this)/=FixedPoint(other);}


		///////////////////////////////////////////////////////////////
		/// FixedPoint Comparisons
		///////////////////////////////////////////////////////////////
		constexpr bool operator<(const FixedPoint &other)  const noexcept {return mantissa  < other.mantissa;}
		constexpr bool operator<=(const FixedPoint &other) const noexcept {return mantissa <= other.mantissa;}
		constexpr bool operator>(const FixedPoint &other)  const noexcept {return mantissa  > other.mantissa;}
		constexpr bool operator>=(const FixedPoint &other) const noexcept {return mantissa >= other.mantissa;}

		constexpr bool operator==(const FixedPoint &other) const noexcept {return mantissa == other.mantissa;}
		constexpr bool operator!=(const FixedPoint &other) const noexcept {return mantissa != other.mantissa;}

		///////////////////////////////////////////////////////////////
		/// Float Comparisons
		///////////////////////////////////////////////////////////////
		constexpr bool operator<(const std::floating_point auto &other)  const noexcept {return *this  < FixedPoint(static_cast<Float_t>(other));}
		constexpr bool operator<=(const std::floating_point auto &other) const noexcept {return *this <= FixedPoint(static_cast<Float_t>(other));}
		constexpr bool operator>(const std::floating_point auto &other)  const noexcept {return *this  > FixedPoint(static_cast<Float_t>(other));}
		constexpr bool operator>=(const std::floating_point auto &other) const noexcept {return *this >= FixedPoint(static_cast<Float_t>(other));}

		constexpr bool operator==(const std::floating_point auto &other) const noexcept {return *this == FixedPoint(static_cast<Float_t>(other));}
		constexpr bool operator!=(const std::floating_point auto &other) const noexcept {return *this != FixedPoint(static_cast<Float_t>(other));}


		// get bits as a string for debugging
		std::string to_string() const noexcept
		{
			std::string raw = std::bitset<Conversion_t::BITS>(std::abs(mantissa)).to_string();
			std::string result;
			result += std::to_string(mantissa<0); //get correct sign bit
			result += "|" + raw.substr(1, Conversion_t::BITS - 1);
			result += " * 2^" + std::to_string(EXPONENT);
			return result;
		}
	};


	///////////////////////////////////////////////////////////////
	/// Utility functions
	///////////////////////////////////////////////////////////////
	template<std::signed_integral Mantissa_t, int OFFSET>
	std::ostream& operator<<(std::ostream& os, const FixedPoint<Mantissa_t,OFFSET> val)
	{
		//print as a float is usually best
		//explicitly cast in case the implicit cast is disabled during debugging
		os << static_cast<FixedPoint<Mantissa_t,OFFSET>::Float_t>(val);
		return os;
	}
}