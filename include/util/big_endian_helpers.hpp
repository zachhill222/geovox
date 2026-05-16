#pragma once

#include "util/compatibility.hpp"

#include <ranges>
#include <type_traits>

namespace GV
{
	//helper function for writing in big endian
	template<typename F_out=void, typename OS, typename F_in> requires (std::is_scalar_v<F_in>)
	void WRITE_BIG_ENDIAN(OS& buffer, F_in val) {
		using F_t  = std::conditional_t<std::same_as<F_out,void>, F_in, F_out>;
		using U_t  = std::conditional_t<sizeof(F_t)==1, uint8_t,
							std::conditional_t<sizeof(F_t)==2, uint16_t,
							std::conditional_t<sizeof(F_t)==4, uint32_t,
							std::conditional_t<sizeof(F_t)==8, uint64_t, void>>>>;
		static_assert(!std::same_as<U_t,void>, "unknown data size");

		const U_t le = std::bit_cast<U_t>(static_cast<F_t>(val));
		U_t be;

		if constexpr (std::same_as<U_t,uint8_t>)       {be = le;}
		else if constexpr (std::same_as<U_t,uint16_t>) {be = __builtin_bswap16(le);}
		else if constexpr (std::same_as<U_t,uint32_t>) {be = __builtin_bswap32(le);}
		else if constexpr (std::same_as<U_t,uint64_t>) {be = __builtin_bswap64(le);}

		buffer.write(reinterpret_cast<const char*>(&be), sizeof(U_t));
	}

	template<typename F_out=void, typename OS, typename F_in>
	void WRITE_BIG_ENDIAN(OS& buffer, std::span<const F_in> val) {
		for (F_in f : val) {
			if constexpr (std::same_as<F_out,void>) {
				WRITE_BIG_ENDIAN(buffer, f);
			}
			else {
				WRITE_BIG_ENDIAN(buffer, static_cast<F_out>(f));
			}
		}
	}

	template<typename F_out=void, typename OS, typename Container_t> requires (std::ranges::contiguous_range<Container_t>)
	void WRITE_BIG_ENDIAN(OS& buffer, const Container_t& val) {
		WRITE_BIG_ENDIAN<F_out>(buffer, as_span(val));
	}

	template<typename F_out=void, typename OS, int N, typename F_in>
	void WRITE_BIG_ENDIAN(OS& buffer, const Point<N,F_in>& val) {
		WRITE_BIG_ENDIAN<F_out>(buffer, as_span(val));
	}

}