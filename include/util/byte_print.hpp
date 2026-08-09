#pragma once

#include <bit>
#include <bitset>
#include <string>
#include <cstdint>

namespace GV {

	inline std::string print_bytes(uint8_t bytes) noexcept {
		//print with LSB on the right
		std::string str = "";
		for (int i=7; i>=0; --i) {
			str += (bytes & (1<<i)) ? "1" : "0";
		}
		return str;
	}

	template<typename T>
	inline std::string print_bytes(const T& obj) noexcept {
		constexpr int n_bytes = sizeof(T);
		using Bytes_t = std::array<uint8_t, n_bytes>;

		std::string str = "";
		Bytes_t bytes = std::bit_cast<Bytes_t>(obj);

		//print with LSB on the right
		for (int i=n_bytes-1; i>=0; --i) {
			str += print_bytes(bytes[i]) + " ";
		}

		return str;
	}

	/////////////////////////////////////////////
	/// Format bytes into B, KB, etc.
	/////////////////////////////////////////////
	[[nodiscard]] inline std::string format_byte_count(size_t bytes) noexcept {
		constexpr double KB = 1024.0, MB = KB*1024.0, GB = MB*1024.0;
		char buf[64];
		if 		(bytes >= (size_t)(GB)) { std::snprintf(buf, sizeof(buf), "%.2f GB", bytes/GB);}
		else if (bytes >= (size_t)(MB)) { std::snprintf(buf, sizeof(buf), "%.2f MB", bytes/MB);}
		else if (bytes >= (size_t)(KB)) { std::snprintf(buf, sizeof(buf), "%.2f KB", bytes/KB);}
		else  					   	    { std::snprintf(buf, sizeof(buf), "%zu B", bytes);}
		return buf;
	}
}


