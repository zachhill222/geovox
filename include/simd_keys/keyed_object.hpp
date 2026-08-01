#pragma once

#include <cstdint>

namespace GV {
namespace Keys {

	////////////////////////////////////////////////////////////
	/// Base type that various objects can specialize.
	////////////////////////////////////////////////////////////
	template<uint64_t Identifier>
	struct KeyedObject {
		static constexpr uint64_t ID = Identifier;
		uint64_t key;
		constexpr explicit KeyedObject(uint64_t k) : key{k} {}
		constexpr operator uint64_t() const {return key;}
	};


	////////////////////////////////////////////////////////////
	/// Unless overridden by the specialization, implement
	/// standard comparisons and hash for using std containers
	////////////////////////////////////////////////////////////
	template<uint64_t Identifier>
	[[nodiscard]] inline constexpr bool operator==(KeyedObject<Identifier> left, 
					KeyedObject<Identifier> right) noexcept {
		return left.key == right.key;
	}

	template<uint64_t Identifier>
	[[nodiscard]] inline constexpr std::strong_ordering operator<=>(KeyedObject<Identifier> left, 
					KeyedObject<Identifier> right) noexcept {
		return left.key <=> right.key;
	}

}}


namespace std {
	template<uint64_t Identifier>
	struct hash<GV::Keys::KeyedObject<Identifier>> {
		[[nodiscard]] size_t operator()(const GV::Keys::KeyedObject<Identifier>& obj) const {
			return hash<uint64_t>{}(obj.key);
		}
	};
}