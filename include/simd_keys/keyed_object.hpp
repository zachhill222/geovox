#pragma once

#include <cstdint>

namespace GV {
namespace Keys {

	////////////////////////////////////////////////////////////
	/// Base type that various objects can specialize.
	////////////////////////////////////////////////////////////
	template<uint64_t Identifier>
	struct KeyedObject {
		//a tag for determining which mesh feature this corresponds to
		static constexpr uint64_t ID = Identifier;
		uint64_t key;

		//treat this type as a uint64_t with a extra features
		constexpr KeyedObject() noexcept : key{0} {}
		explicit constexpr KeyedObject(uint64_t k) noexcept : key{k} {}
		[[nodiscard]] explicit constexpr operator uint64_t() const noexcept {return key;}
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