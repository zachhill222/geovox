#pragma once

#include "octree/digit_key.hpp"
#include "octree/node.hpp"

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>

namespace GV
{
	//forward declare so we can specialize DATA_T=void to use unique pointers
	template<typename DATA_T, uint8_t MAX_DATA_=64, typename KEY_T = OctreeDigitKey>
	class Octree;

	template<typename DATA_T, uint8_t MAX_DATA_, typename KEY_T>
	class Octree
	{
	public:
		//define common aliases
		using value_type = DATA_T;
		using key_type   = KEY_T;
		using node_type  = OctreeNode<KEY_T>;
		static_assert(std::is_same_v<key_type, typename node_type::key_type>);

		static constexpr uint8_t MAX_DATA = MAX_DATA_;

		//define common utility operations
		void sort_and_merge();

	private:
		

		//main octree storage
		std::vector<Node> leafs;
	};
}