#pragma once

#include "octree/digit_key.hpp"

namespace GV
{
	//define core components
	template<typename KEY_T=OctreeDigitKey>
	struct OctreeNodeBase
	{
		//define common aliases
		using key_type 	 = KEY_T;

		//store the key
		KEY_T key{};
		uint8_t cursor{0}; //points to next empty storage, also the number of stored values

		//pass comparisons to the key
		bool constexpr operator==(const OctreeNodeBase other) const {return key == other.key;}
		bool constexpr operator!=(const OctreeNodeBase other) const {return key != other.key;}
		bool constexpr operator<=(const OctreeNodeBase other) const {return key <= other.key;}
		bool constexpr operator>=(const OctreeNodeBase other) const {return key >= other.key;}
		bool constexpr operator<( const OctreeNodeBase other) const {return key <  other.key;}
		bool constexpr operator>( const OctreeNodeBase other) const {return key >  other.key;}

		//simple constructor
		OctreeNodeBase(KEY_T key) : key{key} {}
	};

	//use template specialization for homogeneous vs inhomogeneous data
	template<typename DATA_T, bool SINGLE_DATA=true, uint8_t MAX_DATA=64, bool IS_HOMOGENEOUS=true, typename KEY_T=OctreeDigitKey>
	struct OctreeNode;


	template<typename DATA_T, bool SINGLE_DATA_, uint8_t MAX_DATA_, typename KEY_T>
	struct OctreeNode<DATA_T, SINGLE_DATA_, MAX_DATA_, true, KEY_T> : public OctreeNodeBase<KEY_T>
	{
		//sanity checks
		static_assert(std::equality_comparable<DATA_T>, "OctreeNode - DATA_T must be equality comparable");

		//convenient aliases
		using base_type  = OctreeNodeBase<KEY_T,MAX_DATA_,true,KEY_T>;
		using value_type = DATA_T;
		using key_type   = KEY_T;

		//constructor
		using base_type::base_type;

		//convenient constants
		static constexpr bool SINGLE_DATA    = SINGLE_DATA_;
		static constexpr uint8_t MAX_DATA    = MAX_DATA_;
		static constexpr bool IS_HOMOGENEOUS = true;

		//bring the cursor and key to this class
		using base_type::cursor;
		using base_type::key;

		//add homogeneous storage
		std::array<DATA_T,MAX_DATA> values{};
		
		//check if data is already included
		bool contains(const DATA_T& val) const {
			for (uint8_t i=0; i<cursor; ++i) {if (values[i]==val) {return true;}}
			return false;
		}

		int insert(DATA_T&& val) {
			if (contains(val)) {return 0;}		//data was already there
			if (cursor>=MAX_DATA) {return -1;}	//data could not be added
			values[cursor] = std::move(val);
			++cursor;
			return 1;							//data was successfully added
		}

		int insert(const DATA_T& val) {return insert(std::move(DATA_T{val}));}

		//merge two nodes when the data is known to be unique
		int merge_unique(OctreeNode&& other) {
			if (cursor + other.cursor > MAX_DATA) {return -1;} //cannot merge
			std::move(other.values.begin(), other.values.begin()+other.cursor, values.begin()+cursor);
			cursor += other.cursor;
			//other.cursor=0; //Uncomment if the other node needs to be in a valid state
			return 1;
		}

		//merge two nodes when the data could be duplicated
		int merge(OctreeNode&& other) {
			while (other.cursor>0 && cursor<MAX_DATA) {
				--other.cursor; //point to newest data
				const int flag = insert(std::move(other.values[other.cursor]));
				assert(flag!=-1); //should never happen with the while loop limits
				if (flag==1) {++cursor;}
			}

			if (other.cursor == 0) {return 1;}		//all data was successfully moved
			if (cursor == MAX_DATA) {return -1;}	//some data could not be added
			return 0;								//there was some duplicate data, but the merge was successful
		}

		//take the data from this node and push it to its children
		template<typename VALID_T>
		std::array<OctreeNode,8> split(VALID_T&& is_valid) {
			//initialize children
			std::array<OctreeNode,8> children;
			for (uint64_t c=0; c<8; ++c) {children[c] = OctreeNode{key.child(c)};}

			//move or copy data to children
			for (uint8_t i=0; i<cursor; ++i) {
				#ifndef NDEBUG
				int n_children = 0;
				#endif
				for (int c=0; c<8; ++c) {
					if (is_valid(children[c].key, values[i])) {
						#ifndef NDEBUG
						++n_children;
						#endif

						if constexpr (SINGLE_DATA) {
							children[c].insert(std::move(values[i]));
							break;
						}
						else {children[c].insert(values[i]);}
					}
				}

				#ifndef NDEBUG
				assert(n_children>0);
				#endif
			}

			//mark that the data has been deleted from this node
			cursor = 0;

			//return the children with their data
			return children;
		}
	};
}