#pragma once

#include <array>
#include <algorithm>

namespace gutil
{
	///////////////////////////////////////////////////////////////
	/// This class provides a "linear" set. This is for sets of elements that support
	/// linear searching is the best option.
	///
	/// Addition and subtraction act as set union and set subtraction.
	///////////////////////////////////////////////////////////////

	template<typename Data_t, int MAX_SIZE=64>
	class LinearSet
	{
	private:
		static_assert(MAX_SIZE>0);

		int cursor_ = 0; //location of next insertion point
		std::array<Data_t,MAX_SIZE> data_; //storage

		template<int L, int R>
		friend LinearSet<Data_t,L+R> set_union(const LinearSet<Data_t,L>&, const LinearSet<Data_t,R>&);
	public:
		bool contains(const Data_t& val) const //check if the element is contained in the set
		{
			for (int i=0; i<cursor_; ++i) {
				if (data_[i] == val) {return true;}
			}
			return false;
		}

		//TODO: add iterator versions of insert
		void insert(const Data_t& val) //insert by copy
		{
			if (!this->contains(val)) {
				if (cursor_ < MAX_SIZE) {
					data_[cursor_++] = val;
				}
				else {
					throw std::runtime_error("LinearSet : Ran out of space.");
				}
			}
		}

		void insert(Data_t&& val) //insert by move
		{
			if (!this->contains(val)) {
				if (cursor_ < MAX_SIZE) {
					data_[cursor_++] = std::move(val);
				}
				else {
					throw std::runtime_error("LinearSet : Ran out of space.");
				}
			}
		}

		void remove(const Data_t& val) //remove a data value if it exists
		{
			for (int i=0; i<cursor_; ++i) {
				if (data_[i] == val) {
					data_[i] = std::move(data_[--cursor_]);
					break;
				}
			}
		}

		//standard container interface
		size_t size() const {return static_cast<size_t>(cursor_);}
		bool empty() const {return cursor_ == 0;}
		void clear() {cursor_=0;}

		//iterator access
		inline auto begin() const {return data_.cbegin();}
		inline auto begin() {return data_.begin();}
		inline auto end() const {return data_.cbegin()+cursor_;}
		inline auto end() {return data_.begin()+cursor_;}

		//unions
		template<int MAX_SIZE_OTHER>
		LinearSet<Data_t,MAX_SIZE>& operator+=(const LinearSet<Data_t,MAX_SIZE_OTHER>& other)
		{
			for (const Data_t& val : other) {
				this->insert(val);
			}

			return *this;
		}

		template<int MAX_SIZE_OTHER>
		LinearSet<Data_t,MAX_SIZE>&  operator+=(LinearSet<Data_t,MAX_SIZE_OTHER>&& other)
		{
			for (Data_t& val : other) {
				this->insert(std::move(val));
			}

			other.clear();
			return *this;
		}

		//set subtraction
		template<int MAX_SIZE_OTHER>
		LinearSet<Data_t,MAX_SIZE>& operator-=(const LinearSet<Data_t,MAX_SIZE_OTHER>& other)
		{
			for (const Data_t& val : other) {
				this->remove(val);
			}

			return *this;
		}
	};


	//symetric set operations
	//both input sets MUST be valid sets (i.e. contain no duplicates)
	template<typename Data_t, int LEFT_SIZE, int RIGHT_SIZE>
	LinearSet<Data_t,LEFT_SIZE+RIGHT_SIZE> set_union(const LinearSet<Data_t,LEFT_SIZE>& left, const LinearSet<Data_t,RIGHT_SIZE>& right)
	{
		LinearSet<Data_t,LEFT_SIZE+RIGHT_SIZE> result;

		if (left.size() >= right.size()) {
			for (const Data_t& val : left) {
				result.data_[ result.cursor_++ ] = val;
			}

			result += right;
		}
		else {
			for (const Data_t& val : right) {
				result.data_[ result.cursor_++ ] = val;
			}

			result += left;
		}

		return result;
	}

	template<typename Data_t, int LEFT_SIZE, int RIGHT_SIZE>
	LinearSet<Data_t, LEFT_SIZE> set_intersect(const LinearSet<Data_t,LEFT_SIZE>& left, const LinearSet<Data_t,RIGHT_SIZE>& right)
	{
		LinearSet<Data_t,LEFT_SIZE> result;
		for (const Data_t& val : left) {
			if (right.contains(val)) {
				result.data_[ result.cursor_++ ] = val;
			}
		}

		return result;
	}
	

}