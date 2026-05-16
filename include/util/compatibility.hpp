#pragma once

#include "util/point.hpp"
#include <span>
#include <vector>
#include <cassert>

#include <Eigen/SparseCore>

namespace GV
{
	//////////////////////////////////////
	/// Adapt std::vector to spans and subspans
	//////////////////////////////////////
	template<typename T>
	inline std::span<T> as_span(std::vector<T>& v) {return {v.data(), v.size()};}

	template<typename T>
	inline std::span<const T> as_span(const std::vector<T>& v) {return {v.data(), v.size()};}

	template<typename T>
	inline std::span<T> as_span(std::vector<T>& v, const size_t start, const size_t length) {
		assert(start+length <= v.size());
		return {v.data()+start, length};
	}

	template<typename T>
	inline std::span<const T> as_span(const std::vector<T>& v, const size_t start, const size_t length) {
		assert(start+length <= v.size());
		return {v.data()+start, length};
	}

	//////////////////////////////////////
	/// Adapt std::array to spans and subspans
	//////////////////////////////////////
	template<typename T, size_t N>
	inline std::span<T,N> as_span(std::array<T,N>& v) {return std::span<T,N>{v};}

	template<typename T, size_t N>
	inline std::span<const T,N> as_span(const std::array<T,N>& v) {return std::span<const T,N>{v};}

	template<typename T, size_t N>
	inline std::span<T> as_span(std::array<T,N>& v, const size_t start, const size_t length) {
		assert(start+length <= N);
		return {v.data()+start, length};
	}

	template<typename T, size_t N>
	inline std::span<const T> as_span(const std::array<T,N>& v, const size_t start, const size_t length) {
		assert(start+length <= N);
		return {v.data()+start, length};
	}


	//////////////////////////////////////
	/// Allow as_span to be called on std::span
	//////////////////////////////////////
	template<typename T>
	inline std::span<T> as_span(std::span<T>& v) {return v;}

	template<typename T>
	inline std::span<const T> as_span(const std::span<T>& v) {return v;}

	template<typename T, size_t N> //fallback to convert a compile time size to a runtime size
	inline std::span<T> as_span(std::span<T,N>& v) {return {v.data(), N};}

	template<typename T, size_t N> //fallback to convert a compile time size to a runtime size
	inline std::span<const T> as_span(const std::span<T,N>& v) {return {v.data(), N};}


	//////////////////////////////////////
	/// Adapt Eigen::VectorXd to spans and subspans
	//////////////////////////////////////
	inline std::span<double> as_span(Eigen::VectorXd& v) {return {v.data(), static_cast<size_t>(v.size())};}

	inline std::span<const double> as_span(const Eigen::VectorXd& v) {return {v.data(), static_cast<size_t>(v.size())};}

	template<typename T>
	inline std::span<T> as_span(Eigen::VectorXd& v, const size_t start, const size_t length) {
		assert(start+length <= static_cast<size_t>(v.size()));
		return {v.data()+start, length};
	}

	template<typename T>
	inline std::span<const T> as_span(const Eigen::VectorXd& v, const size_t start, const size_t length) {
		assert(start+length <= static_cast<size_t>(v.size()));
		return {v.data()+start, length};
	}


	//////////////////////////////////////
	/// Adapt GV::Point to spans and subspans
	//////////////////////////////////////
	template<typename T, int N>
	inline std::span<T,N> as_span(Point<N,T>& v) {return std::span<T,N>{v._data, size_t(N)};}

	template<typename T, int N>
	inline std::span<const T,N> as_span(const Point<N,T>& v) {return std::span<const T,N>{v._data, size_t(N)};}

	template<typename T, int N>
	inline std::span<T> as_span(Point<N,T>& v, const size_t start, const size_t length) {
		assert(start+length <= size_t(N));
		return {v._data+start, length};
	}

	template<typename T, int N>
	inline std::span<const T> as_span(const Point<N,T>& v, const size_t start, const size_t length) {
		assert(start+length <= size_t(N));
		return {v._data+start, length};
	}
}