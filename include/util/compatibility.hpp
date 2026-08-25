#pragma once

#include "gutil.hpp"

#include <span>
#include <vector>
#include <cassert>

#include <Eigen/SparseCore>

namespace GV
{
	//////////////////////////////////////
	/// Generic adapter to a span
	//////////////////////////////////////
	template<typename T> requires (std::ranges::contiguous_range<T>)
	inline std::span<T> as_span(T& v) { return std::span<T>{v.begin(), v.end()}; }

	template<typename T> requires (std::ranges::contiguous_range<T>)
	inline std::span<const T> as_span(const T& v) { return std::span<const T>{v.cbegin(), v.cend()}; }
	


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
	/// Adapt Eigen::VectorXd (or other dynamic vectors) to spans and subspans
	//////////////////////////////////////
	#ifdef EIGEN_MAJOR_VERSION
	template<typename T>
	inline std::span<T> as_span(Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {return {v.data(), static_cast<size_t>(v.size())};}

	template<typename T>
	inline std::span<const T> as_span(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {return {v.data(), static_cast<size_t>(v.size())};}

	template<typename T>
	inline std::span<T> as_span(Eigen::Matrix<T, Eigen::Dynamic, 1>& v, const size_t start, const size_t length) {
		assert(start+length <= static_cast<size_t>(v.size()));
		return {v.data()+start, length};
	}

	template<typename T>
	inline std::span<const T> as_span(const Eigen::Matrix<T, Eigen::Dynamic, 1>& v, const size_t start, const size_t length) {
		assert(start+length <= static_cast<size_t>(v.size()));
		return {v.data()+start, length};
	}
	#endif

	//////////////////////////////////////
	/// Adapt gutil::Point to spans and subspans
	//////////////////////////////////////
	template<typename T, int N>
	inline std::span<T,N> as_span(gutil::Point<N,T>& v) {return std::span<T,N>{v.data, size_t(N)};}

	template<typename T, int N>
	inline std::span<const T,N> as_span(const gutil::Point<N,T>& v) {return std::span<const T,N>{v.data, size_t(N)};}

	template<typename T, int N>
	inline std::span<T> as_span(gutil::Point<N,T>& v, const size_t start, const size_t length) {
		assert(start+length <= size_t(N));
		return {v.data+start, length};
	}

	template<typename T, int N>
	inline std::span<const T> as_span(const gutil::Point<N,T>& v, const size_t start, const size_t length) {
		assert(start+length <= size_t(N));
		return {v.data+start, length};
	}
}