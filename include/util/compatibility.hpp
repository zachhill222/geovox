#pragma once


#include <span>
#include <vector>
#include <Eigen/SparseCore>



namespace GV
{
	//////////////////////////////////////
	/// Adapt containers to spans
	//////////////////////////////////////
	template<typename T>
	inline std::span<T> as_span(std::vector<T>& v) {return {v.data(), v.size()};}

	template<typename T>
	inline std::span<const T> as_span(const std::vector<T>& v) {return {v.data(), v.size()};}

	template<typename T>
	inline std::span<T> as_span(std::span<T>& v) {return v;}

	template<typename T>
	inline std::span<const T> as_span(const std::span<T>& v) {return v;}

	inline std::span<double> as_span(Eigen::VectorXd& v) {return {v.data(), static_cast<size_t>(v.size())};}

	inline std::span<const double> as_span(const Eigen::VectorXd& v) {return {v.data(), static_cast<size_t>(v.size())};}
}