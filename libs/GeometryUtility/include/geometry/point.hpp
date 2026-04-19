#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <concepts>
#include <vector>
#include <iostream>
#include <cassert>

namespace gutil
{
	template<int DIM, typename T> requires (DIM>0)
	class Point;



	///////////////////////////////////////////////////////////
	/// Concept to ensure that a type is some form of point
	/// defined in the class below
	///////////////////////////////////////////////////////////
	template<typename T>
	concept pointlike = requires
	{
		typename std::integral_constant<int, T::dim>;
		typename T::scalar_type;
	} and std::same_as<T, Point<T::dim, typename T::scalar_type>>;



	///////////////////////////////////////////////////////////
	/// Utility scalar operations
	///////////////////////////////////////////////////////////
	template<typename T>
	inline constexpr T abs(const T& val) noexcept {return val > T{0} ? val : -val;}

	template<typename T>
	inline constexpr T& abs(T& val) noexcept {
		if (val<T{0}) {val=-val;}
		return val;
	}

	template<typename T>
	inline constexpr T min(const T& a, const T& b) noexcept {return a<b ? a : b;}

	template<typename T>
	inline constexpr T max(const T& a, const T& b) noexcept {return a>b ? a : b;}

	//////////////////////////////////////////////////////////
	/// Pre define any special operations that are constexpr and used in the class
	//////////////////////////////////////////////////////////
	template<int DIM, typename T> requires (DIM>0)
	inline constexpr T dot(const Point<DIM,T>& left, const Point<DIM,T>& right) noexcept
	{
		if constexpr (DIM==1) {
			return left[0]*right[0];
		}
		else if constexpr (DIM==2) {
			return left[0]*right[0] + left[1]*right[1];
		}
		else if constexpr (DIM==3) {
			return left[0]*right[0] + left[1]*right[1] + left[2]*right[2];
		}
		else if constexpr (DIM==4) {
			return left[0]*right[0] + left[1]*right[1] + left[2]*right[2] + left[3]*right[3];
		}
		else {
			T result{left[0]*right[0]};
			for (int i=1; i<DIM; i++) {result += left[i]*right[i];}
			return result;
		}
	}

	template<typename T>
	inline constexpr Point<3,T> cross(const Point<3,T>& left, const Point<3,T>& right) noexcept
	{
		//*this x other
		Point<3,T> result{};
		result[0] = left[1]*right[2] - left[2]*right[1];
		result[1] = left[2]*right[0] - left[0]*right[2];
		result[2] = left[0]*right[1] - left[1]*right[0];
		return result;
	}


	

	//////////////////////////////////////////////////////////
	/// A class for cartesian points in space
	///
	/// Can be treated as a vector with no column or row orientation.
	///
	/// @tparam DIM   The spacial dimension. These points are allocated
	///                   on the stack, so DIM shouldn't be too large.
	/// @tparam T     The underlying numeric type
	///
	/// Note that the type T must implement any comparisons if
	/// floating point rounding is important and unacceptable.
	//////////////////////////////////////////////////////////
	template<int DIM, typename T=double> requires (DIM>0)
	class Point
	{
	public:
		//track type information
		static constexpr int dim = DIM;
		using scalar_type = T;

		//essential constructors
		constexpr Point() noexcept : _data{} {}
		constexpr Point(const Point& other) noexcept : _data{} {std::copy(other._data, other._data+DIM, this->_data);}
		constexpr Point(Point&& other) noexcept : _data{} {std::move(other._data, other._data+DIM, this->_data);}

		//initialize via Point v{1,2,3}
		template<typename U> requires (std::is_nothrow_convertible<U,T>::value)
		constexpr Point(std::initializer_list<U> init_list) noexcept : _data{}
		{
			int i=0;
			for (auto it=init_list.begin(); it!=init_list.end() and i<DIM; ++it, ++i) {
				_data[i] = static_cast<T>(*it);
			}

			//if the initializer list was too small, zero out the remaining indices
			for (;i<DIM; ++i) {_data[i] = T{0};}
		}

		//initialize via Point v(1,2,3)
		template<typename... Ts> requires (sizeof...(Ts)==DIM)
		constexpr Point(Ts... args) : _data{args...} {}

		//initialize via fill
		constexpr Point(const T& val) : _data{} {std::fill(_data, _data+DIM, val);}

		//destructor
		constexpr ~Point() noexcept {}

		//copy and move assignment
		constexpr Point& operator=(const Point& other) noexcept
		{
			if (this != &other) {std::copy(other._data, other._data+DIM, this->_data);}
			return *this;
		}

		constexpr Point& operator=(Point&& other) noexcept
		{
			if (this != &other) {std::move(other._data, other._data+DIM, this->_data);}
			return *this;
		}

		//element access
		inline constexpr const T& operator[](const int idx) const noexcept {assert(0<=idx and idx<DIM); return _data[idx];}
		inline constexpr T& operator[](const int idx) noexcept {assert(0<=idx and idx<DIM); return _data[idx];}

		//type conversion
		template<int OTHER_DIM, typename OTHER_T> requires std::is_nothrow_convertible<T,OTHER_T>::value
		explicit constexpr operator Point<OTHER_DIM, OTHER_T>() const noexcept
		{
			Point<OTHER_DIM,OTHER_T> result{};
			constexpr int min_dim = DIM < OTHER_DIM ? DIM : OTHER_DIM;
			//copy with cast valid data
			for (int i=0; i<min_dim; i++) {result[i] = static_cast<OTHER_T>(_data[i]);}
			//append 0 if necessary
			for (int i=min_dim; i<OTHER_DIM; i++) {result[i] = OTHER_T{0};}
			return result;
		}

		//asymetric/in-place math operations
		constexpr Point& operator+=(const Point& other) noexcept
		{
			if constexpr (DIM==1) {
				_data[0] += other._data[0];
			}
			else if constexpr (DIM==2) {
				_data[0] += other._data[0];
				_data[1] += other._data[1];
			}
			else if constexpr (DIM==3) {
				_data[0] += other._data[0];
				_data[1] += other._data[1];
				_data[2] += other._data[2];
			}
			else if constexpr (DIM==4) {
				_data[0] += other._data[0];
				_data[1] += other._data[1];
				_data[2] += other._data[2];
				_data[3] += other._data[3];
			}
			else {
				for (int i=0; i<DIM; i++) {_data[i] += other._data[i];}
			}

			return *this;
		}

		constexpr Point& operator-=(const Point& other) noexcept
		{
			if constexpr (DIM==1) {
				_data[0] -= other._data[0];
			}
			else if constexpr (DIM==2) {
				_data[0] -= other._data[0];
				_data[1] -= other._data[1];
			}
			else if constexpr (DIM==3) {
				_data[0] -= other._data[0];
				_data[1] -= other._data[1];
				_data[2] -= other._data[2];
			}
			else if constexpr (DIM==4) {
				_data[0] -= other._data[0];
				_data[1] -= other._data[1];
				_data[2] -= other._data[2];
				_data[3] -= other._data[3];
			}
			else {
				for (int i=0; i<DIM; i++) {_data[i] -= other._data[i];}
			}

			return *this;
		}

		constexpr Point& operator*=(const Point& other) noexcept
		{
			if constexpr (DIM==1) {
				_data[0] *= other._data[0];
			}
			else if constexpr (DIM==2) {
				_data[0] *= other._data[0];
				_data[1] *= other._data[1];
			}
			else if constexpr (DIM==3) {
				_data[0] *= other._data[0];
				_data[1] *= other._data[1];
				_data[2] *= other._data[2];
			}
			else if constexpr (DIM==4) {
				_data[0] *= other._data[0];
				_data[1] *= other._data[1];
				_data[2] *= other._data[2];
				_data[3] *= other._data[3];
			}
			else {
				for (int i=0; i<DIM; i++) {_data[i] *= other._data[i];}
			}

			return *this;
		}

		constexpr Point& operator/=(const Point& other) noexcept
		{
			if constexpr (DIM==1) {
				_data[0] /= other._data[0];
			}
			else if constexpr (DIM==2) {
				_data[0] /= other._data[0];
				_data[1] /= other._data[1];
			}
			else if constexpr (DIM==3) {
				_data[0] /= other._data[0];
				_data[1] /= other._data[1];
				_data[2] /= other._data[2];
			}
			else if constexpr (DIM==4) {
				_data[0] /= other._data[0];
				_data[1] /= other._data[1];
				_data[2] /= other._data[2];
				_data[3] /= other._data[3];
			}
			else {
				for (int i=0; i<DIM; i++) {_data[i] /= other._data[i];}
			}

			return *this;
		}

		template<typename U> requires std::is_nothrow_convertible<U,T>::value
		constexpr Point& operator*=(const U& scalar) noexcept
		{
			const T s = static_cast<T>(scalar);

			if constexpr (DIM==1) {
				_data[0] *= s;
			}
			else if constexpr (DIM==2) {
				_data[0] *= s;
				_data[1] *= s;
			}
			else if constexpr (DIM==3) {
				_data[0] *= s;
				_data[1] *= s;
				_data[2] *= s;
			}
			else if constexpr (DIM==4) {
				_data[0] *= s;
				_data[1] *= s;
				_data[2] *= s;
				_data[3] *= s;
			}
			else {
				for (int i=0; i<DIM; i++) {_data[i] *= s;}
			}

			return *this;
		}

		template<typename U> requires std::is_nothrow_convertible<U,T>::value
		constexpr Point& operator/=(const U& other) noexcept
		{
			return *this *= static_cast<T>(U{1}/other);
		}

		constexpr Point& operator%=(const Point& other) noexcept requires(std::integral<T>)
		{
			if constexpr (DIM==1) {
				_data[0] %= other._data[0];
			}
			else if constexpr (DIM==2) {
				_data[0] %= other._data[0];
				_data[1] %= other._data[1];
			}
			else if constexpr (DIM==3) {
				_data[0] %= other._data[0];
				_data[1] %= other._data[1];
				_data[2] %= other._data[2];
			}
			else if constexpr (DIM==4) {
				_data[0] %= other._data[0];
				_data[1] %= other._data[1];
				_data[2] %= other._data[2];
				_data[3] %= other._data[3];
			}
			else {
				for (int i=0; i<DIM; i++) {_data[i] %= other._data[i];}
			}

			return *this;
		}

		constexpr Point operator-() const noexcept
		{
			Point result;
			
			if constexpr (DIM==1) {
				result._data[0] = -_data[0];
			}
			else if constexpr (DIM==2) {
				result._data[0] = -_data[0];
				result._data[1] = -_data[1];
			}
			else if constexpr (DIM==3) {
				result._data[0] = -_data[0];
				result._data[1] = -_data[1];
				result._data[2] = -_data[2];
			}
			else if constexpr (DIM==4) {
				result._data[0] = -_data[0];
				result._data[1] = -_data[1];
				result._data[2] = -_data[2];
				result._data[3] = -_data[3];
			}
			else {
				for (int i=0; i<DIM; i++) {result._data[i] = -_data[i];}
			}



			return result;
		}

		//norms
		constexpr T norminfty() const noexcept
		{
			T result{abs(_data[0])};
			for (int i=1; i<DIM; i++) {
				T val = abs(_data[i]);
				result = val > result ? val : result;
			}
			return result;
		}

		constexpr T norm1() const noexcept
		{
			T result{abs(_data[0])};
			for (int i=1; i<DIM; i++) {result += abs(_data[i]);}
			return result;
		}

		constexpr T squaredNorm() const noexcept
		{
			return gutil::dot(*this,*this);
		}

		//accumulators
		constexpr T prod() const noexcept
		{
			T result{_data[0]};
			for (int i=1; i<DIM; i++) {result *= _data[i];}
			return result;
		}

		constexpr T sum() const noexcept
		{
			T result{_data[0]};
			for (int i=1; i<DIM; i++) {result += _data[i];}
			return result;
		}

		constexpr T max() const noexcept
		{
			T result{_data[0]};
			for (int i=1; i<DIM; i++) {
				result = _data[i] > result ? _data[i] : result;
			}

			return result;
		}

		constexpr T min() const noexcept
		{
			T result{_data[0]};
			for (int i=1; i<DIM; i++) {
				result = _data[i] < result ? _data[i] : result;
			}

			return result;
		}

		//standard vector operations
		inline constexpr T dot(const Point& other) const noexcept
		{
			return gutil::dot(*this, other);
		}

		inline constexpr Point<3,T> cross(const Point& other) const noexcept requires (DIM==3)
		{
			return cross(*this, other);
		}

	protected:
		T _data[DIM];
	};

	///////////////////////////////////////////////////////////////////
	/// Ensure that the concept 'pointlike' is valid
	///////////////////////////////////////////////////////////////////
	static_assert(pointlike<Point<3,float>>);
	static_assert(pointlike<Point<2,float>>);
	static_assert(pointlike<Point<3,double>>);
	static_assert(pointlike<Point<2,double>>);


	///////////////////////////////////////////////////////////////////
	//////////////////////// COMPARISON (CONE) ////////////////////////
	///////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	constexpr bool operator==(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		for (int i=0; i<DIM; i++) {
			if (left[i]!=right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	constexpr bool operator!=(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		return !(left == right);
	}

	template<int DIM, typename T>
	constexpr bool operator<(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		for (int i=0; i<DIM; i++) {
			if (left[i] >= right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	constexpr bool operator<=(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		for (int i=0; i<DIM; i++) {
			if (left[i] > right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	constexpr bool operator>(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		for (int i=0; i<DIM; i++) {
			if (left[i] <= right[i]) {return false;}
		}
		return true;
	}

	template<int DIM, typename T>
	constexpr bool operator>=(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		for (int i=0; i<DIM; i++) {
			if (left[i] < right[i]) {return false;}
		}
		return true;
	}

	/////////////////////////////////////////////////////////////////////////////
	//////////////////////// ARITHMETIC (COMPONENT-WISE) ////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	constexpr Point<DIM,T> operator+(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{left};
		return result+=right;
	}

	template<int DIM, typename T>
	constexpr Point<DIM,T> operator-(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{left};
		return result-=right;
	}

	template<int DIM, typename T>
	constexpr Point<DIM,T> operator*(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{left};
		return result*=right;
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	constexpr Point<DIM,T> operator*(const U& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{right};
		return result*=static_cast<T>(left);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	constexpr Point<DIM,T> operator*(const Point<DIM,T>& left, const U& right) {
		Point<DIM,T> result{left};
		return result*=static_cast<T>(right);
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	constexpr Point<DIM,T> operator/(const Point<DIM,T>& left, const U& right) {
		Point<DIM,T> result{left};
		return result/=static_cast<T>(right);
	}

	template<int DIM, typename T>
	constexpr Point<DIM,T> operator/(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{left};
		return result/=right;
	}

	template<int DIM, typename T, typename U> requires std::is_nothrow_convertible<U,T>::value
	constexpr Point<DIM,T> operator/(const U& left, const Point<DIM,T>& right) {
		Point<DIM,T> result(static_cast<T>(left));
		return result/=right;
	}

	template<int DIM, typename T> requires std::integral<T>
	constexpr Point<DIM,T> operator%(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{left};
		return result%=right;
	}

	///////////////////////////////////////////////////////////////////////////////
	//////////////////////// TRADITIONAL VECTOR OPERATIONS ////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	inline constexpr T squaredNorm(const Point<DIM,T>& point) noexcept {return gutil::dot(point,point);}

	template<int DIM, typename T>
	inline T norm2(const Point<DIM,T>& point) noexcept {
		if constexpr (sizeof(T) == 4) {return T{std::sqrt(static_cast<float>(point.squaredNorm()))};}
		if constexpr (sizeof(T) == 8) {return T{std::sqrt(static_cast<double>(point.squaredNorm()))};}
	}

	template<int DIM, typename T>
	inline constexpr T norm1(const Point<DIM,T>& point) noexcept {return point.norm1();}

	template<int DIM, typename T>
	inline constexpr T norminfty(const Point<DIM,T>& point) noexcept {return point.norminfty();}

	template<int DIM, typename T>
	constexpr Point<DIM,T> normalize(const Point<DIM,T>& point) noexcept
	{
		T scale{norm2(point)};
		assert(scale!=T{0});
		return point/scale;
	}

	//////////////////////////////////////////////////////////////////////////
	//////////////////////// OTHER UTILITY OPERATIONS ////////////////////////
	//////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T>
	constexpr Point<DIM,T> abs(const Point<DIM,T>& point) noexcept {
		Point<DIM,T> result{};
		for (int i=0; i<DIM; i++) {result[i] = abs(point[i]);}
		return result;
	}

	template<int DIM, typename T>
	constexpr Point<DIM,T> elmax(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{};
		for (int i=0; i<DIM; i++) {result[i] = max(left[i], right[i]);}
		return result;
	}

	//note CONTAINTER_T should be a std::vector or std::array
	template<typename CONTAINER_T>
	constexpr auto elmax(const CONTAINER_T& container) {
		using P = typename CONTAINER_T::value_type;
		auto n = container.size();
		using Index_t = decltype(n);
		assert(container.size()>Index_t{0});

		P result = container[0];
		for (Index_t i=1; i<n; ++i) {
			result = elmax(result, container[i]);
		}

		return result;
	}

	template<int DIM, typename T>
	constexpr Point<DIM,T> elmin(const Point<DIM,T>& left, const Point<DIM,T>& right) {
		Point<DIM,T> result{};
		for (int i=0; i<DIM; i++) {result[i] = min(left[i], right[i]);}
		return result;
	}

	//note CONTAINTER_T should be a std::vector or std::array
	template<typename CONTAINER_T>
	constexpr auto elmin(const CONTAINER_T& container) {
		using P = typename CONTAINER_T::value_type;
		auto n = container.size();
		using Index_t = decltype(n);
		assert(container.size()>Index_t{0});

		P result = container[0];
		for (Index_t i=1; i<n; ++i) {
			result = elmin(result, container[i]);
		}

		return result;
	}

	template<int DIM, typename T>
	inline constexpr T max(const Point<DIM,T>& point) {return point.max();}

	template<int DIM, typename T>
	inline constexpr T min(const Point<DIM,T>& point) {return point.min();}

	template<int DIM, typename T>
	inline constexpr T sum(const Point<DIM,T>& point) {return point.sum();}

	template<int DIM, typename T>
	inline constexpr T prod(const Point<DIM,T>& point) {return point.prod();}

	template<int DIM, typename T>
	std::ostream& operator<<(std::ostream& os, const Point<DIM,T>& point) {
		for (int i = 0; i < DIM-1; i++) {os << point[i] << " ";}
		os << point[DIM-1];
		return os;
	}


	////////////////////////////////////////////////////////////////////////////////
	/// Sum points in careful precision order
	///
	/// @param points A vector of points to add
	///
	/// @tparam W is the input point type
	/// @tparam U is the type that the arithmetic should be done in
	/// @tparam T is the output type
	////////////////////////////////////////////////////////////////////////////////
	template<int DIM, typename T, typename U, typename W>
	constexpr Point<DIM,T> sorted_sum(const std::vector<Point<DIM,W>>& points) noexcept {
		if (points.empty()) {return Point<DIM,T>();}
		
		Point<DIM,T> result;
		std::vector<T> component;
		component.reserve(points.size());
		for (int i = 0; i < DIM; i++) {
			component.clear();
			for ( const Point<DIM,W> &p : points) {
				component.push_back(static_cast<U>(p[i]));
			}

			std::sort(component.begin(), component.end(), [](T a, T b) {
				            return abs(a) < abs(b);});

			U sum = U{0};
			for (U val : component) {
				sum += val;
			}
			result[i] = static_cast<T>(sum);
		}
		return result;
	}

	/// Convenient ways to call the sorted sum.
	template<int DIM, typename T, typename U, typename W>
	inline constexpr Point<DIM,T> sorted_sum(std::initializer_list<Point<DIM,W>> points) noexcept {
	    return sorted_sum<DIM,T,U,W>(std::vector<Point<DIM,W>>(points.begin(), points.end()));
	}

	

}