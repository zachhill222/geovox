#pragma once

#include "geometry/point.hpp"

#include <string>
#include <sstream>
#include <cmath>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <type_traits>

namespace gutil {
	template<int DIM=3, typename T=double>
	class Box {
	private:
		Point<DIM,T> _low;
		Point<DIM,T> _high;

	public:
		static constexpr int dim = DIM;
		static constexpr int n_vertices = std::pow(2,DIM);
		using scalar_type = T;
		using Point_t = Point<DIM,T>;

		////////////////////////////////////////////////////////////////
		// Constructors
		////////////////////////////////////////////////////////////////
		
		// Default constructor: unit box centered at origin
		constexpr Box() noexcept : _low(Point_t(-1.0)), _high(Point_t(1.0)) {}

		// Constructor from two points (automatically orders them)
		constexpr Box(const Point_t &vertex1, const Point_t &vertex2) noexcept : 
			_low(elmin(vertex1, vertex2)), _high(elmax(vertex1, vertex2)) {
			assert(_low <= _high);
		}

		//use default copy and move constructors and assignment
		constexpr Box(const Box &other) noexcept = default;
		constexpr Box(Box &&other) noexcept = default;
		constexpr Box& operator=(const Box &other) = default;
		constexpr Box& operator=(Box &&other) noexcept = default;
		~Box() = default;
		
		////////////////////////////////////////////////////////////////
		// Attributes
		////////////////////////////////////////////////////////////////
		constexpr const Point_t& low()  const noexcept {return _low;}
		constexpr const Point_t& high() const noexcept {return _high;}
		constexpr Point_t center() const noexcept {return T(0.5) * (_low + _high);}
		constexpr Point_t sidelength() const noexcept {return _high - _low;}
		constexpr T diameter() const noexcept {return norm2(_high - _low);}
		constexpr T volume() const noexcept {
			T vol = 1;
			for (int i = 0; i < DIM; i++) {
				vol *= (_high[i] - _low[i]);
			}
			return vol;
		}

		////////////////////////////////////////////////////////////////
		// Vertex access
		////////////////////////////////////////////////////////////////
		
		/// Get i-th vertex in VTK pixel/voxel order
		/// Binary encoding: bit i determines whether to use low[i] or high[i]
		constexpr Point_t voxelvertex(const int idx) const noexcept {
			assert(idx >= 0 && idx < (1 << DIM));
			Point_t vertex;
			int p = idx;
			for (int i = 0; i < DIM; i++) {
				vertex[i] = (p & 1) ? _high[i] : _low[i];
				p >>= 1;
			}
			return vertex;
		}

		/// Get i-th vertex in VTK quad/hexahedron order
		/// (swaps vertices 2-3 and 6-7 from voxel ordering)
		constexpr Point_t hexvertex(const int idx) const noexcept requires(DIM==2 or DIM==3) {
			switch (idx) {
				case 2: return voxelvertex(3);
				case 3: return voxelvertex(2);
				case 6: return voxelvertex(7);
				case 7: return voxelvertex(6);
				default: return voxelvertex(idx);
			}
		}
		////////////////////////////////////////////////////////////////
		// Containment and intersection
		////////////////////////////////////////////////////////////////
		/// Check if point is in the closed box
		constexpr bool contains(const Point_t &point) const noexcept {
			return _low <= point && point <= _high;
		}
		
		/// Check if point is in the open box
		constexpr bool contains_strict(const Point_t &point) const noexcept {
			return _low < point && point < _high;
		}
		
		/// Check if this box contains the other box
		constexpr bool contains(const Box<DIM,T> &other) const noexcept {
			return _low <= other._low && other._high <= _high;
		}
		
		/// Check if this box intersects the other box (check projection onto each axis)
		constexpr bool intersects(const Box<DIM,T> &other) const noexcept {
			for (int i = 0; i < DIM; i++) {
				if (_high[i] < other._low[i] || other._high[i] < _low[i]) {
					return false;
				}
			}
			return true;
		}

		/// Find the support point: vertex that maximizes dot(vertex, direction)
		constexpr Point_t support(const Point_t &direction) const noexcept {
			T maxdot = dot(direction, voxelvertex(0));
			int maxind = 0;

			for (int i = 1; i < (1 << DIM); i++) {
				T tempdot = dot(direction, voxelvertex(i));
				if (tempdot > maxdot) {
					maxdot = tempdot;
					maxind = i;
				}
			}
			return voxelvertex(maxind);
		}

		////////////////////////////////////////////////////////////////
		// Geometric transformations
		////////////////////////////////////////////////////////////////
		
		/// Shift box by vector
		constexpr Box& operator+=(const Point_t &shift) noexcept {
			_low += shift;
			_high += shift;
			return *this;
		}

		constexpr Box operator+(const Point_t &shift) const noexcept {
			return Box(_low + shift, _high + shift);
		}

		constexpr Box& operator-=(const Point_t &shift) noexcept {
			_low -= shift;
			_high -= shift;
			return *this;
		}

		constexpr Box operator-(const Point_t &shift) const noexcept {
			return Box(_low - shift, _high - shift);
		}

		/// Scale box relative to its center
		template<typename U>
		constexpr Box& operator*=(const U& scale) noexcept {
			Point_t c = center();
			T s = static_cast<T>(scale);
			_low = c + s * (_low - c);
			_high = c + s * (_high - c);
			return *this;
		}

		template<typename U>
		constexpr Box operator*(const U& scale) const noexcept requires(std::is_convertible<U,T>::value) {
			Point_t c = center();
			T s = static_cast<T>(scale);
			return Box(c + s * (_low - c), c + s * (_high - c));
		}

		template<typename U>
		constexpr Box& operator/=(const U& scale) {
			return (*this) *= (T(1) / static_cast<T>(scale));
		}

		template<typename U>
		constexpr Box operator/(const U& scale) const {
			return (*this) * (T(1) / static_cast<T>(scale));
		}

		////////////////////////////////////////////////////////////////
		// Comparison
		////////////////////////////////////////////////////////////////
		
		constexpr bool operator==(const Box<DIM,T> &other) const {
			return _low == other._low && _high == other._high;
		}

		constexpr bool operator!=(const Box<DIM,T> &other) const {
			return !(*this == other);
		}
	};

	////////////////////////////////////////////////////////////////
	// Free functions
	////////////////////////////////////////////////////////////////
	template <int DIM, typename T>
	Box<DIM,T> combine(const Box<DIM,T>& A, const Box<DIM,T>& B) {
		return {elmin(A.low(),B.low()), elmax(B.low(),B.high())};
	}

	/// Return intersection of two boxes (undefined if boxes don't intersect)
	template <int DIM, typename T>
	Box<DIM,T> intersection(const Box<DIM,T>& A, const Box<DIM,T>& B) {
		assert(A.intersects(B));
		return Box(elmax(A.low(), B.low()), elmin(A.high(), B.high()));
	}


	template <int DIM, typename T, typename U>
	constexpr Box<DIM,T> operator*(const U &scale, const Box<DIM,T> &box) {
		return box * scale;
	}

	template <int DIM, typename T>
	constexpr T distance_squared(const Box<DIM,T> &box, const Point<DIM,T> &point) {
		if (box.contains(point)) {
			return T{0};
		}

		// Compute distance to closest point on box surface
		T dist_sq = 0;
		for (int i=0; i<DIM; i++) {
			if (point[i] < box.low()[i]) {
				T diff = box.low()[i] - point[i];
				dist_sq += diff * diff;
			} else if (point[i] > box.high()[i]) {
				T diff = point[i] - box.high()[i];
				dist_sq += diff * diff;
			} //no contribution in this axis if the point is in the box interval
		}
		return dist_sq;
	}

	template <int DIM, typename T>
	inline T distance(const Box<DIM,T> &box, const Point<DIM,T> &point) {
		return std::sqrt(distance_squared(box, point));
	}

	template<int DIM, typename T>
	std::ostream& operator<<(std::ostream& os, const Box<DIM,T>& box) {
		return os << "(" << box.low() << ") to (" << box.high() << ")";
	}
}