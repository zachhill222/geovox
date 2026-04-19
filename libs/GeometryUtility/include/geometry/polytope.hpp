#pragma once

#include "geometry/point.hpp"

#include <iostream>
#include <array>
#include <initializer_list>

namespace gutil {
	////////////////////////////////
	/// POLYTOPE DEFINITION
	///
	/// @tparam N   the number of vertices in the polytope
	/// @tparam DIM the spacial dimension the polytope is embedded in
	/// @tparam T   the scalar type that models the real line
	////////////////////////////////
	template<int N, int DIM, typename T=double>
	class Polytope {
	public:
		//type aliases and helpful constants
		using Point_t = Point<DIM,T>;
		static constexpr int n_vertices = N;
		static constexpr int dim = DIM;

		//constructors
		constexpr Polytope() noexcept {}
		Polytope(std::initializer_list<Point_t> list) noexcept {
			int i;
			for (auto it=list.begin(); it!=list.end() and i<N; ++it, ++i) {
				_vertices[i] = std::move(list[i]);
			}
			for (;i<N; i++) {_vertices[i] = Point_t(T{0});}
		}

		constexpr Polytope(const Polytope& other) noexcept : _vertices{other._vertices} {}
		constexpr Polytope(Polytope&& other) noexcept : _vertices{std::move(other._vertices)} {}

		//assignment
		constexpr Polytope& operator=(const Polytope& other) noexcept {
			if (this != &other) {
				_vertices = other._vertices;
			}
			return *this;
		}

		constexpr Polytope& operator=(Polytope&& other) noexcept {
			if (this != &other) {
				_vertices = std::move(other._vertices);	
			}
			return *this;
		}

		//accessors
		inline constexpr const Point_t& operator[](const int idx) const noexcept;
		inline constexpr Point_t& operator[](const int idx) noexcept;
		inline constexpr const Point_t& vertex(const int idx) const noexcept {return (*this)[idx%N];}
		inline constexpr Point_t& vertex(const int idx) noexcept {return (*this)[idx%N];}

		//iterators
		constexpr auto begin()        noexcept { return _vertices.begin(); }
		constexpr auto begin()  const noexcept { return _vertices.cbegin();}
		constexpr auto cbegin() const noexcept { return _vertices.cbegin();}
		constexpr auto end()          noexcept { return _vertices.end();   }
		constexpr auto end()    const noexcept { return _vertices.cend();  }
		constexpr auto cend()   const noexcept { return _vertices.cend();  }

		//geometry operations
		constexpr Point_t support(const Point_t& direction) const noexcept;
		constexpr Box<DIM,T> bbox() const noexcept; //axis aligned bounding box
		
		template<int M>
		constexpr bool share_vertex(const Polytope<M,DIM,T>& other, const T& tol=0) const noexcept;
		
		template<int M>
		constexpr int n_shared_vertices(const Polytope<M,DIM,T>& other, const T& tol=0) const noexcept;

	protected:
		std::array<Point_t, N> _vertices;
	};


	//POLYTOPE IMPLEMENTATION

	template<int N, int DIM, typename T>
	template<int M>
	constexpr bool Polytope<N,DIM,T>::share_vertex(const Polytope<M,DIM,T>& other, const T& tol) const noexcept {
		const T tol2{tol*tol};
		for (int i=0; i<N; i++) {
			for (int j=0; j<M; j++) {
				if (squaredNorm(_vertices[i] - other[j]) <= tol2) {return true;}
			}
		}
		return false;
	}

	template<int N, int DIM, typename T>
	template<int M>
	constexpr int Polytope<N,DIM,T>::n_shared_vertices(const Polytope<M,DIM,T>& other, const T& tol) const noexcept {
		const T tol2{tol*tol};
		int count = 0;
		for (int i=0; i<N; i++) {
			for (int j=0; j<M; j++) {
				if (squaredNorm(_vertices[i] - other[j]) <= tol2) {
					count++;
					break;
				}
			}
		}
		return count;
	}

	template<int N, int DIM, typename T>
	inline constexpr const Point<DIM,T>& Polytope<N,DIM,T>::operator[](const int idx) const noexcept {
		assert(0<=idx and idx<N);
		return _vertices[idx];
	}
	
	template<int N, int DIM, typename T>
	inline constexpr Point<DIM,T>& Polytope<N,DIM,T>::operator[](const int idx) noexcept {
		assert(0<=idx and idx<N);
		return _vertices[idx];
	}
	
	template<int N, int DIM, typename T>
	constexpr Point<DIM,T> Polytope<N,DIM,T>::support(const Point<DIM,T>& direction) const noexcept {
		T maxdot = dot(direction, _vertices[0]);
		int maxind = 0;

		for (int i=1; i<N; i++){
			const T tempdot = dot(direction, _vertices[i]);
			if (tempdot > maxdot){
				maxdot = tempdot;
				maxind = i;
			}
		}

		return _vertices[maxind];
	}
	
	template<int N, int DIM, typename T>
	constexpr Box<DIM,T> Polytope<N,DIM,T>::bbox() const noexcept {
		Point<DIM,T> low  = _vertices[0];
		Point<DIM,T> high = _vertices[0]; 

		for (int i=1; i<N; i++) {
			low  = elmin(low, _vertices[i]);
			high = elmax(high, _vertices[i]);
		}

		return Box<DIM,T>{low,high};
	}

	template<int N, int DIM, typename T>
	constexpr bool operator==(const Polytope<N,DIM,T>& left, const Polytope<N,DIM,T>& right) {
		for (int i=0; i<N; i++) {
			if (left[i] != right[i]) {return false;}
		}
		return true;
	}

	template<int N, int DIM, typename T>
	std::ostream& operator<<(std::ostream& stream, const Polytope<N,DIM,T>& polytope) {
		int i=0;
		for (const auto& v : polytope){
			stream << i << ": " << v << "\n";
			i++;
		}
		return stream;
	}
}
