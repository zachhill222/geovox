#pragma once

#include "geometry/point.hpp"
#include "math/quaternion.hpp"

#include <iostream>

namespace gutil {

	template<typename T=double>
	class Plane {
	public:
		using Point_t = Point<3,T>;

		Plane(): _origin(Point_t{0,0,0}), _normal(Point_t{0,0,1}) {calcbasis();}
		Plane( const Point_t& origin, const Point_t& normal): _origin(origin), _normal(normalize(normal)) {calcbasis();}
		Plane( const Point_t& p1, const Point_t& p2, const Point_t& p3): _origin(p1), _normal(normalize(cross(p3-p1,p2-p1))) {calcbasis();}

		T dist(const Point_t& point) const; //signed distence to the plane

		Point_t project(const Point_t& point) const; //project point from global coordinates to the plane along the _normal direction. return a 2D point in local coordinates.

		Point_t tolocal(const Point_t& point) const; //write a point in global coordinates in terms of local (_basis[0], _basis[1], _normal) coordinates.

		Point_t toglobal(const Point_t& point) const; //write a local point in the global coordinate system (works for 2D points on the plane or 3D point in local coordinates).

	private:
		Point_t _origin;
		Point_t _normal;
		Point_t _basis[2];

		void calcbasis();
	};

	template<typename T>
	void Plane<T>::calcbasis()
	{
		Point_t vec1, vec2;

		vec1 = cross(_normal,Point_t{1,0,0});
		vec2 = cross(_normal,Point_t{0,1,0});
		if (squaredNorm(vec2) > squaredNorm(vec1)) {vec1 = vec2;
		}

		vec2 = cross(_normal,Point_t{0,0,1});
		if (squaredNorm(vec2) > squaredNorm(vec1)) {vec1 = vec2;}

		_basis[0] = vec1;
		_basis[1] = cross(_normal,_basis[0]);
	}


	template<typename T>
	T Plane<T>::dist(const Point_t& point) const {return dot(point-_origin,_normal);}

	template<typename T>
	typename Plane<T>::Point_t Plane<T>::project(const Point_t& point) const
	{
		Point_t local = tolocal(point); //write point in local coordinates
		return local[0]*_basis[0] + local[1]*_basis[1];
	}

	template<typename T>
	typename Plane<T>::Point_t Plane<T>::tolocal(const Point_t& point) const
	{
		Point_t shift = point - _origin;
		T a = dot(shift,_basis[0]);
		T b = dot(shift,_basis[1]);
		T c = dot(shift,_normal);
		return Point_t(a,b,c);
	}

	template<typename T>
	typename Plane<T>::Point_t Plane<T>::toglobal(const Point_t& point) const
	{
		Point_t result = _origin + point[0]*_basis[0] + point[1]*_basis[1] + point[2]*_normal;
		return result;
	}
}