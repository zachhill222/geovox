#ifndef PLANE_H
#define PLANE_H

#include "util/point.hpp"
#include "util/quaternion.hpp"
#include <iostream>

namespace GeoVox::util{
	class Plane{
	public:
		Plane(): _origin(Point(0,0,0)), _normal(Point(0,0,1)) {calcbasis();}
		Plane( const Point& origin, const Point& normal): _origin(origin), _normal(normal.normalize()) {calcbasis();}
		Plane( const Point& p1, const Point& p2, const Point& p3): _origin(p1), _normal((p2-p1).cross(p3-p1).normalize()) {calcbasis();}
		Plane( const Plane& other){
			_basis[0] = other[0];
			_basis[1] = other[1];
			_normal   = other[2];
		}

		Point operator[](int idx) const;

		void print(std::ostream &stream) const;

		double dist(const Point& point) const; //signed distence to the plane

		Point project(const Point& point) const; //project point from global coordinates to the plane along the _normal direction. return a 2D point in local coordinates.

		Point tolocal(const Point& point) const; //write a point in global coordinates in terms of local (_basis[0], _basis[1], _normal) coordinates.

		Point toglobal(const Point& point) const; //write a local point in the global coordinate system (works for 2D points on the plane or 3D point in local coordinates).

		//rotating
		Plane* operator*=(const GeoVox::util::Quaternion& other);
		Plane  operator*(const GeoVox::util::Quaternion& other) const;

	private:
		Point _origin;
		Point _normal;
		Point _basis[2];

		void calcbasis();
	};
}




#endif