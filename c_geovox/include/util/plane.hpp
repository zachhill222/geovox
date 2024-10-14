#ifndef PLANE_H
#define PLANE_H

#include "util/point.hpp"
#include "util/quaternion.hpp"

#include <iostream>

namespace GeoVox::util{
	
	class Plane{
	public:
		Plane(): _origin(Point3(0,0,0)), _normal(Point3(0,0,1)) {calcbasis();}
		Plane( const Point3& origin, const Point3& normal): _origin(origin), _normal(normal.normalized()) {calcbasis();}
		Plane( const Point3& p1, const Point3& p2, const Point3& p3): _origin(p1), _normal((p3-p1).cross(p2-p1).normalized()) {calcbasis();}

		double dist(const Point3& point) const; //signed distence to the plane

		Point3 project(const Point3& point) const; //project point from global coordinates to the plane along the _normal direction. return a 2D point in local coordinates.

		Point3 tolocal(const Point3& point) const; //write a point in global coordinates in terms of local (_basis[0], _basis[1], _normal) coordinates.

		Point3 toglobal(const Point3& point) const; //write a local point in the global coordinate system (works for 2D points on the plane or 3D point in local coordinates).

		//rotating
		Plane* operator*=(const GeoVox::util::Quaternion& other);
		Plane  operator*(const GeoVox::util::Quaternion& other) const;

	private:
		Point3 _origin;
		Point3 _normal;
		Point3 _basis[2];

		void calcbasis();
	};
}




#endif