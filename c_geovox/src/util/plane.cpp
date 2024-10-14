#include "util/plane.hpp"

namespace GeoVox::util{
	void Plane::calcbasis(){
		Point3 vec1, vec2;

		vec1 = _normal.cross(Point3(1,0,0));
		vec2 = _normal.cross(Point3(0,1,0));
		if (vec2.squaredNorm() > vec1.squaredNorm()){
			vec1 = vec2;
		}

		vec2 = _normal.cross(Point3(0,0,1));
		if (vec2.squaredNorm() > vec1.squaredNorm()){
			vec1 = vec2;
		}

		_basis[0] = vec1;
		_basis[1] = _normal.cross(_basis[0]);
	}


	double Plane::dist(const Point3& point) const{
		return (point-_origin).dot(_normal);
	}

	Point3 Plane::project(const Point3& point) const{
		Point3 local = tolocal(point); //write point in local coordinates
		return local[0]*_basis[0] + local[1]*_basis[1];
	}

	Point3 Plane::tolocal(const Point3& point) const{
		Point3 shift = point - _origin;
		double a = shift.dot(_basis[0]);
		double b = shift.dot(_basis[1]);
		double c = shift.dot(_normal);
		return Point3(a,b,c);
	}

	Point3 Plane::toglobal(const Point3& point) const{
		Point3 result = _origin + point[0]*_basis[0] + point[1]*_basis[1];
		if (point.size()>2){
			result += point[2]*_normal;
		}
		return result;
	}

	Plane* Plane::operator*=(const GeoVox::util::Quaternion& other){
		_normal = other.rotate(_normal);
		_basis[0] = other.rotate(_basis[0]);
		_basis[1] = other.rotate(_basis[1]);
		return this;
	}

	Plane  Plane::operator*(const GeoVox::util::Quaternion& other) const{
		Plane result = *this;
		result*=other;
		return result;
	}
}