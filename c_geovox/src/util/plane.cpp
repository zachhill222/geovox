#include "util/plane.hpp"

namespace GeoVox::util{
	void Plane::calcbasis(){
		Point vec1 = _normal.cross(Point(1,0,0));
		
		Point vec2 = _normal.cross(Point(0,1,0));
		if (vec2.norm2() > vec1.norm2()){
			vec1 = vec2;
		}

		vec2 = _normal.cross(Point(0,0,1));
		if (vec2.norm2() > vec1.norm2()){
			vec1 = vec2;
		}

		_basis[0] = vec1;
		_basis[1] = _normal.cross(_basis[0]);
	}


	Point Plane::operator[](int idx) const{
		switch (idx) {
		case 0:
			return _basis[0];
			break;
		case 1:
			return _basis[1];
			break;
		case 2:
			return _normal;
			break;
		}

		std::out_of_range("A Plane only has 3 basis vectors. Index must be between 0 and 2.");
		return Point();
	}

	double Plane::dist(const Point& point) const{
		return (point-_origin).dot(_normal);
	}

	Point Plane::project(const Point& point) const{
		Point local = this->tolocal(point); //write point in local coordinates
		return Point(local[0], local[1]);
	}

	Point Plane::tolocal(const Point& point) const{
		Point shift = point - _origin;
		double a = shift.dot(_basis[0]);
		double b = shift.dot(_basis[1]);
		double c = shift.dot(_normal);
		return Point(a,b,c);
	}

	Point Plane::toglobal(const Point& point) const{
		Point result = _origin + point[0]*_basis[0] + point[1]*_basis[1];
		if (point.len()>2){
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


	void Plane::print(std::ostream& stream) const{
		stream << "origin= ";
		_origin.print(stream);
		stream << std::endl;
		stream << "normal= ";
		_normal.print(stream);
		stream << std::endl;
		stream << "basis[0]= ";
		_basis[0].print(stream);
		stream << std::endl;
		stream << "basis[1]= ";
		_basis[1].print(stream);
		stream << std::endl;
	}
}