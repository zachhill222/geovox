#include "geometry/particles.hpp"

double sgn(const double& x){
	if (x<0){
		return -1.0;
	}
	return 1.0;
}

namespace GeoVox::geometry{
	Point3 SuperEllipsoid::center() const{
		return _center;
	}

	SuperEllipsoid SuperEllipsoid::copy() const{
		SuperEllipsoid result = SuperEllipsoid(_r, _eps1, _eps2, _center, _Q);
		return result;
	}

	SuperEllipsoid* SuperEllipsoid::operator+=(const Point3& other){
		_center+= other;
		return this;
	}

	SuperEllipsoid SuperEllipsoid::operator+(const Point3& other) const{
		SuperEllipsoid result = copy();
		result+=other;
		return result;
	}

	SuperEllipsoid* SuperEllipsoid::operator*=(const Quaternion& other){
		_Q*= other;
		return this;
	}

	SuperEllipsoid SuperEllipsoid::operator*(const Quaternion& other) const{
		SuperEllipsoid result = copy();
		result*=other;
		return result;
	}

	//scaling
	SuperEllipsoid* SuperEllipsoid::operator*=(const double& other){
		_r*=other;
		return this;
	}
	SuperEllipsoid SuperEllipsoid::operator*(const double& other) const{
		SuperEllipsoid result = *this;
		result*=other;
		return result;
	}

	Point3 SuperEllipsoid::tolocal(const Point3& point) const{
		Point3 result = _Q.rotate(point - _center);
		return result;
	}

	Point3 SuperEllipsoid::toglobal(const Point3& point) const{
		Point3 result = _Q.conj().rotate(point) + _center;
		return result;
	}

	void SuperEllipsoid::print(std::ostream &stream) const{
		stream << "SuperEllipsoid( r=(" << _r[0] << ", " << _r[1] << ", " << _r[2] << "), eps1=" << _eps1 << ", eps2=" << _eps2 << ", center=(" << _center[0] << ", " << _center[1] << ", " << _center[2] << "), Q=(" << _Q[0] << ", " << _Q[1] << ", " << _Q[2] << ", " << _Q[3] << ") )\n";
	}

	double SuperEllipsoid::levelval(const Point3& point) const{
		Point3 localpoint = tolocal(point);
		localpoint/=_r;
		localpoint*=localpoint;
		localpoint[0] = std::pow(localpoint[0], _POWERS[1]);
		localpoint[1] = std::pow(localpoint[1], _POWERS[1]);
		localpoint[2] = std::pow(localpoint[2], _POWERS[0]);
		return std::pow(localpoint[0]+localpoint[1], _POWERS[2]) + localpoint[2];
	}

	Point3 SuperEllipsoid::levelgrad(const Point3& point) const{
		return toglobal(local_levelgrad(tolocal(point)));
	}

	Point3 SuperEllipsoid::local_levelgrad(const Point3& point) const{
		Point3 localpoint = point;
		localpoint/=_r;

		double val1 = pow(localpoint[0],2.0/_eps1) + pow(localpoint[1],2.0/_eps2);
		val1 = (2.0/_eps1)*pow(val1, _POWERS[2]-1.0);

		Point3 grad = Point3();
		grad[0] = val1*pow(localpoint[0],(2.0-_eps2)/_eps2)/_r[0];
		grad[1] = val1*pow(localpoint[1],(2.0-_eps2)/_eps2)/_r[1];
		grad[2] = 2.0*pow(localpoint[2],(2.0-_eps1)/_eps1)/(_r[2]*_eps1);

		return grad;
	}

	Point3 SuperEllipsoid::support(const Point3& direction) const{
		Point3 d = tolocal(direction);
		


		//get angles
		double theta = atan2( sgn(d[1]) * pow(abs(_r[1]*d[1]), _INVPOWERS[0]), sgn(d[0]) * pow(abs(_r[0]*d[0]), _INVPOWERS[0]) );
		if (theta<0){
			theta += 6.283185307;
		}

		double C = _r[2]*d[2]*pow(abs(cos(theta)), 2.0-_eps2);
		double phi   = atan2( sgn(d[2]) * pow(abs(C), _INVPOWERS[1]), pow(abs(_r[0]*d[0]), _INVPOWERS[1]) );
		if (abs(phi)>1.570796327){
			if (phi<0){
				phi += 1.570796327;
			}
			phi -= 1.570796327;
		}


		std::cout << "theta= " << theta << "\tphi= " << phi << std::endl;

		//get local point
		Point3 point = Point3();

		double COS1 = cos(theta);
		double SIN1 = sin(theta);
		double COS2 = cos(phi);
		double SIN2 = sin(phi);

		double POW1 = pow(abs(COS2), _eps2);

		point[0] = sgn(COS1)*_r[0]*pow(abs(COS1),_eps1)*POW1;
		point[1] = sgn(SIN1)*_r[1]*pow(abs(SIN1),_eps1)*POW1;
		point[0] = sgn(SIN2)*_r[2]*pow(abs(SIN2),_eps2);

		std::cout << "point= ";
		toglobal(point).print(std::cout);
		std::cout << std::endl;

		return toglobal(point);

	}




	//SPHERE
	Point3 Sphere::center() const{
		return _center;
	}

	Sphere Sphere::copy() const{
		Sphere result = Sphere(_r, _center);
		return result;
	}

	Sphere* Sphere::operator+=(const Point3& other){
		_center+= other;
		return this;
	}

	Sphere Sphere::operator+(const Point3& other) const{
		Sphere result = copy();
		result+=other;
		return result;
	}

	Sphere* Sphere::operator*=(const Quaternion& other){
		return this;
	}

	Sphere Sphere::operator*(const Quaternion& other) const{
		Sphere result = copy();
		return result;
	}

	//scaling
	Sphere* Sphere::operator*=(const double& other){
		_r*=other;
		return this;
	}
	Sphere Sphere::operator*(const double& other) const{
		Sphere result = *this;
		result*=other;
		return result;
	}

	Point3 Sphere::tolocal(const Point3& point) const{
		Point3 result =point - _center;
		return result;
	}

	Point3 Sphere::toglobal(const Point3& point) const{
		Point3 result = point + _center;
		return result;
	}

	void Sphere::print(std::ostream &stream) const{
		stream << "Sphere( r=" << _r << ", center=(" << _center[0] << ", " << _center[1] << ", " << _center[2] << ") )\n";
	}

	double Sphere::levelval(const Point3& point) const{
		return tolocal(point).norm2()*_1r2;
	}

	Point3 Sphere::support(const Point3& direction) const{
		return _center + _r*direction.normalize();
	}

}
