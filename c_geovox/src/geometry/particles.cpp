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


	Point3 SuperEllipsoid::radius() const{
		return _r;
	}

	double SuperEllipsoid::eps1() const{
		return _eps1;
	}

	double SuperEllipsoid::eps2() const{
		return _eps2;
	}

	Quaternion SuperEllipsoid::quaternion() const{
		return _Q;
	}

	SuperEllipsoid SuperEllipsoid::copy() const{
		SuperEllipsoid result = SuperEllipsoid(_r, _eps1, _eps2, _center, _Q);
		return result;
	}

	Polytope SuperEllipsoid::bbox() const{
		Polytope box = Polytope(8);
		Box local_box = Box(-_r,_r);

		for (int i=0; i<8; i++){
			box.addpoint(toglobal(local_box[i]));
		}

		return box;
	}

	Box SuperEllipsoid::axis_alligned_bbox() const{
		Point3 low = Point3();
		Point3 high = Point3();
		Point3 support_point;

		//GET LOW CORNER
		support_point = support(Point3(-1,0,0));
		low[0] = support_point[0];
		support_point = support(Point3(0,-1,0));
		low[1] = support_point[1];
		support_point = support(Point3(0,0,-1));
		low[2] = support_point[2];

		//GET HIGH CORNER
		support_point = support(Point3(1,0,0));
		high[0] = support_point[0];
		support_point = support(Point3(0,1,0));
		high[1] = support_point[1];
		support_point = support(Point3(0,0,1));
		high[2] = support_point[2];

		Box box = Box(low, high);
		return box;
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
		//convert to local coordinates
		Point3 d = tolocal(direction);
		
		//get omega
		double x = sgn(d[1])*pow(abs(_r[1]*d[1]), _INVPOWERS[1]);
		double y = sgn(d[0])*pow(abs(_r[0]*d[0]), _INVPOWERS[1]);
		double omega = atan2(x,y); //in [-pi,pi]

		//get eta
		x = pow(abs(_r[0]*d[0]), _INVPOWERS[0]);
		y = sgn(d[2]) * pow( abs( _r[2]*d[2]*cos_pow(omega,2.0-_eps2) ) , _INVPOWERS[0]);
		double eta = atan2(x,y); //in [-pi/2,pi/2] because x >= 0

		//get normal in global coordinates
		Point3 result = parametric(eta, omega);
		return result;
	}

	Point3 SuperEllipsoid::parametric(const double eta, const double omega) const{
		//compute sines and cosines
		double C_eta = cos_pow(eta, _eps1);
		double S_eta = sin_pow(eta, _eps1);
		double C_omega = cos_pow(omega, _eps2);
		double S_omega = sin_pow(omega, _eps2);

		Point3 result = Point3(_r[0]*C_eta*C_omega, _r[1]*C_eta*S_omega, _r[2]*S_eta);
		return toglobal(result);
	}

	Point3 SuperEllipsoid::normal_parametric(const double eta, const double omega) const{
		//compute sines and cosines
		double C_eta = cos_pow(eta, 2.0-_eps1);
		double S_eta = sin_pow(eta, 2.0-_eps1);
		double C_omega = cos_pow(omega, 2.0-_eps2);
		double S_omega = sin_pow(omega, 2.0-_eps2);

		Point3 result = Point3(C_eta*C_omega/_r[0], C_eta*S_omega/_r[1], S_eta/_r[2]);
		return toglobal(result);
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





	double cos_pow(const double theta, const double eps){
		double C = cos(theta);
		if (C<0){
			return -pow(-C,eps);
		}
		else{
			return pow(C,eps);
		}
	}

	double sin_pow(const double theta, const double eps){
		double S = sin(theta);
		if (S<0){
			return -pow(-S,eps);
		}
		else{
			return pow(S,eps);
		}
	}

}
