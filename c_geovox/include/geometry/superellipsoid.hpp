#ifndef SUPERELLIPSOID_H
#define SUPERELLIPSOID_H

#include "util.hpp"
#include <cmath>

namespace GeoVox::geometry{
	using Point3 = GeoVox::util::Point<3>;
	
	class SuperEllipsoid{
	public:
		SuperEllipsoid(): _r(GeoVox::util::Point3(1,1,1)), _eps1(1), _eps2(1), _center(GeoVox::util::Point3(0,0,0)), _Q(GeoVox::util::Quaternion(1,0,0,0)) {
			_POWERS[0] = 1.0;
			_POWERS[1] = 1.0;
			_POWERS[2] = 1.0;
		}
		
		SuperEllipsoid(const GeoVox::util::Point3& r, const double& eps1, const double& eps2, const GeoVox::util::Point3& center, const GeoVox::util::Quaternion Q): _r(r), _eps1(eps1), _eps2(eps2), _center(center), _Q(Q) {
			_POWERS[0] = 1.0/eps1;
			_POWERS[1] = 1.0/eps2;
			_POWERS[2] = eps2/eps1;
		}

		//copy
		SuperEllipsoid copy() const;

		//shifting
		SuperEllipsoid* operator+=(const GeoVox::util::Point3& other);
		SuperEllipsoid  operator+(const GeoVox::util::Point3& other) const;

		//rotating
		SuperEllipsoid* operator*=(const GeoVox::util::Quaternion& other);
		SuperEllipsoid  operator*(const GeoVox::util::Quaternion& other) const;

		//coordinate system changes
		GeoVox::util::Point3 tolocal(const GeoVox::util::Point3& point) const;
		GeoVox::util::Point3 toglobal(const GeoVox::util::Point3& point) const;
		
		//print
		void print(std::ostream &stream) const;

		//level set evaluation
		double levelval(const GeoVox::util::Point3& point) const;

	private:
		GeoVox::util::Point3 _r;
		double _eps1;
		double _eps2;
		GeoVox::util::Point3 _center;
		GeoVox::util::Quaternion _Q;
		double _POWERS[3];
	};
}



#endif