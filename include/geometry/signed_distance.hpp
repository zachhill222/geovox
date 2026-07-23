#pragma once

#include "gutil.hpp"


namespace GV {

	template<typename T>
	struct SignedDistanceSpheres : public gutil::VolumeOctree<gutil::Sphere<3,T>> {
		using BASE = gutil::VolumeOctree<gutil::Sphere<3,T>>;

		using Sphere_t = typename BASE::value_type;
		using Point_t = typename BASE::point_type;
		using Box_t = typename BASE::box_type;

		using BASE::BASE;

		[[nodiscard]] T heaviside(const Point_t& point, const T eps) const noexcept {
			//return 0 if sgndist(point) < -eps (inside a particle)
			//return 1 if sgndist(point) > eps (outside all particles)
			assert(eps>T{0});

			const T sd = this->signed_distance(point);

			if (sd < -eps) {return 0;}
			else if (sd > eps) {return 1;}

			const T ratio = sd/eps;
			return T{0.5}*( T{1.0} + ratio + T{0.15915494309} * gutil::sin( T{3.14159265359}*ratio));
		}

		[[nodiscard]] T dirac(const Point_t& point, const T eps) const noexcept {
			//return 0 if |sgndist(point)| > eps (heaviside function is constant)
			assert(eps>0);

			const T sd = this->signed_distance(point);
			if ( gutil::abs(sd) > eps) {return 0;}

			const T ratio = T{1.0}/eps;
			return T{0.5}*ratio*( T{1.0} + gutil::cos( T{3.14159265359}*sd*ratio));
		}

	};


}



