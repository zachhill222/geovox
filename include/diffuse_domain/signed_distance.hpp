#pragma once

#include "gutil.hpp"

#include "diffuse_domain/periodic_octree.hpp"

#include <cmath>	//TODO: replace tanh with interpolation

namespace GV {

	template<gutil::IsReal T, uint8_t Period=0> requires(Period<8)
	struct SignedDistanceSpheres : public GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period> {
		using BASE = GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period>;

		using Particle_t 	= typename BASE::value_type;
		using Scalar_t		= typename BASE::scalar_type;
		using Point_t 		= typename BASE::point_type;
		using Box_t 		= typename BASE::box_type;

		using BASE::BASE;

		[[nodiscard]] T signed_distance(const Point_t& point) const noexcept {
			const size_t idx = this->find_nearest(point);
			GUTIL_ASSERT(idx< this->data_.size());
			return this->data_[idx].signed_distance(point);
		}


		[[nodiscard]] Point_t grad_signed_distance(const Point_t& point) const noexcept {
			const size_t idx = this->find_nearest(point);
			GUTIL_ASSERT(idx< this->data_.size());
			return this->data_[idx].grad_signed_distance_impl(point);
		}

		[[nodiscard]] T heaviside(const Point_t& point, T eps) const noexcept {
			//return 0 if sgndist(point) < -eps (inside a particle)
			//return 1 if sgndist(point) > eps (outside all particles)
			GUTIL_ASSERT(eps>T{0});

			const T sd = signed_distance(point);

			if (sd < -eps) {return 0;}
			else if (sd > eps) {return 1;}

			const T ratio = sd/eps;
			return T{0.5}*( T{1.0} + ratio + T{0.15915494309} * gutil::sin( T{3.14159265359}*ratio));
		}

		[[nodiscard]] T dirac(const Point_t& point, T eps) const noexcept {
			//return 0 if |sgndist(point)| > eps (heaviside function is constant)
			GUTIL_ASSERT(eps>T{0});

			const T sd = signed_distance(point);
			if ( gutil::abs(sd) > eps) {return 0;}

			const T ratio = T{1.0}/eps;
			return T{0.5}*ratio*( T{1.0} + gutil::cos( T{3.14159265359}*sd*ratio));
		}


		[[nodiscard]] static T fast_tanh(T x) noexcept {
			//TODO: build interpolation table
			return std::tanh(x);
		}

		[[nodiscard]] T heaviside_tanh(const Point_t& point, T eps) const noexcept {
			GUTIL_ASSERT(eps>T{0});
			return T{0.5} * (T{1} - fast_tanh(T{3}*signed_distance(point)/eps)); //scale by 3, see https://arxiv.org/pdf/2509.25115v1
		}

		[[nodiscard]] Point_t heaviside_tanh_grad(const Point_t& point, T eps) const noexcept {
			GUTIL_ASSERT(eps>T{0});
			const size_t idx = this->find_nearest(point);
			GUTIL_ASSERT(idx<this->data_.size());
			const Particle_t& nearest = this->data_[idx];

			const T sdf = nearest.signed_distance(point);
			const T eps_inv = T{3}/eps;	//scale by 3, see https://arxiv.org/pdf/2509.25115v1

			const T heavi = T{0.5} * (T{1} - fast_tanh(sdf * eps_inv));

			return T{-2} * eps_inv * heavi * (T{1} - heavi) * (nearest.grad_signed_distance(point));
		}
	};
}



