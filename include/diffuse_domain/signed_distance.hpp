#pragma once

#include "gutil.hpp"

#include "diffuse_domain/periodic_octree.hpp"

#include <cmath>	//TODO: replace tanh with interpolation

namespace GV {

	template<gutil::IsReal T, uint8_t Period=0, bool Interior=false> requires(Period<8)
	struct SignedDistanceSpheres : public GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period> {
		using BASE = GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period>;

		using Particle_t 	= typename BASE::value_type;
		using Scalar_t		= typename BASE::scalar_type;
		using Point_t 		= typename BASE::point_type;
		using Box_t 		= typename BASE::box_type;

		using BASE::BASE;
		using BASE::data_;

		[[nodiscard]] T signed_distance(const Point_t& point) const noexcept {
			const size_t idx = this->find_nearest(point);
			GUTIL_ASSERT(idx< data_.size());
			if constexpr (Interior) {
				return data_[idx].signed_distance(point);
			}
			else {
				return -data_[idx].signed_distance(point);
			}
		}
		

		[[nodiscard]] Point_t grad_signed_distance(const Point_t& point) const noexcept {
			const size_t idx = this->find_nearest(point);
			GUTIL_ASSERT(idx< data_.size());
			if constexpr (Interior) {
				return data_[idx].grad_signed_distance(point);
			}
			else {
				return -data_[idx].grad_signed_distance(point);
			}
		}

		void signed_distance(std::span<T> sdf, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
			GUTIL_ASSERT(sdf.size()==x.size() && x.size()==y.size() && y.size()==z.size());
			const size_t N = x.size();
			for (size_t i=0; i<N; ++i) {
				const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
				GUTIL_ASSERT(idx<data_.size());
				if constexpr (Interior) {
					sdf[i] = data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
				}
				else {
					sdf[i] = -data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
				}
			}
		}

		void grad_signed_distance(std::span<T> grad, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
			GUTIL_ASSERT(grad.size()==3*x.size() && x.size()==y.size() && y.size()==z.size());
			const size_t N = x.size();
			for (size_t i=0; i<N; ++i) {
				const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
				GUTIL_ASSERT(idx<data_.size());
				Point_t gr;

				if constexpr (Interior) {
					gr = data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
				}
				else {
					gr = -data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
				}
				
				grad[i] = gr[0];
				grad[i+N] = gr[1];
				grad[i+2*N] = gr[2];
			}
		}

		void signed_distance(std::span<T> sdf, std::span<T> grad, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
			GUTIL_ASSERT(3*sdf.size()==grad.size() && sdf.size()==x.size() && x.size()==y.size() && y.size()==z.size());
			const size_t N = x.size();
			for (size_t i=0; i<N; ++i) {
				const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
				GUTIL_ASSERT(idx<data_.size());
				Point_t gr;

				if constexpr (Interior) {
					sdf[i] = data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
					gr = data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
				}
				else {
					sdf[i] = -data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
					gr = -data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
				}
				
				grad[i] = gr[0];
				grad[i+N] = gr[1];
				grad[i+2*N] = gr[2];
			}
		}


		////////////////////////////////////////////////////////
		/// Using smoothed heaviside sin/cos (compact support)
		////////////////////////////////////////////////////////
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



		////////////////////////////////////////////////////////
		/// Using smoothed heaviside tanh (no compact support)
		////////////////////////////////////////////////////////
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
			GUTIL_ASSERT(idx<data_.size());
			const Particle_t& nearest = data_[idx];

			const T sdf = nearest.signed_distance(point);
			const T eps_inv = T{3}/eps;	//scale by 3, see https://arxiv.org/pdf/2509.25115v1

			const T heavi = T{0.5} * (T{1} - fast_tanh(sdf * eps_inv));

			return T{-2} * eps_inv * heavi * (T{1} - heavi) * (nearest.grad_signed_distance(point));
		}

		static void heaviside_tanh(std::span<T> result, std::span<const T> sdf, const T eps) noexcept {
			GUTIL_ASSERT(result.size()==sdf.size());
			GUTIL_ASSERT(eps>0);
			const size_t N = sdf.size();
			
			const T scale = T{3}/eps;
			// GUTIL_SIMD() //TODO: make interpolation table lookup simd-compatible
			for (size_t i=0; i<N; ++i) {
				result[i] = T{0.5} * (T{1} - fast_tanh(scale * sdf[i]));
			}
		}

		static void heaviside_tanh_grad(std::span<T> result, std::span<const T> sdf, std::span<const T> sdf_grad, const T eps) noexcept {
		GUTIL_ASSERT(result.size()==3*sdf.size());
		GUTIL_ASSERT(result.size()==sdf_grad.size());
		GUTIL_ASSERT(eps>0);

		const size_t N = sdf.size();
		const T scale = T{3}/eps;
		for (size_t i=0; i<N; ++i) {
			const T phi = T{0.5} * (T{1} - fast_tanh(scale * sdf[i]));
			const T coef = -(T{6}/eps) * phi * (T{1} - phi);

			result[i]     = coef * sdf_grad[i];
			result[i+N]   = coef * sdf_grad[i+N];
			result[i+2*N] = coef * sdf_grad[i+2*N];
		}
	}
	};
}



