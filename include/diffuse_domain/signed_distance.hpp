#pragma once

#include "gutil.hpp"

#include "diffuse_domain/periodic_octree.hpp"

#include <cmath>	//TODO: replace tanh with interpolation

namespace GV {

	template<gutil::IsReal T, uint8_t Period=0, bool Interior=false, int HeavisideFunc=0> requires(Period<8)
	struct SignedDistanceSpheres : public GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period> {
		using BASE = GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period>;

		using Particle_t	= typename BASE::value_type;
		using Scalar_t		= typename BASE::scalar_type;
		using Point_t 		= typename BASE::point_type;
		using Box_t 		= typename BASE::box_type;

		using BASE::BASE;
		using BASE::data_;


		/////////////////////////////////////////////////////////////////////////////////
		/// Static methods for expensive functions.
		/// TODO: implement lookup tables.
		/////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] static T fast_cos(T x)  noexcept { return std::cos(x);  }
		[[nodiscard]] static T fast_sin(T x)  noexcept { return std::sin(x);  }
		[[nodiscard]] static T fast_tanh(T x) noexcept { return std::tanh(x); }
		
		[[nodiscard]] static T heaviside_of_sdf(T sd, T, T three_over_eps) noexcept requires(HeavisideFunc==0) {
			return T{0.5} * (T{1} - fast_tanh(sd*three_over_eps));
		}
		[[nodiscard]] static T heaviside_grad_coef_of_sdf(T sd, T, T three_over_eps) noexcept requires(HeavisideFunc==0) {
			//multiply with grad_sdf to get the gradient of the heaviside
			const T phi = heaviside_of_sdf(sd, T{0}, three_over_eps);
			return T{2} * three_over_eps * phi * (phi - T{1});
		}

		[[nodiscard]] static T heaviside_of_sdf(T sd, T eps, T neg_one_over_eps) noexcept requires(HeavisideFunc==1) {
			if (sd < -eps) {return T{1};}
			if (sd >  eps) {return T{0};}
			const T r = sd*neg_one_over_eps;
			return  T{0.5}*( T{1} + r + T{0.15915494309} * fast_sin( T{3.14159265359}*r ));
		}
		[[nodiscard]] static T heaviside_grad_coef_of_sdf(T sd, T eps, T neg_one_over_eps) noexcept requires(HeavisideFunc==1) {
			//multiply with grad_sdf to get the gradient of the heaviside
			if ( gutil::abs(sd) > eps) {return T{0};}
			return T{0.5}*neg_one_over_eps*( T{1} + fast_cos( T{3.14159265359}*sd*neg_one_over_eps) );
		}

		[[nodiscard]] static T compute_reciprocal_coef(T eps) noexcept {
			if constexpr (HeavisideFunc==0) {return T{3}/eps;}
			else if constexpr (HeavisideFunc==1) {return T{-1}/eps;}
			else {
				GUTIL_ABORT("unkown HeavisideFunc option");
				return T{0};
			}
		}

		/////////////////////////////////////////////////////////////////////////////////
		/// Compute the signed distance and/or its gradient.
		/////////////////////////////////////////////////////////////////////////////////
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


		/////////////////////////////////////////////////////////////////////////////////
		/// Define heaviside functions for HeavisideFunc=0 (use tanh, no compact support).
		/////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] T heaviside(const Point_t& point, T eps) const noexcept {
			GUTIL_ASSERT(eps>T{0});
			const T c = compute_reciprocal_coef(eps);
			return heaviside_of_sdf(signed_distance(point), eps, c);
		}

		[[nodiscard]] Point_t heaviside_grad(const Point_t& point, T eps) const noexcept {
			GUTIL_ASSERT(eps>T{0});
			const size_t idx = this->find_nearest(point);
			
			GUTIL_ASSERT(idx<data_.size());
			const Particle_t& nearest = data_[idx];

			const T sdf = nearest.signed_distance(point);
			const Point_t sdf_grad = nearest.grad_signed_distance(point);

			const T c = compute_reciprocal_coef(eps);
			if constexpr (Interior) {
				return heaviside_grad_coef_of_sdf(sdf, eps, c) * sdf_grad;
			}
			else {
				return -heaviside_grad_coef_of_sdf(-sdf, eps, c) * sdf_grad;
			}
		}

		static void heaviside(std::span<T> result, std::span<const T> sdf, const T eps) noexcept {
			GUTIL_ASSERT(result.size()==sdf.size());
			GUTIL_ASSERT(eps>0);
			const size_t N = sdf.size();
			
			const T c = compute_reciprocal_coef(eps);
			for (size_t i=0; i<N; ++i) {
				result[i] = heaviside_of_sdf(sdf[i], eps, c);
			}
		}

		static void heaviside_grad(std::span<T> result, std::span<const T> sdf, std::span<const T> sdf_grad, const T eps) noexcept {
			GUTIL_ASSERT(result.size()==3*sdf.size());
			GUTIL_ASSERT(result.size()==sdf_grad.size());
			GUTIL_ASSERT(eps>0);

			const size_t N = sdf.size();
			const T c = compute_reciprocal_coef(eps);
			for (size_t i=0; i<N; ++i) {
				const T coef  = heaviside_grad_coef_of_sdf(sdf, eps, c);

				result[i]     = coef * sdf_grad[i];
				result[i+N]   = coef * sdf_grad[i+N];
				result[i+2*N] = coef * sdf_grad[i+2*N];
			}
		}
	};
}



