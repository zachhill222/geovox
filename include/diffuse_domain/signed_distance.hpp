#pragma once

#include "gutil.hpp"

#include "diffuse_domain/periodic_octree.hpp"

#include <cmath>	//TODO: replace tanh with interpolation

namespace GV {


	/////////////////////////////////////////////////////////////////////////////////
	/// Standard functions that can be used for building diffuse domain functions.
	/// Define by their normalized input r=sd/eps. Note that the user must apply the chain
	/// rule for the derivatives. (i.e, multiply the result by 1/eps).
	///
	/// Note that the sign convention is sdf(x)<0 on the interior of the domain and
	/// heaviside(sdf(x)/eps) is approximately the characteristic function on the domain
	/// (i.e., 1 on the interior and 0 on the exterior).
	/////////////////////////////////////////////////////////////////////////////////
	namespace diffuse {
		/// Tanh
		GUTIL_DECLARE_SIMD()
		template<typename T>
		[[nodiscard]] inline constexpr T heaviside_tanh(T r) noexcept {
			return T{0.5} * (T{1} - std::tanh(r*T{3}));
		}

		GUTIL_DECLARE_SIMD()
		template<typename T>
		[[nodiscard]] inline constexpr T delta_tanh(T r) noexcept {
			const T th = std::tanh(r*T{3});
			return T{-1.5} * (T{1} - th*th);
		}


		/// Compact sin/cos
		GUTIL_DECLARE_SIMD()
		template<typename T>
		[[nodiscard]] inline constexpr T heaviside_compact_trig(T r) noexcept {
			//1/2 - r/2 - 1/2pi * sin(pi*r)
			if (r < -T{1}) {return T{1};}
			if (r >  T{1}) {return T{0};}
			return  T{0.5}*( T{1} - r - T{0.3183098862} * std::sin( T{3.14159265359}*r ));
		}

		GUTIL_DECLARE_SIMD()
		template<typename T>
		[[nodiscard]] inline constexpr T delta_compact_trig(T r) noexcept {
			if (r < -T{1} || r > T{1}) {return T{0};}
			return  T{-0.5} * (T{1} + std::cos(T{3.14159265359}*r));
		}


		/// Compact Hermite interpolation. These are cheap to evaluate, so we write them in a more simd friendly way.
		GUTIL_DECLARE_SIMD()
		template<typename T>
		[[nodiscard]] inline constexpr T heaviside_compact_poly5(T r) noexcept {
			//p(r) = 1/2 - 15/16*r + 5/8*r^3 - 3/16*r^5
			r = std::clamp(r,T{-1}, T{1});
			const T r2 = r*r;
			return  T{0.5} + r*( T{-0.9375} + r2*(T{0.625} - r2*T{0.1875}));
		}

		GUTIL_DECLARE_SIMD()
		template<typename T>
		[[nodiscard]] inline constexpr T delta_compact_poly5(T r) noexcept {
			//p'(r) = -15/16 + 15/8*r^2 - 15/16*r^4 = -15/16 * (1 - 2r^2 + r^4) = -15/16 * (1-r^2)^2
			r = std::clamp(r,T{-1}, T{1});
			const T w = T{1}-r*r;
			return  T{-0.9375}*w*w;
		}
	}


	/////////////////////////////////////////////////////////////////////////////////
	/// A class to combine the heaviside and delta approximations. When using more expensive
	/// computations (e.g, tanh), it may be useful to have variants with cached results.
	///
	/// The individual HeavisideFn and DeltaFn callables must be stateless.
	/////////////////////////////////////////////////////////////////////////////////
	template<typename T, auto HeavisideFn, auto DeltaFn, auto JointFn=nullptr>
	struct DiffuseDomainPhi {
		static constexpr bool HAS_HEAVISIDE = std::is_invocable_r_v<T,decltype(HeavisideFn),T>;
		static constexpr bool HAS_DELTA = std::is_invocable_r_v<T,decltype(DeltaFn),T>;
		static constexpr bool HAS_JOINT = std::is_invocable_r_v<void,decltype(JointFn),T*,T*,T>;

		static_assert(HAS_HEAVISIDE || HAS_JOINT);
		
		T eps{1};
		T eps_inv{1};
		DiffuseDomainPhi(T ep=1) : eps{ep}, eps_inv{T{1}/ep} {}

		constexpr void set_eps(T ep) noexcept {
			eps = ep;
			eps_inv = T{1}/ep;
		}

		GUTIL_DECLARE_SIMD(notinbranch uniform(this))
		[[nodiscard]] constexpr T eval_heaviside(T sd) const noexcept {
			GUTIL_ASSERT(eps>T{0});
			if constexpr (HAS_HEAVISIDE) {
				return HeavisideFn(sd*eps_inv);
			}
			else {
				T h{0}, d{0};
				JointFn(&h,&d,sd);
				return h;
			}
		}

		GUTIL_DECLARE_SIMD(notinbranch uniform(this))
		[[nodiscard]] constexpr T eval_delta(T sd) const noexcept requires (HAS_DELTA || HAS_JOINT) {
			GUTIL_ASSERT(eps>T{0});
			if constexpr (HAS_DELTA) {
				return eps_inv*DeltaFn(sd*eps_inv);
			}
			else {
				T h{0}, d{0};
				JointFn(&h,&d,sd);
				return eps_inv*d;
			}
		}

		GUTIL_DECLARE_SIMD(notinbranch uniform(this) linear(h:1) linear(d:1))
		constexpr void eval_heaviside_delta(T* GUTIL_RESTRICT h, T* GUTIL_RESTRICT d, T sd) const noexcept requires(HAS_DELTA || HAS_JOINT) {
			GUTIL_ASSERT(eps>T{0});
			if constexpr (HAS_JOINT) {
				JointFn(h,d,sd*eps_inv);
				*d *=eps_inv;
			}
			else {
				sd *= eps_inv;
				*h  = HeavisideFn(sd);
				*d  = eps_inv*DeltaFn(sd);
			}
		}
	};

	using DefaultDiffuseDomainPhi = DiffuseDomainPhi<double, diffuse::heaviside_compact_poly5<double>, diffuse::delta_compact_poly5<double>>;



	/////////////////////////////////////////////////////////////////////////////////
	/// A class to take a signed distance function and compute the phase field function
	/// that represents the diffuse domain. 
	/////////////////////////////////////////////////////////////////////////////////
	template<typename T, typename Sdf=std::nullptr_t, typename Grad=std::nullptr_t, typename JointEval=std::nullptr_t, typename Phi=DefaultDiffuseDomainPhi>
	struct SDFunction {
		using Scalar_t = T;

		static constexpr bool HAS_SDF = !std::same_as<Sdf,std::nullptr_t>;
		static constexpr bool HAS_GRAD = !std::same_as<Grad,std::nullptr_t>;
		static constexpr bool HAS_JOINT = !std::same_as<JointEval,std::nullptr_t>;

		static_assert( HAS_SDF || HAS_JOINT );
		static_assert( !HAS_GRAD || Phi::HAS_JOINT || Phi::HAS_DELTA );
		static_assert( !HAS_SDF || std::is_invocable_r_v<void,Sdf,std::span<T>,std::span<const T>,std::span<const T>,std::span<const T>> );
		static_assert( !HAS_GRAD || std::is_invocable_r_v<void,Grad,std::span<T>,std::span<const T>,std::span<const T>,std::span<const T>> );
		static_assert( !HAS_JOINT || std::is_invocable_r_v<void,JointEval,std::span<T>,std::span<T>,std::span<const T>,std::span<const T>,std::span<const T>> );

		Sdf       sdf{};
		Grad      grad{};
		Phi       phi{};
		JointEval jeval{};

		SDFunction(Sdf s, Grad g, JointEval j) : sdf{std::move(s)}, grad{std::move(g)}, jeval{std::move(j)} {}
		
		constexpr void set_eps(T eps) noexcept {
			phi.set_eps(eps);
		}
		
		constexpr void set_sdf_eval(Sdf&& s) noexcept requires(HAS_SDF) {
			sdf = std::forward<Sdf>(s);
		}
		
		constexpr void set_grad_eval(Grad&& g) noexcept requires(HAS_GRAD) {
			grad = std::forward<Grad>(g);
		}

		constexpr void set_joint_eval(JointEval&& g) noexcept requires(HAS_JOINT) {
			jeval = std::forward<JointEval>(g);
		}


		////////////////////////////////////////////////////////////////////
		/// Heaviside and Delta approximations
		////////////////////////////////////////////////////////////////////
		template<size_t N>
		constexpr void eval_heaviside(std::span<T,N> result, std::span<const T,N> x, std::span<const T,N> y, std::span<const T,N> z) const noexcept {
			if constexpr (HAS_SDF) {sdf(result,x,y,z);}
			else {
				static thread_local T extra[3*N] {};
				jeval(result, std::span<T,3*N>(extra),x,y,z);
			}

			GUTIL_SIMD()
			for (size_t i=0; i<N; ++i) {
				result[i] = phi.eval_heaviside(result[i]);
			}
		}

		template<size_t N>
		constexpr void eval_grad(std::span<T,3*N> result, std::span<const T,N> x, std::span<const T,N> y, std::span<const T,N> z) const noexcept requires(HAS_GRAD || HAS_JOINT) {
			static thread_local T delta[N] {};
			if constexpr (HAS_JOINT) {jeval(std::span<T,N>(delta),result,x,y,z);}
			else {
				sdf(std::span<T,N>(delta),x,y,z);
				grad(result,x,y,z);
			}

			GUTIL_SIMD()
			for (size_t i=0; i<N; ++i) {
				const T d = phi.eval_delta(delta[i]);

				result[i]     *= d;
				result[i+N]   *= d;
				result[i+2*N] *= d;
			}
		}

		template<size_t N>
		constexpr void eval_heaviside_grad(std::span<T,N> h_res, std::span<T,3*N> g_res, std::span<const T,N> x, std::span<const T,N> y, std::span<const T,N> z) const noexcept requires(HAS_GRAD || HAS_JOINT) {
			if constexpr (HAS_JOINT) {jeval(h_res,g_res,x,y,z);}
			else {
				sdf(h_res,x,y,z);
				grad(g_res,x,y,z);
			}

			GUTIL_SIMD()
			for (size_t i=0; i<N; ++i) {
				//Note h_res stores the sdf values
				T d{0};
				phi.eval_heaviside_delta(&h_res[i], &d, h_res[i]);
				
				g_res[i]     *= d;
				g_res[i+N]   *= d;
				g_res[i+2*N] *= d;
			}
		}
	};



	/////////////////////////////////////////////////////////////////////////////////
	/// Octree-backed signed distance function over an assembly of spheres.
	///
	/// This class is responsible ONLY for the geometric signed-distance query
	/// (and its gradient). Sign convention: sdf(x)<0 on the interior of the domain,
	/// matching the convention used throughout the diffuse-domain machinery.
	/// It knows nothing about eps or which mollifier is used -- that composition
	/// happens externally via DiffuseDomainPhi/SDFunction (see make_sdfunction below).
	/////////////////////////////////////////////////////////////////////////////////
	template<gutil::IsReal T, uint8_t Period=0, bool Interior=false> requires(Period<8)
	struct SignedDistanceSpheres : public GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period> {
		using BASE = GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period>;

		using Particle_t	= typename BASE::value_type;
		using Scalar_t		= typename BASE::scalar_type;
		using Point_t 		= typename BASE::point_type;
		using Box_t 		= typename BASE::box_type;

		using BASE::BASE;
		using BASE::data_;

		/////////////////////////////////////////////////////////////////////////////////
		/// Single-point queries.
		/////////////////////////////////////////////////////////////////////////////////
		[[nodiscard]] T signed_distance(const Point_t& point) const noexcept {
			const size_t idx = this->find_nearest(point);
			GUTIL_ASSERT(idx< data_.size());
			if constexpr (Interior) { return data_[idx].signed_distance(point); }
			else                    { return -data_[idx].signed_distance(point); }
		}

		[[nodiscard]] Point_t grad_signed_distance(const Point_t& point) const noexcept {
			const size_t idx = this->find_nearest(point);
			GUTIL_ASSERT(idx< data_.size());
			if constexpr (Interior) { return data_[idx].grad_signed_distance(point); }
			else                    { return -data_[idx].grad_signed_distance(point); }
		}

		/////////////////////////////////////////////////////////////////////////////////
		/// Bulk queries. Signatures intentionally match SDFunction's Sdf / Grad /
		/// JointEval requirements, so an instance of this class can be bound directly.
		/////////////////////////////////////////////////////////////////////////////////
		void signed_distance(std::span<T> sdf, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
			GUTIL_ASSERT(sdf.size()==x.size() && x.size()==y.size() && y.size()==z.size());
			const size_t N = x.size();
			for (size_t i=0; i<N; ++i) {
				const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
				GUTIL_ASSERT(idx<data_.size());
				if constexpr (Interior) { sdf[i] = data_[idx].signed_distance(Point_t{x[i],y[i],z[i]}); }
				else                    { sdf[i] = -data_[idx].signed_distance(Point_t{x[i],y[i],z[i]}); }
			}
		}

		void grad_signed_distance(std::span<T> grad, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
			GUTIL_ASSERT(grad.size()==3*x.size() && x.size()==y.size() && y.size()==z.size());
			const size_t N = x.size();
			for (size_t i=0; i<N; ++i) {
				const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
				GUTIL_ASSERT(idx<data_.size());
				Point_t gr;
				if constexpr (Interior) { gr = data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]}); }
				else                    { gr = -data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]}); }
				grad[i] = gr[0]; grad[i+N] = gr[1]; grad[i+2*N] = gr[2];
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
				grad[i] = gr[0]; grad[i+N] = gr[1]; grad[i+2*N] = gr[2];
			}
		}
	};


	/////////////////////////////////////////////////////////////////////////////////
	/// Convenience factory: captures the octree by reference in lambdas and constructs
	/// an SDFunction from that.
	/////////////////////////////////////////////////////////////////////////////////
	template<gutil::IsReal T, uint8_t Period, bool Interior, typename Phi=DefaultDiffuseDomainPhi<T>>
	[[nodiscard]] auto make_sdfunction(SignedDistanceSpheres<T,Period,Interior>& assembly, T eps) {
		auto sdf_eval = [&assembly](std::span<T> sdf, std::span<const T> x, std::span<const T> y, std::span<const T> z) {
			assembly.signed_distance(sdf,x,y,z);
		};
		auto grad_eval = [&assembly](std::span<T> grad, std::span<const T> x, std::span<const T> y, std::span<const T> z) {
			assembly.grad_signed_distance(grad,x,y,z);
		};
		auto joint_eval = [&assembly](std::span<T> sdf, std::span<T> grad, std::span<const T> x, std::span<const T> y, std::span<const T> z) {
			assembly.signed_distance(sdf,grad,x,y,z);
		};

		SDFunction<T, decltype(sdf_eval), decltype(grad_eval), decltype(joint_eval), Phi>
		sd_fn{std::move(sdf_eval), std::move(grad_eval), std::move(joint_eval)};
		
		sd_fn.set_eps(eps);
		return sd_fn;
	}
}














	// template<gutil::IsReal T, uint8_t Period=0, bool Interior=false, int HeavisideFunc=0> requires(Period<8)
	// struct SignedDistanceSpheres : public GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period> {
	// 	using BASE = GV::PeriodicVolumeOctree<gutil::Sphere<3,T>, Period>;

	// 	using Particle_t	= typename BASE::value_type;
	// 	using Scalar_t		= typename BASE::scalar_type;
	// 	using Point_t 		= typename BASE::point_type;
	// 	using Box_t 		= typename BASE::box_type;

	// 	using BASE::BASE;
	// 	using BASE::data_;


	// 	/////////////////////////////////////////////////////////////////////////////////
	// 	/// Static methods for expensive functions.
	// 	/// TODO: implement lookup tables.
	// 	/////////////////////////////////////////////////////////////////////////////////
	// 	[[nodiscard]] static T fast_cos(T x)  noexcept { return std::cos(x);  }
	// 	[[nodiscard]] static T fast_sin(T x)  noexcept { return std::sin(x);  }
	// 	[[nodiscard]] static T fast_tanh(T x) noexcept { return std::tanh(x); }
		
	// 	[[nodiscard]] static T heaviside_of_sdf(T sd, T, T three_over_eps) noexcept requires(HeavisideFunc==0) {
	// 		return T{0.5} * (T{1} - fast_tanh(sd*three_over_eps));
	// 	}
	// 	[[nodiscard]] static T heaviside_grad_coef_of_sdf(T sd, T, T three_over_eps) noexcept requires(HeavisideFunc==0) {
	// 		//multiply with grad_sdf to get the gradient of the heaviside
	// 		const T phi = heaviside_of_sdf(sd, T{0}, three_over_eps);
	// 		return T{2} * three_over_eps * phi * (phi - T{1});
	// 	}

	// 	[[nodiscard]] static T heaviside_of_sdf(T sd, T eps, T neg_one_over_eps) noexcept requires(HeavisideFunc==1) {
	// 		if (sd < -eps) {return T{1};}
	// 		if (sd >  eps) {return T{0};}
	// 		const T r = sd*neg_one_over_eps;
	// 		return  T{0.5}*( T{1} + r + T{0.15915494309} * fast_sin( T{3.14159265359}*r ));
	// 	}
	// 	[[nodiscard]] static T heaviside_grad_coef_of_sdf(T sd, T eps, T neg_one_over_eps) noexcept requires(HeavisideFunc==1) {
	// 		//multiply with grad_sdf to get the gradient of the heaviside
	// 		if ( gutil::abs(sd) > eps) {return T{0};}
	// 		return T{0.5}*neg_one_over_eps*( T{1} + fast_cos( T{3.14159265359}*sd*neg_one_over_eps) );
	// 	}

	// 	[[nodiscard]] static T compute_reciprocal_coef(T eps) noexcept {
	// 		if constexpr (HeavisideFunc==0) {return T{3}/eps;}
	// 		else if constexpr (HeavisideFunc==1) {return T{-1}/eps;}
	// 		else {
	// 			GUTIL_ABORT("unkown HeavisideFunc option");
	// 			return T{0};
	// 		}
	// 	}

	// 	/////////////////////////////////////////////////////////////////////////////////
	// 	/// Compute the signed distance and/or its gradient.
	// 	/////////////////////////////////////////////////////////////////////////////////
	// 	[[nodiscard]] T signed_distance(const Point_t& point) const noexcept {
	// 		const size_t idx = this->find_nearest(point);
	// 		GUTIL_ASSERT(idx< data_.size());
	// 		if constexpr (Interior) {
	// 			return data_[idx].signed_distance(point);
	// 		}
	// 		else {
	// 			return -data_[idx].signed_distance(point);
	// 		}
	// 	}
		

	// 	[[nodiscard]] Point_t grad_signed_distance(const Point_t& point) const noexcept {
	// 		const size_t idx = this->find_nearest(point);
	// 		GUTIL_ASSERT(idx< data_.size());
	// 		if constexpr (Interior) {
	// 			return data_[idx].grad_signed_distance(point);
	// 		}
	// 		else {
	// 			return -data_[idx].grad_signed_distance(point);
	// 		}
	// 	}

	// 	void signed_distance(std::span<T> sdf, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
	// 		GUTIL_ASSERT(sdf.size()==x.size() && x.size()==y.size() && y.size()==z.size());
	// 		const size_t N = x.size();
	// 		for (size_t i=0; i<N; ++i) {
	// 			const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
	// 			GUTIL_ASSERT(idx<data_.size());
	// 			if constexpr (Interior) {
	// 				sdf[i] = data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
	// 			}
	// 			else {
	// 				sdf[i] = -data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
	// 			}
	// 		}
	// 	}

	// 	void grad_signed_distance(std::span<T> grad, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
	// 		GUTIL_ASSERT(grad.size()==3*x.size() && x.size()==y.size() && y.size()==z.size());
	// 		const size_t N = x.size();
	// 		for (size_t i=0; i<N; ++i) {
	// 			const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
	// 			GUTIL_ASSERT(idx<data_.size());
	// 			Point_t gr;

	// 			if constexpr (Interior) {
	// 				gr = data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
	// 			}
	// 			else {
	// 				gr = -data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
	// 			}
				
	// 			grad[i] = gr[0];
	// 			grad[i+N] = gr[1];
	// 			grad[i+2*N] = gr[2];
	// 		}
	// 	}

	// 	void signed_distance(std::span<T> sdf, std::span<T> grad, std::span<const T> x, std::span<const T> y, std::span<const T> z) const noexcept {
	// 		GUTIL_ASSERT(3*sdf.size()==grad.size() && sdf.size()==x.size() && x.size()==y.size() && y.size()==z.size());
	// 		const size_t N = x.size();
	// 		for (size_t i=0; i<N; ++i) {
	// 			const size_t idx = this->find_nearest(Point_t{x[i],y[i],z[i]});
	// 			GUTIL_ASSERT(idx<data_.size());
	// 			Point_t gr;

	// 			if constexpr (Interior) {
	// 				sdf[i] = data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
	// 				gr = data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
	// 			}
	// 			else {
	// 				sdf[i] = -data_[idx].signed_distance(Point_t{x[i],y[i],z[i]});
	// 				gr = -data_[idx].grad_signed_distance(Point_t{x[i],y[i],z[i]});
	// 			}
				
	// 			grad[i] = gr[0];
	// 			grad[i+N] = gr[1];
	// 			grad[i+2*N] = gr[2];
	// 		}
	// 	}


	// 	/////////////////////////////////////////////////////////////////////////////////
	// 	/// Define heaviside functions for HeavisideFunc=0 (use tanh, no compact support).
	// 	/////////////////////////////////////////////////////////////////////////////////
	// 	[[nodiscard]] T heaviside(const Point_t& point, T eps) const noexcept {
	// 		GUTIL_ASSERT(eps>T{0});
	// 		const T c = compute_reciprocal_coef(eps);
	// 		return heaviside_of_sdf(signed_distance(point), eps, c);
	// 	}

	// 	[[nodiscard]] Point_t heaviside_grad(const Point_t& point, T eps) const noexcept {
	// 		GUTIL_ASSERT(eps>T{0});
	// 		const size_t idx = this->find_nearest(point);
			
	// 		GUTIL_ASSERT(idx<data_.size());
	// 		const Particle_t& nearest = data_[idx];

	// 		const T sdf = nearest.signed_distance(point);
	// 		const Point_t sdf_grad = nearest.grad_signed_distance(point);

	// 		const T c = compute_reciprocal_coef(eps);
	// 		if constexpr (Interior) {
	// 			return heaviside_grad_coef_of_sdf(sdf, eps, c) * sdf_grad;
	// 		}
	// 		else {
	// 			return -heaviside_grad_coef_of_sdf(-sdf, eps, c) * sdf_grad;
	// 		}
	// 	}

	// 	static void heaviside(std::span<T> result, std::span<const T> sdf, const T eps) noexcept {
	// 		GUTIL_ASSERT(result.size()==sdf.size());
	// 		GUTIL_ASSERT(eps>0);
	// 		const size_t N = sdf.size();
			
	// 		const T c = compute_reciprocal_coef(eps);
	// 		for (size_t i=0; i<N; ++i) {
	// 			result[i] = heaviside_of_sdf(sdf[i], eps, c);
	// 		}
	// 	}

	// 	static void heaviside_grad(std::span<T> result, std::span<const T> sdf, std::span<const T> sdf_grad, const T eps) noexcept {
	// 		GUTIL_ASSERT(result.size()==3*sdf.size());
	// 		GUTIL_ASSERT(result.size()==sdf_grad.size());
	// 		GUTIL_ASSERT(eps>0);

	// 		const size_t N = sdf.size();
	// 		const T c = compute_reciprocal_coef(eps);
	// 		for (size_t i=0; i<N; ++i) {
	// 			const T coef  = heaviside_grad_coef_of_sdf(sdf, eps, c);

	// 			result[i]     = coef * sdf_grad[i];
	// 			result[i+N]   = coef * sdf_grad[i+N];
	// 			result[i+2*N] = coef * sdf_grad[i+2*N];
	// 		}
	// 	}
	// };
}



