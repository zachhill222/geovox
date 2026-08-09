#pragma once

#include "util/concepts.hpp"

#include <functional>
#include <array>
#include <span>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//Evaulatuation policies for various bilinear forms
	template<typename TestDOF_t, typename WeightFun_t = std::nullptr_t, typename PredFun_t = std::nullptr_t>
	struct LinearL2
	{
		//get the type of quadrature element
		using QuadElem_t = typename TestDOF_t::QuadElem_t::NonPeriodicVariant;

		//set up evaluation logic
		using Fun_t      = std::conditional_t< NULLPTR_T<WeightFun_t>, std::nullptr_t, std::function<double(double,double,double)>>;
		using Pred_t     = std::conditional_t< NULLPTR_T<PredFun_t>, std::nullptr_t, std::function<bool(double,double,double)>>;

		Fun_t 	weight_fun;
		Pred_t  pred_fun;
		double	const_weight = 0.0;

		LinearL2() : weight_fun{}, pred_fun{} {}
		LinearL2(const double wt) requires (NULLPTR_T<WeightFun_t>&&NULLPTR_T<Pred_t>) : weight_fun{}, pred_fun{}, const_weight{wt} {}
		LinearL2(WeightFun_t wfun, PredFun_t pfun = nullptr) requires(!NULLPTR_T<WeightFun_t>) : weight_fun(wfun), pred_fun(pfun) {}


		//set evaluation method based on which templates were provided
		//case 1 : constant function over entire domain
		#pragma omp declare simd notinbranch, uniform(this)
		inline double eval_w(const double x, const double y, const double z) const requires (NULLPTR_T<Pred_t> && NULLPTR_T<Fun_t>) {
			return const_weight;
		}
		//case 2 : piecewise constant function over portions of the domain
		#pragma omp declare simd notinbranch, uniform(this)
		inline double eval_w(const double x, const double y, const double z) const requires (!NULLPTR_T<Pred_t> && NULLPTR_T<Fun_t>) {
			return pred_fun(x,y,z) ? const_weight : 0.0;
		}
		//case 3 : function defined over entire domain
		#pragma omp declare simd notinbranch, uniform(this)
		inline double eval_w(const double x, const double y, const double z) const requires (NULLPTR_T<Pred_t> && !NULLPTR_T<Fun_t>) {
			return weight_fun(x,y,z);
		}
		//case 4 : function defined over a portion of the domain
		#pragma omp declare simd notinbranch, uniform(this)
		inline double eval_w(const double x, const double y, const double z) const requires (!NULLPTR_T<Pred_t> && !NULLPTR_T<Fun_t>) {
			return pred_fun(x,y,z) ? weight_fun(x,y,z) : 0.0;
		}

		//let the caller know if the geometric points need to be supplied
		static constexpr bool NEEDS_GEOMETRY_POINTS = !(NULLPTR_T<Pred_t> && NULLPTR_T<Fun_t>);
		std::vector<double> x, y, z;

		template<uint64_t N>
		void evaluate(	std::span<double, N> val, 
						const double Jxx, const double Jyy, const double Jzz,
						const TestDOF_t psi_i, const QuadElem_t spt_i, 
						std::span<const double,N> X_i, std::span<const double,N> Y_i, std::span<const double,N> Z_i) const
		{
			if constexpr (NEEDS_GEOMETRY_POINTS) {
				assert(x.size()==N);
				assert(y.size()==N);
				assert(z.size()==N);
			}

			std::array<double,N> psi_i_vals;
			psi_i.template eval<N>(psi_i_vals, spt_i, X_i, Y_i, Z_i);

			const double jac_det = Jxx*Jyy*Jzz;

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				if constexpr (NEEDS_GEOMETRY_POINTS) {
					val[i] = psi_i_vals[i]*eval_w(x[i],y[i],z[i])*jac_det;
				}
				else {
					val[i] = psi_i_vals[i]*const_weight*jac_det;
				}
			}
		}
	};
}