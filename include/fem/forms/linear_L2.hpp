#pragma once

#include "fem/forms/linear_form.hpp"
#include "fem/forms/form_options.hpp"

#include "util/concepts.hpp"

#include <functional>
#include <array>
#include <cstdint>


#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	template<typename DofHandler_type, LinearFormOptions OPTIONS=LinearFormOptions::assemble(), typename WeightFun_t = std::nullptr_t, typename PredFun_t = std::nullptr_t>
	struct LinearL2 : public LinearForm<DofHandler_type, OPTIONS>
	{
		using BASE       = LinearForm<DofHandler_type, OPTIONS>;
		using QuadElem_t = typename BASE::QuadElem_t;
		using DOF_t      = typename BASE::TestDOF_t;
		
		using Fun_t      = std::conditional_t< NULLPTR_T<WeightFun_t>, std::nullptr_t, std::function<double(double,double,double)>>;
		using Pred_t     = std::conditional_t< NULLPTR_T<PredFun_t>, std::nullptr_t, std::function<bool(double,double,double)>>;

		Fun_t 	weight_fun;
		Pred_t  pred_fun;
		double	const_weight = 0.0;

		explicit LinearL2(const DofHandler_type& dofhandler) : BASE(dofhandler), weight_fun{}, pred_fun{} {}
		
		LinearL2(const DofHandler_type& dofhandler, WeightFun_t wfun, PredFun_t pfun = nullptr) requires(!NULLPTR_T<WeightFun_t>) :
			BASE(dofhandler), weight_fun(wfun), pred_fun(pfun) {}


		//set evaluation method based on which templates were provided
		//case 1 : constant function over entire domain
		#pragma omp declare simd notinbranch, uniform(this)
		double eval_w(const double x, const double y, const double z) const requires (NULLPTR_T<Pred_t> && NULLPTR_T<Fun_t>) {
			return const_weight;
		}
		//case 2 : piecewise constant function over portions of the domain
		#pragma omp declare simd notinbranch, uniform(this)
		double eval_w(const double x, const double y, const double z) const requires (!NULLPTR_T<Pred_t> && NULLPTR_T<Fun_t>) {
			return pred_fun(x,y,z) ? const_weight : 0.0;
		}
		//case 3 : function defined over entire domain
		#pragma omp declare simd notinbranch, uniform(this)
		double eval_w(const double x, const double y, const double z) const requires (NULLPTR_T<Pred_t> && !NULLPTR_T<Fun_t>) {
			return weight_fun(x,y,z);
		}
		//case 4 : function defined over a portion of the domain
		#pragma omp declare simd notinbranch, uniform(this)
		double eval_w(const double x, const double y, const double z) const requires (!NULLPTR_T<Pred_t> && !NULLPTR_T<Fun_t>) {
			return pred_fun(x,y,z) ? weight_fun(x,y,z) : 0.0;
		}

		void set_constant(double val) {const_weight=val;}

		//only provides the evaluation
		//should be vectorized with simd
		template<uint64_t N>
		constexpr void eval(
			std::array<double,N>& val, 
			const double Jxx, const double Jyy, const double Jzz, 
			const DOF_t psi,
			const QuadElem_t spt,
			const std::array<double,N>& X,
			const std::array<double,N>& Y,
			const std::array<double,N>& Z) const
		{
			std::array<double,N> psi_vals;
			psi.eval(psi_vals, spt, X, Y, Z);

			std::array<double,N> x, y, z;
			this->ref2geo(x,y,z,spt,X,Y,Z);
			const double jac_det = Jxx*Jyy*Jzz;

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				val[i] = psi_vals[i]*eval_w(x[i],y[i],z[i])*jac_det;
			}
		}
	};
}
