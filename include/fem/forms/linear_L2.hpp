#pragma once

#include "fem/forms/linear_form.hpp"
#include "fem/forms/form_actions.hpp"

#include <array>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//A symmetric mass kernel for bilinear forms a(psi,phi) = integral_D phi*psi
	template<typename DofHandler_type, typename Action_type=ScatterAction, typename DERIVED=void>
	struct LinearL2 : public LinearForm<DofHandler_type, Action_type>
	{
		using BASE       = LinearForm<DofHandler_type, Action_type>;
		using QuadElem_t = typename BASE::QuadElem_t;
		using DOF_t      = typename BASE::TestDOF_t;

		using BASE::BASE;

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

			std::array<double,N> w_vals;
			if constexpr (!std::is_same_v<DERIVED,void>) {
				std::array<double,N> x, y, z;
				this->ref2geo(x,y,z,spt,X,Z,Z);
				static_cast<const DERIVED*>(this) -> eval_w(w_vals, x, y, z);
			}
			else {
				w_vals.fill(0.0);
			}
			
			const double jac_det = Jxx*Jyy*Jzz;

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				val[i] = psi_vals[i]*w_vals[i]*jac_det;
			}
		}
	};
}
