#pragma once

#include "fem/forms/bilinear_form.hpp"
#include "fem/forms/form_options.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//A symmetric kernel for bilinear forms a(psi,phi) = integral_D grad_phi*grad_psi
	template<typename 			 Handler_type,
			 BilinearFormOptions OPTIONS = BilinearFormOptions::assemble(true),
			 typename 			 DERIVED = void>
	struct SymmetricH1 : public BilinearForm<Handler_type,Handler_type,OPTIONS>
	{
		using BASE       = BilinearForm<Handler_type,Handler_type,OPTIONS>;
		using QuadElem_t = typename BASE::QuadElem_t;
		using DOF_t      = typename Handler_type::DOF_t;

		using BASE::BASE;

		//only provides the evaluation
		//should be vectorized with simd
		template<uint64_t N>
		constexpr void eval(
			std::array<double,N>& val, 
			const double Jxx, const double Jyy, const double Jzz, 
			const DOF_t psi_i,
			const QuadElem_t spt_i,
			const std::array<double,N>& X_i,
			const std::array<double,N>& Y_i,
			const std::array<double,N>& Z_i, 
			const DOF_t phi_j,
			const QuadElem_t spt_j,
			const std::array<double,N>& X_j,
			const std::array<double,N>& Y_j,
			const std::array<double,N>& Z_j) const requires (std::is_same_v<DERIVED,void>)
		{
			std::array<double,N> psi_i_gx, psi_i_gy, psi_i_gz;
			psi_i.grad(psi_i_gx, psi_i_gy, psi_i_gz, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_gx, phi_j_gy, phi_j_gz;
			phi_j.grad(phi_j_gx, phi_j_gy, phi_j_gz, spt_j, X_j, Y_j, Z_j);

			const double jac_det = Jxx*Jyy*Jzz;
			const double J2_ti_xx = 1.0/(Jxx*Jxx);
			const double J2_ti_yy = 1.0/(Jyy*Jyy);
			const double J2_ti_zz = 1.0/(Jzz*Jzz);
			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				val[i] = ( 	psi_i_gx[i]*phi_j_gx[i]*J2_ti_xx + 
							psi_i_gy[i]*phi_j_gy[i]*J2_ti_yy + 
							psi_i_gz[i]*phi_j_gz[i]*J2_ti_zz ) * jac_det;
			}
		}

		template<uint64_t N>
		constexpr void eval(
			std::array<double,N>& val, 
			const double Jxx, const double Jyy, const double Jzz, 
			const DOF_t psi_i,
			const QuadElem_t spt_i,
			const std::array<double,N>& X_i,
			const std::array<double,N>& Y_i,
			const std::array<double,N>& Z_i, 
			const DOF_t phi_j,
			const QuadElem_t spt_j,
			const std::array<double,N>& X_j,
			const std::array<double,N>& Y_j,
			const std::array<double,N>& Z_j) const requires (!std::is_same_v<DERIVED,void>)
		{
			std::array<double,N> psi_i_gx, psi_i_gy, psi_i_gz;
			psi_i.grad(psi_i_gx, psi_i_gy, psi_i_gz, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_gx, phi_j_gy, phi_j_gz;
			phi_j.grad(phi_j_gx, phi_j_gy, phi_j_gz, spt_j, X_j, Y_j, Z_j);

			//evaluate the weight. should not depend on which support is used
			std::array<double,N> w_x, w_y, w_z, x, y, z;
			this->ref2geo(x,y,z,spt_i,X_i,Y_i,Z_i);
			static_cast<const DERIVED*>(this) -> eval_w(w_x, w_y, w_z, x, y, z);

			const double jac_det = Jxx*Jyy*Jzz;
			const double inv_J_xx_sq = 1.0/(Jxx*Jxx);
			const double inv_J_yy_sq = 1.0/(Jyy*Jyy);
			const double inv_J_zz_sq = 1.0/(Jzz*Jzz);
			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				val[i] = ( 	psi_i_gx[i]*phi_j_gx[i]*w_x[i]*inv_J_xx_sq + 
							psi_i_gy[i]*phi_j_gy[i]*w_y[i]*inv_J_yy_sq + 
							psi_i_gz[i]*phi_j_gz[i]*w_z[i]*inv_J_zz_sq ) * jac_det;
			}
		}
	};

}
