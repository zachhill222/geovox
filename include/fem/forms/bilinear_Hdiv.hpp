#pragma once

#include "fem/forms/bilinear_form.hpp"

#include <array>
#include <cstdint>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//A non-symmetric kernel for bilinear forms b(psi,phi) = -integral_D phi*\partial_k psi
	//here psi is (the k-th component of) the vector dof and phi is the scalar dof.
	//Here we treat the vector as the test dof and the scalar as the trial.
	template<typename 	TestHandler_type, 	//vector component
			 typename 	TrialHandler_type,  //scalar
			 int        k, 				    //component to use
			 typename 	ActionType 	= ScatterAction>
	struct BilinearHdiv : public BilinearForm<TestHandler_type,TrialHandler_type,false,ActionType>
	{
		static_assert(k>=0 && k<3, "BilinearHdiv - the component k must be between 0 and 2");

		using BASE       = BilinearForm<TestHandler_type,TrialHandler_type,false,ActionType>;
		using QuadElem_t = typename BASE::QuadElem_t;
		using TestDOF_t  = typename TestHandler_type::DOF_t;
		using TrialDOF_t = typename TrialHandler_type::DOF_t;

		using BASE::BASE;

		//only provides the evaluation
		//should be vectorized with simd
		template<uint64_t N>
		constexpr void eval(
			std::array<double,N>& val, 
			const double Jxx, const double Jyy, const double Jzz, 
			const TestDOF_t psi_i,
			const QuadElem_t spt_i,
			const std::array<double,N>& X_i,
			const std::array<double,N>& Y_i,
			const std::array<double,N>& Z_i, 
			const TrialDOF_t phi_j,
			const QuadElem_t spt_j,
			const std::array<double,N>& X_j,
			const std::array<double,N>& Y_j,
			const std::array<double,N>& Z_j) const
		{
			std::array<double,N> psi_i_gx, psi_i_gy, psi_i_gz;
			psi_i.grad(psi_i_gx, psi_i_gy, psi_i_gz, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_vals;
			phi_j.eval(phi_j_vals, spt_j, X_j, Y_j, Z_j);

			const double J2_ti_xx = Jyy*Jzz; //det/Jxx
			const double J2_ti_yy = Jzz*Jxx; //det/Jyy
			const double J2_ti_zz = Jxx*Jyy; //det/Jzz
			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				if constexpr (k==0) {
					val[i] = -psi_i_gx[i]*phi_j_vals[i]*J2_ti_xx;
				}
				else if constexpr (k==1) {
					val[i] = -psi_i_gy[i]*phi_j_vals[i]*J2_ti_yy;
				}
				else {
					assert(k==2);
					val[i] = -psi_i_gz[i]*phi_j_vals[i]*J2_ti_zz;
				}
			}
		}
	};

	//swap the roles of the test and trial spaces for the transpose/adjoint form
	template<typename 	TestHandler_type, 	//scalar
			 typename 	TrialHandler_type,  //vector component
			 int        k, 				    //component to use
			 typename 	ActionType 	= ScatterAction>
	struct BilinearHdivAdjoint : public BilinearForm<TestHandler_type,TrialHandler_type,false,ActionType>
	{
		static_assert(k>=0 && k<3, "BilinearHdivAdjoint - the component k must be between 0 and 2");

		using BASE       = BilinearForm<TestHandler_type,TrialHandler_type,false,ActionType>;
		using QuadElem_t = typename BASE::QuadElem_t;
		using TestDOF_t  = typename TestHandler_type::DOF_t;
		using TrialDOF_t = typename TrialHandler_type::DOF_t;

		using BASE::BASE;

		//only provides the evaluation
		//should be vectorized with simd
		template<uint64_t N>
		constexpr void eval(
			std::array<double,N>& val, 
			const double Jxx, const double Jyy, const double Jzz, 
			const TestDOF_t psi_i,
			const QuadElem_t spt_i,
			const std::array<double,N>& X_i,
			const std::array<double,N>& Y_i,
			const std::array<double,N>& Z_i, 
			const TrialDOF_t phi_j,
			const QuadElem_t spt_j,
			const std::array<double,N>& X_j,
			const std::array<double,N>& Y_j,
			const std::array<double,N>& Z_j) const
		{
			std::array<double,N> psi_i_vals;
			psi_i.eval(psi_i_vals, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_gx, phi_j_gy, phi_j_gz;
			phi_j.grad(phi_j_gx, phi_j_gy, phi_j_gz, spt_j, X_j, Y_j, Z_j);

			const double J2_ti_xx = Jyy*Jzz; //det/Jxx
			const double J2_ti_yy = Jzz*Jxx; //det/Jyy
			const double J2_ti_zz = Jxx*Jyy; //det/Jzz
			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				if constexpr (k==0) {
					val[i] = -psi_i_vals[i]*phi_j_gx[i]*J2_ti_xx;
				}
				else if constexpr (k==1) {
					val[i] = -psi_i_vals[i]*phi_j_gy[i]*J2_ti_yy;
				}
				else {
					assert(k==2);
					val[i] = -psi_i_vals[i]*phi_j_gz[i]*J2_ti_zz;
				}
			}
		}
	};
}
