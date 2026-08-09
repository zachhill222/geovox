#pragma once

#include <array>
#include <span>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//Evaulatuation policies for various bilinear forms
	template<typename TestDOF_t, typename TrialDOF_t>
	struct BilinearH1
	{
		//flags for essential compile time decisions
		static constexpr bool NEEDS_GEOMETRY_POINTS = false;
		static constexpr bool IS_SYMMETRIC = true;

		//get the type of quadrature element
		using QuadElem_t = typename TestDOF_t::QuadElem_t::NonPeriodicVariant;
		static_assert(std::same_as<QuadElem_t, typename TrialDOF_t::QuadElem_t::NonPeriodicVariant>,
			"BilinearH1 - test and trial dofs need the same type of quadruatrue element");

		template<uint64_t N>
		void evaluate(	std::span<double, N> val, 
						const double Jxx, const double Jyy, const double Jzz,
						const TestDOF_t psi_i, const QuadElem_t spt_i, 
						std::span<const double,N> X_i, std::span<const double,N> Y_i, std::span<const double,N> Z_i,
						const TrialDOF_t phi_j, const QuadElem_t spt_j, 
						std::span<const double,N> X_j, std::span<const double,N> Y_j, std::span<const double,N> Z_j) const
		{
			std::array<double,N> psi_i_gx, psi_i_gy, psi_i_gz;
			psi_i.template grad<N>(psi_i_gx, psi_i_gy, psi_i_gz, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_gx, phi_j_gy, phi_j_gz;
			phi_j.template grad<N>(phi_j_gx, phi_j_gy, phi_j_gz, spt_j, X_j, Y_j, Z_j);

			// const double jac_det  = Jxx*Jyy*Jzz;
			const double J2_ti_xx = Jyy*Jzz/Jxx; // jac_det * 1.0/(Jxx*Jxx);
			const double J2_ti_yy = Jzz*Jxx/Jyy; // jac_det * 1.0/(Jyy*Jyy);
			const double J2_ti_zz = Jxx*Jyy/Jzz; // jac_det * 1.0/(Jzz*Jzz);
			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				val[i] = 	psi_i_gx[i]*phi_j_gx[i]*J2_ti_xx + 
							psi_i_gy[i]*phi_j_gy[i]*J2_ti_yy + 
							psi_i_gz[i]*phi_j_gz[i]*J2_ti_zz ;
			}
		}
	};

	
	template<typename TestDOF_t, typename TrialDOF_t>
	struct BilinearL2
	{
		//flag to tell the caller if it needs to compute the geometry point of the reference quadrature points
		static constexpr bool NEEDS_GEOMETRY_POINTS = false;
		static constexpr bool IS_SYMMETRIC = true;

		//get the type of quadrature element
		using QuadElem_t = typename TestDOF_t::QuadElem_t::NonPeriodicVariant;
		static_assert(std::same_as<QuadElem_t, typename TrialDOF_t::QuadElem_t::NonPeriodicVariant>,
			"BilinearH1 - test and trial dofs need the same type of quadruatrue element");

		template<uint64_t N>
		void evaluate(	std::span<double, N> val, 
						const double Jxx, const double Jyy, const double Jzz,
						const TestDOF_t psi_i, const QuadElem_t spt_i, 
						std::span<const double,N> X_i, std::span<const double,N> Y_i, std::span<const double,N> Z_i,
						const TrialDOF_t phi_j, const QuadElem_t spt_j, 
						std::span<const double,N> X_j, std::span<const double,N> Y_j, std::span<const double,N> Z_j) const
		{
			std::array<double,N> psi_i_vals;
			psi_i.template eval<N>(psi_i_vals, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_vals;
			phi_j.template eval<N>(phi_j_vals, spt_j, X_j, Y_j, Z_j);

			const double jac_det = Jxx*Jyy*Jzz;

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				val[i] = psi_i_vals[i]*phi_j_vals[i]*jac_det;
			}
		}
	};


	//bilinear forms a(psi,phi) = -integral_D phi*\partial_k psi
	//here psi is (the k-th component of) the vector dof and phi is the scalar dof.
	//Here we treat the vector as the test dof and the scalar as the trial.
	template<typename TestDOF_t, typename TrialDOF_t, int k>
	struct BilinearHdiv
	{
		//flag to tell the caller if it needs to compute the geometry point of the reference quadrature points
		static constexpr bool NEEDS_GEOMETRY_POINTS = false;
		static constexpr bool IS_SYMMETRIC = false;

		//get the type of quadrature element
		using QuadElem_t = typename TestDOF_t::QuadElem_t::NonPeriodicVariant;
		static_assert(std::same_as<QuadElem_t, typename TrialDOF_t::QuadElem_t::NonPeriodicVariant>,
			"BilinearH1 - test and trial dofs need the same type of quadruatrue element");

		template<uint64_t N>
		void evaluate(	std::span<double, N> val, 
						const double Jxx, const double Jyy, const double Jzz,
						const TestDOF_t psi_i, const QuadElem_t spt_i, 
						std::span<const double,N> X_i, std::span<const double,N> Y_i, std::span<const double,N> Z_i,
						const TrialDOF_t phi_j, const QuadElem_t spt_j, 
						std::span<const double,N> X_j, std::span<const double,N> Y_j, std::span<const double,N> Z_j) const
		{
			std::array<double,N> psi_i_gx, psi_i_gy, psi_i_gz;
			psi_i.template grad<N>(psi_i_gx, psi_i_gy, psi_i_gz, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_vals;
			phi_j.template eval<N>(phi_j_vals, spt_j, X_j, Y_j, Z_j);

			const double J2_ti_xx = Jyy*Jzz; //jac_det/Jxx
			const double J2_ti_yy = Jzz*Jxx; //jac_det/Jyy
			const double J2_ti_zz = Jxx*Jyy; //jac_det/Jzz
			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {
				if constexpr (k==0) {
					val[i] = -psi_i_gx[i]*phi_j_vals[i]*J2_ti_xx;
				}
				else if constexpr (k==1) {
					val[i] = -psi_i_gy[i]*phi_j_vals[i]*J2_ti_yy;
				}
				else {
					val[i] = -psi_i_gz[i]*phi_j_vals[i]*J2_ti_zz;
				}
			}
		}
	};


	//bilinear forms a(psi,phi) = -integral_D psi*\partial_k phi
	//here phi is (the k-th component of) the vector dof and psi is the scalar dof.
	//Here we treat the scalar as the test dof and the vector as the trial.
	template<typename TestDOF_t, typename TrialDOF_t, int k>
	struct BilinearHdivAdjoint
	{
		//flag to tell the caller if it needs to compute the geometry point of the reference quadrature points
		static constexpr bool NEEDS_GEOMETRY_POINTS = false;
		static constexpr bool IS_SYMMETRIC = false;

		//get the type of quadrature element
		using QuadElem_t = typename TestDOF_t::QuadElem_t::NonPeriodicVariant;
		static_assert(std::same_as<QuadElem_t, typename TrialDOF_t::QuadElem_t::NonPeriodicVariant>,
			"BilinearH1 - test and trial dofs need the same type of quadruatrue element");

		template<uint64_t N>
		void evaluate(	std::span<double, N> val, 
						const double Jxx, const double Jyy, const double Jzz,
						const TestDOF_t psi_i, const QuadElem_t spt_i, 
						std::span<const double,N> X_i, std::span<const double,N> Y_i, std::span<const double,N> Z_i,
						const TrialDOF_t phi_j, const QuadElem_t spt_j, 
						std::span<const double,N> X_j, std::span<const double,N> Y_j, std::span<const double,N> Z_j) const
		{
			std::array<double,N> psi_i_vals;
			psi_i.template eval<N>(psi_i_vals, spt_i, X_i, Y_i, Z_i);

			std::array<double,N> phi_j_gx, phi_j_gy, phi_j_gz;
			phi_j.template grad<N>(phi_j_gx, phi_j_gy, phi_j_gz, spt_j, X_j, Y_j, Z_j);

			const double J2_ti_xx = Jyy*Jzz; //jac_det/Jxx
			const double J2_ti_yy = Jzz*Jxx; //jac_det/Jyy
			const double J2_ti_zz = Jxx*Jyy; //jac_det/Jzz
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