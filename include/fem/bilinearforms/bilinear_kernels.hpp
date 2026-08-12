#pragma once

#include "gutil.hpp"

#include "fem/mesh_quadrature.hpp"

#include <concepts>
#include <type_traits>

namespace GV {


	///////////////////////////////////////////////////////////////////
	/// A few evaluation methods for standard bilinear kernels
	///////////////////////////////////////////////////////////////////
	struct ZeroBilinearKernel {
		//kernel for the bilinear form B(phi,psi) = 0
		static constexpr bool NEEDS_GEO_POINTS = false;
		static constexpr bool IS_SYMMETRIC     = false;

		template<typename TestDof_t, typename TrialDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TestDof_t,TrialDof_t,const QuadRule_t&) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			return Scalar_t{0};
		}
	};

	struct IdentityBilinearKernel {
		//kernel for the identity matrix
		static constexpr bool NEEDS_GEO_POINTS = false;
		static constexpr bool IS_SYMMETRIC     = true;

		template<typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TestDof_t phi, TestDof_t psi,const QuadRule_t&) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			return Scalar_t{phi==psi};
		}
	};

	template<bool IsSymmetric=false, typename Derived=void>
	struct MassBilinearKernel {
		//kernel for the linear form B(phi,psi) = int_D(phi*psi)
		//optionally B(phi,psi) = int_D(phi*psi*f) where f is supplied by CRTP in the Derived class
		//and is a function of x,y,z.
		static constexpr bool NEEDS_GEO_POINTS = !std::same_as<Derived,void>;
		static constexpr bool IS_SYMMETRIC     = IsSymmetric;

		template<typename TestDof_t, typename TrialDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TestDof_t phi, TrialDof_t psi, const QuadRule_t& qr) const noexcept requires(!NEEDS_GEO_POINTS) {
			
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t test_vals[N];
			const uint8_t test_d    = phi.depth();
			const uint8_t test_loc  = phi.local_dof_number(qr.support_element[test_d]);
			auto qx=qr.quad_x(test_d), qy=qr.quad_y(test_d), qz=qr.quad_z(test_d);
			phi.evaluate_simd(test_loc, test_vals, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t trial_vals[N];
			const uint8_t trial_d   = psi.depth();
			const uint8_t trial_loc = psi.local_dof_number(qr.support_element[trial_d]);
			qx=qr.quad_x(trial_d), qy=qr.quad_y(trial_d), qz=qr.quad_z(trial_d);
			psi.evaluate_simd(trial_loc, trial_vals, qx.data(), qy.data(), qz.data(), N);
			
			//accumulation
			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += test_vals[i]*trial_vals[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}

		template<typename TestDof_t, typename TrialDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TestDof_t phi, TrialDof_t psi, const QuadRule_t& qr) const noexcept requires(NEEDS_GEO_POINTS) {
			
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t test_vals[N];
			const uint8_t test_d    = phi.depth();
			const uint8_t test_loc  = phi.local_dof_number(qr.support_element[test_d]);
			auto qx=qr.quad_x(test_d), qy=qr.quad_y(test_d), qz=qr.quad_z(test_d);
			phi.evaluate_simd(test_loc, test_vals, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t trial_vals[N];
			const uint8_t trial_d   = psi.depth();
			const uint8_t trial_loc = psi.local_dof_number(qr.support_element[trial_d]);
			qx=qr.quad_x(trial_d), qy=qr.quad_y(trial_d), qz=qr.quad_z(trial_d);
			psi.evaluate_simd(trial_loc, trial_vals, qx.data(), qy.data(), qz.data(), N);
			
			//measure/weight evalutation
			Scalar_t f_vals[N];
			Derived::eval_weight(std::span<Scalar_t,N>{f_vals},
						qr.geo_x(),qr.geo_y(),qr.geo_z());

			//accumulation
			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += test_vals[i]*trial_vals[i]*f_vals[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};

	template<bool IsSymmetric=false, typename Derived=void>
	struct StiffBilinearKernel {
		//kernel for the linear form B(phi,psi) = int_D(grad(phi)*grad(psi))
		//optionally B(phi,psi) = int_D(grad(phi)*grad(psi)*f) where f is supplied by CRTP in the Derived class
		//f must be a function of x,y,z.
		static constexpr bool NEEDS_GEO_POINTS = !std::same_as<Derived,void>;
		static constexpr bool IS_SYMMETRIC     = IsSymmetric;

		template<typename TestDof_t, typename TrialDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TestDof_t phi, TrialDof_t psi, const QuadRule_t& qr) const noexcept requires(!NEEDS_GEO_POINTS) {
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t  test_vals[3*N];
			Scalar_t* test_gx = test_vals;
			Scalar_t* test_gy = test_vals+N;
			Scalar_t* test_gz = test_vals+2*N;
			const uint8_t test_d = phi.depth();
			const uint8_t test_loc = phi.local_dof_number(qr.support_element[test_d]);
			auto qx=qr.quad_x(test_d), qy=qr.quad_y(test_d), qz=qr.quad_z(test_d);
			phi.gradient_simd(test_loc, test_gx, test_gy, test_gz, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t  trial_vals[3*N];
			Scalar_t* trial_gx = trial_vals;
			Scalar_t* trial_gy = trial_vals+N;
			Scalar_t* trial_gz = trial_vals+2*N;
			const uint8_t trial_d = psi.depth();
			const uint8_t trial_loc = psi.local_dof_number(qr.support_element[trial_d]);
			qx=qr.quad_x(trial_d), qy=qr.quad_y(trial_d), qz=qr.quad_z(trial_d);
			psi.gradient_simd(trial_loc, trial_gx, trial_gy, trial_gz, qx.data(), qy.data(), qz.data(), N);
			
			//accumulation
			auto qw = qr.quad_w();
			auto& jac_inv = qr.jacobian_diag_inv[qr.q_el.depth()];
			Scalar_t j_i_xx = jac_inv[0]*jac_inv[0];
			Scalar_t j_i_yy = jac_inv[1]*jac_inv[1];
			Scalar_t j_i_zz = jac_inv[2]*jac_inv[2];

			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += ( test_gx[i]*trial_gx[i]*j_i_xx +
							test_gy[i]*trial_gy[i]*j_i_yy +
							test_gz[i]*trial_gz[i]*j_i_zz )*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}

		template<typename TestDof_t, typename TrialDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TestDof_t phi, TrialDof_t psi, const QuadRule_t& qr) const noexcept requires(NEEDS_GEO_POINTS) {
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t  test_vals[3*N];
			Scalar_t* test_gx = test_vals;
			Scalar_t* test_gy = test_vals+N;
			Scalar_t* test_gz = test_vals+2*N;
			const uint8_t test_d = phi.depth();
			const uint8_t test_loc = phi.local_dof_number(qr.support_element[test_d]);
			auto qx=qr.quad_x(test_d), qy=qr.quad_y(test_d), qz=qr.quad_z(test_d);
			phi.gradient_simd(test_loc, test_gx, test_gy, test_gz, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t  trial_vals[3*N];
			Scalar_t* trial_gx = trial_vals;
			Scalar_t* trial_gy = trial_vals+N;
			Scalar_t* trial_gz = trial_vals+2*N;
			const uint8_t trial_d = psi.depth();
			const uint8_t trial_loc = psi.local_dof_number(qr.support_element[trial_d]);
			qx=qr.quad_x(trial_d), qy=qr.quad_y(trial_d), qz=qr.quad_z(trial_d);
			psi.gradient_simd(trial_loc, trial_gx, trial_gy, trial_gz, qx.data(), qy.data(), qz.data(), N);
			
			//measure/weight evalutation
			Scalar_t f_vals[N];
			Derived::eval_weight(std::span<Scalar_t,N>{f_vals},
						qr.geo_x(),qr.geo_y(),qr.geo_z());

			//accumulation
			auto qw = qr.quad_w();
			auto& jac_inv = qr.jacobian_diag_inv[qr.q_el.depth()];
			Scalar_t j_i_xx = jac_inv[0]*jac_inv[0];
			Scalar_t j_i_yy = jac_inv[1]*jac_inv[1];
			Scalar_t j_i_zz = jac_inv[2]*jac_inv[2];

			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += ( test_gx[i]*trial_gx[i]*j_i_xx +
							test_gy[i]*trial_gy[i]*j_i_yy +
							test_gz[i]*trial_gz[i]*j_i_zz )* f_vals[i] * qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};

	
}