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

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TrialDof_t, TestDof_t,const QuadRule_t&) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			return Scalar_t{0};
		}
	};

	struct IdentityBilinearKernel {
		//kernel for the identity matrix
		static constexpr bool NEEDS_GEO_POINTS = false;
		static constexpr bool IS_SYMMETRIC     = true;

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t> requires(std::same_as<TrialDof_t,TestDof_t>)
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(TrialDof_t u, TestDof_t v,const QuadRule_t&) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			return static_cast<Scalar_t>(u==v);
		}
	};

	template<bool IsSymmetric=false, typename Derived=void>
	struct L2BilinearKernel {
		//kernel for the linear form B(u,v) = int_D(u*v)
		//optionally B(u,v) = int_D(u*v*f) where f is supplied by CRTP in the Derived class
		//and is a function of x,y,z.
		static constexpr bool NEEDS_GEO_POINTS = !std::same_as<Derived,void>;
		static constexpr bool IS_SYMMETRIC     = IsSymmetric;

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TrialDof_t u, TestDof_t v, const QuadRule_t& qr) const noexcept requires(!NEEDS_GEO_POINTS) {
			
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t v_vals[N];
			const uint8_t v_depth   = v.depth();
			const uint8_t v_loc  	= v.local_dof_number(qr.support_element[v_depth]);
			auto qx=qr.quad_x(v_depth), qy=qr.quad_y(v_depth), qz=qr.quad_z(v_depth);
			v.evaluate_simd(v_loc, v_vals, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t u_vals[N];
			const uint8_t u_depth   = u.depth();
			const uint8_t u_loc 	= u.local_dof_number(qr.support_element[u_depth]);
			qx=qr.quad_x(u_depth), qy=qr.quad_y(u_depth), qz=qr.quad_z(u_depth);
			u.evaluate_simd(u_loc, u_vals, qx.data(), qy.data(), qz.data(), N);
			
			//accumulation
			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += u_vals[i]*v_vals[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TrialDof_t u, TestDof_t v, const QuadRule_t& qr) const noexcept requires(NEEDS_GEO_POINTS) {
			
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t v_vals[N];
			const uint8_t v_depth   = v.depth();
			const uint8_t v_loc  	= v.local_dof_number(qr.support_element[v_depth]);
			auto qx=qr.quad_x(v_depth), qy=qr.quad_y(v_depth), qz=qr.quad_z(v_depth);
			v.evaluate_simd(v_loc, v_vals, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t u_vals[N];
			const uint8_t u_depth   = u.depth();
			const uint8_t u_loc 	= u.local_dof_number(qr.support_element[u_depth]);
			qx=qr.quad_x(u_depth), qy=qr.quad_y(u_depth), qz=qr.quad_z(u_depth);
			u.evaluate_simd(u_loc, u_vals, qx.data(), qy.data(), qz.data(), N);
			
			//measure/weight evalutation
			Scalar_t f_vals[N];
			Derived::eval_weight(std::span<Scalar_t,N>{f_vals},
						qr.geo_x(),qr.geo_y(),qr.geo_z());

			//accumulation
			auto qw = qr.quad_w();
			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += u_vals[i]*v_vals[i]*f_vals[i]*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};

	template<bool IsSymmetric=false, typename Derived=void>
	struct H1BilinearKernel {
		//kernel for the linear form B(u,v) = int_D(grad(u)*grad(v))
		//optionally B(u,v) = int_D(grad(u)*grad(v)*f) where f is supplied by CRTP in the Derived class
		//f must be a function of x,y,z.
		static constexpr bool NEEDS_GEO_POINTS = !std::same_as<Derived,void>;
		static constexpr bool IS_SYMMETRIC     = IsSymmetric;

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TrialDof_t u, TestDof_t v, const QuadRule_t& qr) const noexcept requires(!NEEDS_GEO_POINTS) {
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t  v_vals[3*N];
			Scalar_t* v_gx = v_vals;
			Scalar_t* v_gy = v_vals+N;
			Scalar_t* v_gz = v_vals+2*N;
			const uint8_t v_depth = v.depth();
			const uint8_t v_loc = v.local_dof_number(qr.support_element[v_depth]);
			auto qx=qr.quad_x(v_depth), qy=qr.quad_y(v_depth), qz=qr.quad_z(v_depth);
			v.gradient_simd(v_loc, v_gx, v_gy, v_gz, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t  u_vals[3*N];
			Scalar_t* u_gx = u_vals;
			Scalar_t* u_gy = u_vals+N;
			Scalar_t* u_gz = u_vals+2*N;
			const uint8_t u_depth = u.depth();
			const uint8_t u_loc = u.local_dof_number(qr.support_element[u_depth]);
			qx=qr.quad_x(u_depth), qy=qr.quad_y(u_depth), qz=qr.quad_z(u_depth);
			u.gradient_simd(u_loc, u_gx, u_gy, u_gz, qx.data(), qy.data(), qz.data(), N);
			
			//accumulation
			auto qw = qr.quad_w();
			const Scalar_t j_i_xx = qr.jacobian_diag_inv[v_depth][0]*qr.jacobian_diag_inv[u_depth][0];
			const Scalar_t j_i_yy = qr.jacobian_diag_inv[v_depth][1]*qr.jacobian_diag_inv[u_depth][1];
			const Scalar_t j_i_zz = qr.jacobian_diag_inv[v_depth][2]*qr.jacobian_diag_inv[u_depth][2];

			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += ( v_gx[i]*u_gx[i]*j_i_xx +
							v_gy[i]*u_gy[i]*j_i_yy +
							v_gz[i]*u_gz[i]*j_i_zz )*qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}

		template<typename TrialDof_t, typename TestDof_t, typename QuadRule_t>
		[[nodiscard]] constexpr typename QuadRule_t::Scalar_t operator()(
				TrialDof_t u, TestDof_t v, const QuadRule_t& qr) const noexcept requires(NEEDS_GEO_POINTS) {
			static_assert(!IS_SYMMETRIC || std::same_as<TestDof_t,TrialDof_t>, 
				"for a symmetric form, the test and trial dofs must be the same type");
			
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

			//row/test evaluation
			Scalar_t  v_vals[3*N];
			Scalar_t* v_gx = v_vals;
			Scalar_t* v_gy = v_vals+N;
			Scalar_t* v_gz = v_vals+2*N;
			const uint8_t v_depth = v.depth();
			const uint8_t v_loc = v.local_dof_number(qr.support_element[v_depth]);
			auto qx=qr.quad_x(v_depth), qy=qr.quad_y(v_depth), qz=qr.quad_z(v_depth);
			v.gradient_simd(v_loc, v_gx, v_gy, v_gz, qx.data(), qy.data(), qz.data(), N);
			
			//column/trial evaluation
			Scalar_t  u_vals[3*N];
			Scalar_t* u_gx = u_vals;
			Scalar_t* u_gy = u_vals+N;
			Scalar_t* u_gz = u_vals+2*N;
			const uint8_t u_depth = u.depth();
			const uint8_t u_loc = u.local_dof_number(qr.support_element[u_depth]);
			qx=qr.quad_x(u_depth), qy=qr.quad_y(u_depth), qz=qr.quad_z(u_depth);
			u.gradient_simd(u_loc, u_gx, u_gy, u_gz, qx.data(), qy.data(), qz.data(), N);
			
			//measure/weight evalutation
			Scalar_t f_vals[N];
			Derived::eval_weight(std::span<Scalar_t,N>{f_vals},
						qr.geo_x(),qr.geo_y(),qr.geo_z());

			//accumulation
			auto qw = qr.quad_w();
			const Scalar_t j_i_xx = qr.jacobian_diag_inv[v_depth][0]*qr.jacobian_diag_inv[u_depth][0];
			const Scalar_t j_i_yy = qr.jacobian_diag_inv[v_depth][1]*qr.jacobian_diag_inv[u_depth][1];
			const Scalar_t j_i_zz = qr.jacobian_diag_inv[v_depth][2]*qr.jacobian_diag_inv[u_depth][2];

			Scalar_t result{0};
			GUTIL_SIMD(reduction(+:result))
			for (int i=0; i<N; ++i) {
				result += ( v_gx[i]*u_gx[i]*j_i_xx +
							v_gy[i]*u_gy[i]*j_i_yy +
							v_gz[i]*u_gz[i]*j_i_zz )* f_vals[i] * qw[i];
			}
			return result * qr.jacobian_det[qr.q_el.depth()];
		}
	};

	
}