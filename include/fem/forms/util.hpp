#pragma once

namespace GV {
	///////////////////////////////////////////////////////////////////
	/// Caching classes to help evaluate dofs as needed and reduce
	/// redundant computations in combined kernels (e.g., K = K1 + 2*K2)
	///////////////////////////////////////////////////////////////////
	template<typename QuadRule_t>
	struct DofValueCache {
		using Scalar_t = typename QuadRule_t::Scalar_t;
		static constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

		Scalar_t val[N];
		const uint8_t depth;

		[[nodiscard]] constexpr Scalar_t operator[](int n) const {
			GUTIL_ASSERT(n>=0 && n<N);
			return val[n];
		}

		template<typename DOF_t>
		DofValueCache(DOF_t dof, const QuadRule_t& qr) : depth(dof.depth_u8()) {
			const uint8_t loc  = dof.local_dof_number(qr.support_element[depth]);
			auto qx=qr.quad_x(depth), qy=qr.quad_y(depth), qz=qr.quad_z(depth);
			dof.evaluate_simd(loc, val, qx.data(), qy.data(), qz.data(), N);
		}
	};

	template<typename QuadRule_t>
	struct DofGradCache {
		using Scalar_t = typename QuadRule_t::Scalar_t;
		static constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;

		Scalar_t grad[3*N];
		const uint8_t depth;

		Scalar_t *gx, *gy, *gz;

		template<typename DOF_t>
		DofGradCache(DOF_t dof, const QuadRule_t& qr) : depth(dof.depth_u8()), gx(grad), gy(grad+N), gz(grad+2*N) {
			const uint8_t loc  = dof.local_dof_number(qr.support_element[depth]);
			auto qx=qr.quad_x(depth), qy=qr.quad_y(depth), qz=qr.quad_z(depth);
			dof.gradient_simd(loc, grad, grad+N, grad+2*N, qx.data(), qy.data(), qz.data(), N);
		}
	};


	///////////////////////////////////////////////////////////////////
	/// Weighting classes to help with evaluating weights of the form
	/// f(x,y,z) or h(u) or h(u)*f(x,y,z)
	///////////////////////////////////////////////////////////////////
	// struct 


}