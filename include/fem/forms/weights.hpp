#pragma once

#include "gutil.hpp"
#include "fem/forms/util.hpp"

namespace GV {
	///////////////////////////////////////////////////////////////////
	/// Many forms require some sort of weight that only needs to be computed once per
	/// quadrature point. These weights could be of the form f(x,y,z) where
	/// x,y,z are the geometric coordinates of a quadrature point or
	/// it could be h(u(x,y,z)) where u(x,y,z) is the evaluation of a scalar field
	/// at the quadrature points.
	///////////////////////////////////////////////////////////////////
	template<typename W>
	concept IsKernelWeight = requires {
		{W::NEEDS_GEO_POINTS}  -> std::convertible_to<bool>;
		{W::NEEDS_SCALAR_VALS} -> std::convertible_to<bool>;
		//also needs a templated build_weights(scalarvals, quad_rule)
		//the quad rule will hold geometric points while scalarvals will
		//point to the values of the scalar field at the quadrature points.
	};


	///////////////////////////////////////////////////////////////////
	/// Standard kernel weights
	///////////////////////////////////////////////////////////////////
	struct NoKernelWeight {
		//weight of a constant 1, probably shouldn't actually evaluate this ever.
		static constexpr bool NEEDS_GEO_POINTS  = false;
		static constexpr bool NEEDS_SCALAR_VALS = false;

		template<typename QuadRule_t>
		static constexpr ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t&) noexcept {
			return ScalarValueCache<QuadRule_t>::Filled(typename QuadRule_t::Scalar_t{1});
		}
	};

	template<auto F>
	struct FunctionWeight {
		static constexpr bool NEEDS_GEO_POINTS  = true;
		static constexpr bool NEEDS_SCALAR_VALS = false;

		template<typename QuadRule_t>
		ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			ScalarValueCache<QuadRule_t> result;
			auto gx=qr.geo_x(), gy=qr.geo_y(), gz=qr.geo_z();

			GUTIL_SIMD()
			for (int i=0; i<N; ++i) { result[i] = static_cast<Scalar_t>(F(gx[i], gy[i], gz[i])); }
			
			return result;
		}
	};


	///////////////////////////////////////////////////////////////////
	/// Diffuse domain weights
	///////////////////////////////////////////////////////////////////
	template<bool Interior=true, typename Assembly_t=void, int StaticID=0>
	struct AssemblyPhaseFieldWeight {
		//phi or 1-phi using the assembly tanh heaviside approximation
		static constexpr bool NEEDS_GEO_POINTS  = true;
		static constexpr bool NEEDS_SCALAR_VALS = false;
		
		//make the assembly and epsilon static so they can be set without worrying about a constructor.
		static constexpr bool INTERIOR = Interior;
		const Assembly_t& assembly;
		typename Assembly_t::Scalar_t eps{0};

		AssemblyPhaseFieldWeight(const Assembly_t& a, typename Assembly_t::Scalar_t e) : assembly(a), eps(e) {}

		template<typename QuadRule_t>
		ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			
			ScalarValueCache<QuadRule_t> result;
			auto gx=qr.geo_x(), gy=qr.geo_y(), gz=qr.geo_z();	//get geometric coordinates of the quadrature points
			for (int i=0; i<N; ++i) {
				typename Assembly_t::Point_t p{gx[i], gy[i], gz[i]};
				if constexpr (INTERIOR) {
					result[i] = static_cast<Scalar_t>(assembly.heaviside_tanh(p,eps));
				}
				else {
					result[i] = Scalar_t{1} - static_cast<Scalar_t>(assembly.heaviside_tanh(p,eps));
				}
			}
			return result;
		}
	};


	template<typename Assembly_t, auto F>
	struct ModifiedFunctionKernelWeight {
		static constexpr bool NEEDS_GEO_POINTS  = true;
		static constexpr bool NEEDS_SCALAR_VALS = false;

		const Assembly_t& assembly;
		const typename Assembly_t::Scalar_t eps;
		const typename Assembly_t::Scalar_t modify_range;

		constexpr ModifiedFunctionKernelWeight(const Assembly_t& a, typename Assembly_t::Scalar_t e,
						typename Assembly_t::Scalar_t range = typename Assembly_t::Scalar_t{3}) noexcept
			: assembly(a), eps(e), modify_range(eps*range) {}

		template<typename QuadRule_t>
		ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			ScalarValueCache<QuadRule_t> result;
			auto gx=qr.geo_x(), gy=qr.geo_y(), gz=qr.geo_z();
			for (int i=0; i<N; ++i) {
				typename Assembly_t::Point_t p{gx[i], gy[i], gz[i]};
				auto sdf = assembly.signed_distance(p);
				if (gutil::abs(sdf)>modify_range) {
					//evaluate standard function
					result[i] = F(gx[i], gy[i], gz[i]);
				}
				else {
					//project to surface and evaluate
					auto grad_phi = assembly.heaviside_tanh_grad(p,eps);
					grad_phi = gutil::normalized(grad_phi);
					Scalar_t x = gx[i] + sdf*grad_phi[0];
					Scalar_t y = gy[i] + sdf*grad_phi[1];
					Scalar_t z = gz[i] + sdf*grad_phi[2];
					result[i] = F(x,y,z);
				}
			}
			return result;
		}
	};


	////////////////////////////////////////////////////////////////////////////////////////
	/// Allow some arithmetic with weights
	////////////////////////////////////////////////////////////////////////////////////////
	template<typename T, IsKernelWeight W>
	struct ScaledKernelWeight {
		static constexpr bool NEEDS_GEO_POINTS  = W::NEEDS_GEO_POINTS;
		static constexpr bool NEEDS_SCALAR_VALS = W::NEEDS_SCALAR_VALS;

		T scale;
		W weight;

		constexpr ScaledKernelWeight(T s, W w) : scale(s), weight(std::move(w)) {}
 
		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			return scale * weight.template build_weights<QuadRule_t>(scalar_field, qr);
		}
	};

	template<IsKernelWeight W1, IsKernelWeight W2>
	struct SumKernelWeight {
		static constexpr bool NEEDS_GEO_POINTS  = W1::NEEDS_GEO_POINTS || W2::NEEDS_GEO_POINTS;
		static constexpr bool NEEDS_SCALAR_VALS = W1::NEEDS_SCALAR_VALS || W2::NEEDS_SCALAR_VALS;

		W1 left;
		W2 right;

		constexpr SumKernelWeight(W1 l, W2 r) : left(std::move(l)), right(std::move(r)) {}
 
		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			return left.template build_weights<QuadRule_t>(scalar_field, qr) + right.template build_weights<QuadRule_t>(scalar_field, qr);
		}
	};

	template<IsKernelWeight W1, IsKernelWeight W2>
	struct ProductKernelWeight {
		static constexpr bool NEEDS_GEO_POINTS  = W1::NEEDS_GEO_POINTS || W2::NEEDS_GEO_POINTS;
		static constexpr bool NEEDS_SCALAR_VALS = W1::NEEDS_SCALAR_VALS || W2::NEEDS_SCALAR_VALS;

		W1 left;
		W2 right;

		constexpr ProductKernelWeight(W1 l, W2 r) : left(std::move(l)), right(std::move(r)) {}
 
		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			return left.template build_weights<QuadRule_t>(scalar_field, qr) * right.template build_weights<QuadRule_t>(scalar_field, qr);
		}
	};
}


