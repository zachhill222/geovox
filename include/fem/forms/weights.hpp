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
		{W::NEEDS_SDF_VALS}    -> std::convertible_to<bool>;
		{W::NEEDS_SDF_GRAD}    -> std::convertible_to<bool>;
		//also needs a templated build_weights(scalarvals, quad_rule)
		//the quad rule will hold geometric points while scalarvals will
		//point to the values of the scalar field at the quadrature points.
	};


	///////////////////////////////////////////////////////////////////
	/// Standard kernel weights
	///////////////////////////////////////////////////////////////////
	struct IdentityKernelWeight {
		//weight of a constant 1, probably shouldn't actually evaluate this ever.
		static constexpr bool NEEDS_GEO_POINTS  = false;
		static constexpr bool NEEDS_SCALAR_VALS = false;
		static constexpr bool NEEDS_SDF_VALS    = false;
		static constexpr bool NEEDS_SDF_GRAD    = false;

		template<typename QuadRule_t>
		static constexpr ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t&) noexcept {
			return ScalarValueCache<QuadRule_t>::Filled(typename QuadRule_t::Scalar_t{1});
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>*, const SdfGradCache<QuadRule_t>*,
				const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			return build_weights(nullptr, qr);
		}
	};

	template<auto F>
	struct FunctionWeight {
		static constexpr bool NEEDS_GEO_POINTS  = true;
		static constexpr bool NEEDS_SCALAR_VALS = false;
		static constexpr bool NEEDS_SDF_VALS    = false;
		static constexpr bool NEEDS_SDF_GRAD    = false;

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			ScalarValueCache<QuadRule_t> result;
			auto gx=qr.geo_x(), gy=qr.geo_y(), gz=qr.geo_z();

			GUTIL_SIMD()
			for (int i=0; i<N; ++i) { result[i] = static_cast<Scalar_t>(F(gx[i], gy[i], gz[i])); }
			
			return result;
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>*, const SdfGradCache<QuadRule_t>*,
				const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			return build_weights(nullptr, qr);
		}
	};


	///////////////////////////////////////////////////////////////////
	/// Diffuse domain weights
	///////////////////////////////////////////////////////////////////
	
	//a base class to staticly link the underlying geometry and transition width
	template<typename Assembly_t, int StaticID=0>
	struct AssemblyDiffuseDomain {
		inline static const Assembly_t* assembly{nullptr};
		inline static typename Assembly_t::Scalar_t eps{0};

		static void SetAssembly(const Assembly_t& a) {assembly = &a;}
		static void SetEps(typename Assembly_t::Scalar_t e) {eps = e;}

		template<typename QuadRule_t>
		static SdfValueCache<QuadRule_t> CacheSdfVals(const QuadRule_t& qr) noexcept {
			GUTIL_ASSERT(assembly);
			return {*assembly, qr};
		}

		template<typename QuadRule_t>
		static SdfGradCache<QuadRule_t> CacheSdfGrad(const QuadRule_t& qr) noexcept {
			GUTIL_ASSERT(assembly);
			return {*assembly, qr};
		}

		template<typename QuadRule_t>
		static void CacheSdfValsGrad(SdfValueCache<QuadRule_t>& sdf, SdfGradCache<QuadRule_t>& grad, const QuadRule_t& qr) {
			GUTIL_ASSERT(assembly);
			constexpr auto N = QuadRule_t::TOTAL_QUAD_POINTS;
			assembly->signed_distance({sdf.data(), N}, {grad.data(), 3*N}, qr.geo_x(), qr.geo_y(), qr.geo_z());
		}
	};



	template<bool Interior=true, typename Assembly_t=void, int StaticID=0>
	struct AssemblyPhaseFieldWeight : public AssemblyDiffuseDomain<Assembly_t,StaticID> {
		//phi or 1-phi using the assembly tanh heaviside approximation
		static constexpr bool NEEDS_GEO_POINTS  = true;
		static constexpr bool NEEDS_SCALAR_VALS = false;
		static constexpr bool NEEDS_SDF_VALS    = true;
		static constexpr bool NEEDS_SDF_GRAD    = false;

		static constexpr bool INTERIOR = Interior;
		using DiffuseDomain = AssemblyDiffuseDomain<Assembly_t,StaticID>;
		using BASE = AssemblyDiffuseDomain<Assembly_t,StaticID>;
		using BASE::assembly;
		using BASE::eps;
		
		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			SdfValueCache<QuadRule_t> sdf(*assembly,qr);
			return build_weights_cached_sdf<QuadRule_t>(&sdf, nullptr, nullptr, qr);
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>* sdf, const SdfGradCache<QuadRule_t>*,
				const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			
			ScalarValueCache<QuadRule_t> result;
			Assembly_t::heaviside_tanh({result.data(),N}, {sdf->data(),N}, eps);

			if constexpr(!INTERIOR){
				GUTIL_SIMD()
				for (int i=0; i<N; ++i) {
					result[i] = Scalar_t{1} - result[i];
				}
			}
			
			return result;
		}
	};


	template<typename Assembly_t, auto F, int StaticID=0>
	struct ModifiedFunctionKernelWeight : public AssemblyDiffuseDomain<Assembly_t,StaticID> {
		static constexpr bool NEEDS_GEO_POINTS  = true;
		static constexpr bool NEEDS_SCALAR_VALS = false;
		static constexpr bool NEEDS_SDF_VALS    = true;
		static constexpr bool NEEDS_SDF_GRAD    = true;

		using DiffuseDomain = AssemblyDiffuseDomain<Assembly_t,StaticID>;
		using BASE = AssemblyDiffuseDomain<Assembly_t,StaticID>;
		using BASE::assembly;
		using BASE::eps;

		inline static typename Assembly_t::Scalar_t modify_range_scale;
		static void SetModifyRangeScale(typename Assembly_t::Scalar_t s) noexcept {modify_range_scale = s;}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			static_assert( std::same_as<typename QuadRule_t::Scalar_t, typename Assembly_t::Scalar_t> );
			
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			SdfValueCache<QuadRule_t> sdf;
			SdfGradCache<QuadRule_t> grad;
			BASE::CacheSdfValsGrad(sdf,grad,qr);
			return build_weights_cached_sdf<QuadRule_t>(&sdf, &grad, nullptr, qr);
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>* sdf, const SdfGradCache<QuadRule_t>* grad,
				const ScalarValueCache<QuadRule_t>*, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			const typename Assembly_t::Scalar_t modify_range = modify_range_scale*eps;

			ScalarValueCache<QuadRule_t> result;
			GUTIL_SIMD()	//for most elements, only one branch is needed
			for (int i=0; i<N; ++i) {
				if (gutil::abs((*sdf)[i]) > modify_range) {
					result[i] = F(qr.geo_x()[i], qr.geo_y()[i], qr.geo_z()[i]);
				}
				else {
					const Scalar_t N2 = (*grad)[0][i]*(*grad)[0][i] + (*grad)[1][i]*(*grad)[1][i] + (*grad)[2][i]*(*grad)[2][i];
					const Scalar_t scale = (*sdf)[i]/gutil::sqrt(N2);
					result[i] = F(qr.geo_x()[i] + scale * (*grad)[0][i],
								  qr.geo_y()[i] + scale * (*grad)[1][i],
								  qr.geo_z()[i] + scale * (*grad)[2][i]);
				}
			}
			return result;
		}
	};


	////////////////////////////////////////////////////////////////////////////////////////
	/// 'Nonlinear' weights for error computation or IMEX schemes
	////////////////////////////////////////////////////////////////////////////////////////


	///////////////////////////////////////////////////////////////////
	/// Computes phi*(u_h - u_exact)^2 at each quadrature point -- the
	/// diffuse-domain L2 error density, using phi as the norm weight
	/// (matching the E_L2 = ||chi_Omega*(u-u_h)||_L2 convention, with
	/// phi approximating chi_Omega). NEEDS_SCALAR_VALS=true means whatever
	/// assembly loop calls build_weights must reconstruct u_h and pass it
	/// in as scalar_field -- this weight never reconstructs it itself.
	///////////////////////////////////////////////////////////////////
	template<typename Assembly_t, auto UExact, int StaticID=0>
	struct L2DiffuseErrorWeight : public AssemblyDiffuseDomain<Assembly_t,StaticID> {
		static constexpr bool NEEDS_GEO_POINTS  = true;
		static constexpr bool NEEDS_SCALAR_VALS = true;
		static constexpr bool NEEDS_SDF_VALS    = true;
		static constexpr bool NEEDS_SDF_GRAD    = false;

		using DiffuseDomain = AssemblyDiffuseDomain<Assembly_t,StaticID>;
		using BASE = AssemblyDiffuseDomain<Assembly_t,StaticID>;
		using BASE::assembly;
		using BASE::eps;

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			SdfValueCache<QuadRule_t> sdf(*assembly, qr);
			return build_weights_cached_sdf<QuadRule_t>(&sdf, nullptr, scalar_field, qr);
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>* sdf, const SdfGradCache<QuadRule_t>*,
				const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			using Scalar_t = typename QuadRule_t::Scalar_t;
			constexpr int N = QuadRule_t::TOTAL_QUAD_POINTS;
			GUTIL_ASSERT(scalar_field);   // NEEDS_SCALAR_VALS=true means this must be provided by the caller

			ScalarValueCache<QuadRule_t> phi;
			Assembly_t::heaviside_tanh({phi.data(),N}, {sdf->data(),N}, eps);

			auto gx=qr.geo_x(), gy=qr.geo_y(), gz=qr.geo_z();
			ScalarValueCache<QuadRule_t> result;
			GUTIL_SIMD()
			for (int i=0; i<N; ++i) {
				const Scalar_t diff = (*scalar_field)[i] - static_cast<Scalar_t>(UExact(gx[i], gy[i], gz[i]));
				result[i] = phi[i] * diff * diff;
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
		static constexpr bool NEEDS_SDF_VALS    = W::NEEDS_SDF_VALS;
		static constexpr bool NEEDS_SDF_GRAD    = W::NEEDS_SDF_GRAD;

		using DiffuseDomain = std::conditional_t< (NEEDS_SDF_VALS||NEEDS_SDF_GRAD), typename W::DiffuseDomain, void>;

		T scale{1};
		W weight{};

		constexpr ScaledKernelWeight() {}
		constexpr ScaledKernelWeight(T s) : scale(s) {}
		constexpr ScaledKernelWeight(T s, W w) : scale(s), weight(std::move(w)) {}
 
		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			return scale * weight.template build_weights<QuadRule_t>(scalar_field, qr);
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>* sdf, const SdfGradCache<QuadRule_t>* grad,
				const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept requires (NEEDS_SDF_VALS || NEEDS_SDF_GRAD) {
			return scale * weight.template build_weights_cached_sdf<QuadRule_t>(sdf,grad,scalar_field,qr);
		}
	};

	template<IsKernelWeight W1, IsKernelWeight W2>
	struct SumKernelWeight {
		static constexpr bool NEEDS_GEO_POINTS  = W1::NEEDS_GEO_POINTS  || W2::NEEDS_GEO_POINTS;
		static constexpr bool NEEDS_SCALAR_VALS = W1::NEEDS_SCALAR_VALS || W2::NEEDS_SCALAR_VALS;
		static constexpr bool NEEDS_SDF_VALS    = W1::NEEDS_SDF_VALS    || W2::NEEDS_SDF_VALS;
		static constexpr bool NEEDS_SDF_GRAD    = W1::NEEDS_SDF_GRAD    || W2::NEEDS_SDF_GRAD;

		using DiffuseDomain = std::conditional_t< (W1::NEEDS_SDF_VALS||W1::NEEDS_SDF_GRAD), typename W1::DiffuseDomain, 
									std::conditional_t< (W2::NEEDS_SDF_VALS||W2::NEEDS_SDF_GRAD), typename W2::DiffuseDomain, void>>;

		W1 left{};
		W2 right{};

		constexpr SumKernelWeight() {}
		constexpr SumKernelWeight(W1 l, W2 r) : left(std::move(l)), right(std::move(r)) {}
 		
		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			if constexpr (NEEDS_SDF_VALS && NEEDS_SDF_GRAD) {
				SdfValueCache<QuadRule_t> sdf;
				SdfGradCache<QuadRule_t> grad;
				DiffuseDomain::CacheSdfValsGrad(sdf,grad,qr);
				return build_weights_cached_sdf<QuadRule_t>(&sdf, &grad, scalar_field, qr);
			}
			else if constexpr (NEEDS_SDF_VALS) {
				SdfValueCache<QuadRule_t> sdf = DiffuseDomain::CacheSdfVals(qr);
				return build_weights_cached_sdf<QuadRule_t>(&sdf, nullptr, scalar_field, qr);
			}
			else if constexpr (NEEDS_SDF_GRAD) {
				SdfGradCache<QuadRule_t> grad = DiffuseDomain::CacheSdfGrad(qr);
				return build_weights_cached_sdf<QuadRule_t>(nullptr, &grad, scalar_field, qr);
			}
			else {
				return left.template build_weights<QuadRule_t>(scalar_field, qr) + right.template build_weights<QuadRule_t>(scalar_field, qr);
			}
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>* sdf, const SdfGradCache<QuadRule_t>* grad,
				const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept requires (NEEDS_SDF_VALS || NEEDS_SDF_GRAD) {
			return left.template build_weights_cached_sdf<QuadRule_t>(sdf, grad, scalar_field, qr) + 
					right.template build_weights_cached_sdf<QuadRule_t>(sdf, grad, scalar_field, qr);
		}
	};

	template<IsKernelWeight W1, IsKernelWeight W2>
	struct ProductKernelWeight {
		static constexpr bool NEEDS_GEO_POINTS  = W1::NEEDS_GEO_POINTS  || W2::NEEDS_GEO_POINTS;
		static constexpr bool NEEDS_SCALAR_VALS = W1::NEEDS_SCALAR_VALS || W2::NEEDS_SCALAR_VALS;
		static constexpr bool NEEDS_SDF_VALS    = W1::NEEDS_SDF_VALS    || W2::NEEDS_SDF_VALS;
		static constexpr bool NEEDS_SDF_GRAD    = W1::NEEDS_SDF_GRAD    || W2::NEEDS_SDF_GRAD;
		
		using DiffuseDomain = std::conditional_t< (W1::NEEDS_SDF_VALS||W1::NEEDS_SDF_GRAD), typename W1::DiffuseDomain, 
									std::conditional_t< (W2::NEEDS_SDF_VALS||W2::NEEDS_SDF_GRAD), typename W2::DiffuseDomain, void>>;

		
		W1 left{};
		W2 right{};

		constexpr ProductKernelWeight() {}
		constexpr ProductKernelWeight(W1 l, W2 r) : left(std::move(l)), right(std::move(r)) {}
 
		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights(const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept {
			if constexpr (NEEDS_SDF_VALS && NEEDS_SDF_GRAD) {
				SdfValueCache<QuadRule_t> sdf;
				SdfGradCache<QuadRule_t> grad;
				DiffuseDomain::CacheSdfValsGrad(sdf,grad,qr);
				return build_weights_cached_sdf<QuadRule_t>(&sdf, &grad, scalar_field, qr);
			}
			else if constexpr (NEEDS_SDF_VALS) {
				SdfValueCache<QuadRule_t> sdf = DiffuseDomain::CacheSdfVals(qr);
				return build_weights_cached_sdf<QuadRule_t>(&sdf, nullptr, scalar_field, qr);
			}
			else if constexpr (NEEDS_SDF_GRAD) {
				SdfGradCache<QuadRule_t> grad = DiffuseDomain::CacheSdfGrad(qr);
				return build_weights_cached_sdf<QuadRule_t>(nullptr, &grad, scalar_field, qr);
			}
			else {
				return left.template build_weights<QuadRule_t>(scalar_field, qr) * right.template build_weights<QuadRule_t>(scalar_field, qr);
			}
		}

		template<typename QuadRule_t>
		[[nodiscard]] ScalarValueCache<QuadRule_t> build_weights_cached_sdf(const SdfValueCache<QuadRule_t>* sdf, const SdfGradCache<QuadRule_t>* grad,
				const ScalarValueCache<QuadRule_t>* scalar_field, const QuadRule_t& qr) const noexcept requires (NEEDS_SDF_VALS || NEEDS_SDF_GRAD) {
			return left.template build_weights_cached_sdf<QuadRule_t>(sdf, grad, scalar_field, qr) * 
					right.template build_weights_cached_sdf<QuadRule_t>(sdf, grad, scalar_field, qr);
		}
	};


	/////////////////////////////////////////////////////////////////////////////////////
	/// Implement the operators
	/////////////////////////////////////////////////////////////////////////////////////
	template<typename T, IsKernelWeight W>
	[[nodiscard]] constexpr ScaledKernelWeight<T,W> operator*(T scale, W weight) noexcept {
		return ScaledKernelWeight<T,W>{scale, weight};
	}

	template<IsKernelWeight W1, IsKernelWeight W2>
	[[nodiscard]] constexpr ProductKernelWeight<W1,W2> operator*(W1 left, W2 right) noexcept {
		return ProductKernelWeight<W1,W2>{left, right};
	}

	template<IsKernelWeight W1, IsKernelWeight W2>
	[[nodiscard]] constexpr SumKernelWeight<W1,W2> operator+(W1 left, W2 right) noexcept {
		return SumKernelWeight<W1,W2>{left, right};
	}

}


