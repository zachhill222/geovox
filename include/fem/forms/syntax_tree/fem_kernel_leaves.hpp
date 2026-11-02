#pragma once

#include "gutil.hpp"
#include "fem/forms/syntax_tree/argument_tags.hpp"
#include "fem/forms/syntax_tree/fem_kernel_tree.hpp"

#include <concepts>

namespace GV {
namespace FemKernel {

	//////////////////////////////////////////////////////////////////////////
	/// Starting types for building bilinear/linear forms from dof_handlers (function spaces)
	///
	/// Form arguments are read from right to left. If A(u,v)=f(v) is the weak form,
	/// then v is argument/Role 0 for both A and f.
	///
	/// Role=0 -> test
	/// Role=1 -> trial
	/// Role-2 -> tertiary argument
	///
	/// When pointing data_ to the values, note that vectors and matrices must be stored
	/// in groups of components. For example, gradient data must be stored as:
	/// [ <grad_x at each quadrature point> <grad_y at each quad. point> <grad_z at each quad. point> ]
	///
	/// Similarly, the geometric points (where needed) are stored [ x coords, y coords, z coords ]
	//////////////////////////////////////////////////////////////////////////

	template<typename Scalar>
	struct KernelLeafDataStorage {
		const Scalar* data_ = nullptr;
		constexpr void set_data(const Scalar* ptr, KernelArgumentTag tag) noexcept {
			data_ = ptr;
		}
		[[nodiscard]] constexpr const Scalar* data() const noexcept {
			return data_;
		}
	};

	template<typename Scalar>
	struct KernelGeoDataStorage {
		const Scalar* geo_data_ = nullptr;
		constexpr void set_geo_data(const Scalar* ptr) noexcept {
			geo_data_ = ptr;
		}
		[[nodiscard]] constexpr const Scalar* geo_data() const noexcept {
			return geo_data_;
		}
	};


	template<typename DOF_t, uint8_t Role, typename Scalar=double>
	struct Value : KernelLeafDataStorage<Scalar> {
		using Scalar_t = Scalar;

		static constexpr KernelTag TAG = [](){
			KernelTag t{};
			t.arg_tag(Role).set_needs_val(true);
			t.arg_tag(Role).set_used(true);
			t.arg_tag(Role).set_arg_no(Role);
			t.set_const(false);
			t.set_rank(0);
			t.set_dim(DOF_t::DIMENSION);
			t.set_needs_geo(false);
			return t;
		}();

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			GUTIL_ASSERT(this->data_);
			return this->data_[quad_point];
		}
	};


	template<typename DOF_t, uint8_t Role, typename Scalar=double>
	struct Grad : KernelLeafDataStorage<Scalar> {
		using Scalar_t = Scalar;

		static constexpr KernelTag TAG = [](){
			KernelTag t{};
			t.arg_tag(Role).set_needs_val(false);
			t.arg_tag(Role).set_needs_grad(true);
			t.arg_tag(Role).set_used(true);
			t.arg_tag(Role).set_arg_no(Role);
			t.set_const(false);
			t.set_rank(1);
			t.set_dim(DOF_t::DIMENSION);
			t.set_needs_geo(false);
			return t;
		}();

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			GUTIL_ASSERT(this->data_);
			return this->data_[NQ*component + quad_point];
		}
	};


	template<typename DOF_t, uint8_t Role, size_t Axis, typename Scalar=double> requires(Axis < (size_t) DOF_t::DIMENSION)
	struct Partial : KernelLeafDataStorage<Scalar> {
		using Scalar_t = Scalar;

		static constexpr KernelTag TAG = [](){
			KernelTag t{};
			t.arg_tag(Role).set_needs_val(false);
			t.arg_tag(Role).set_needs_grad(true);
			t.arg_tag(Role).set_used(true);
			t.arg_tag(Role).set_arg_no(Role);
			t.set_const(false);
			t.set_rank(0);
			t.set_dim(DOF_t::DIMENSION);
			t.set_needs_geo(false);
			return t;
		}();

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			//set data identical to the gradient data
			GUTIL_ASSERT(this->data_);
			return this->data_[NQ*Axis + quad_point];
		}
	};


	//////////////////////////////////////////////////////////////////////////
	/// Starting types for building forms from lambda captures
	/// This can be used in linear forms for weights or non-linear forms as an exact solution
	/// These do not correspond to any argument of a form.
	///
	/// For example, to compute an L2 weighted exact error, E(v) = int( (v-f)^2 * w), both f and w
	/// should be lambdas, and the final form has one Value type argument.
	///
	/// For an approximate L2 weighted error, E(u,v) = int( (u-v)^2 * w), u and v can both come from
	/// dof_handlers and w is a lambda.
	///
	/// Use stateless captures of the form: lambda(size_t component, T x,T y,T z) -> T
	///
	/// Form arguments are read from right to left. If A(u,v)=f(v) is the weak form,
	/// then v is argument/Role 0 for both A and f.
	//////////////////////////////////////////////////////////////////////////
	template<typename Lambda, uint8_t Rank, typename Scalar=double> requires(std::is_empty_v<Lambda>)
	struct StaticGeometricFunction : KernelGeoDataStorage<Scalar> {
		using Scalar_t = Scalar;
		static constexpr uint8_t DIM = std::is_invocable_r_v<Scalar,Lambda,size_t,Scalar> ? 1 :
									   std::is_invocable_r_v<Scalar,Lambda,size_t,Scalar,Scalar> ? 2 :
									   std::is_invocable_r_v<Scalar,Lambda,size_t,Scalar,Scalar,Scalar> ? 3 : 4;
		static_assert(DIM<4);

		static constexpr KernelTag TAG = [](){
			KernelTag t{};
			t.set_rank(Rank);
			t.set_dim(DIM);
			t.set_const(false);
			t.set_needs_geo(true);
			return t;
		}();

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			GUTIL_ASSERT(this->geo_data_);
			if constexpr (DIM==1) {return lambda(component, this->geo_data_[quad_point]);}
			else if constexpr (DIM==2) {return lambda(component, this->geo_data_[quad_point], this->geo_data_[NQ+quad_point]);}
			else {return Lambda{}(component, this->geo_data_[quad_point], this->geo_data_[NQ+quad_point], this->geo_data_[2*NQ+quad_point]);}
		}
	};


	template<typename Lambda, uint8_t Rank, typename Scalar=double> requires(!std::is_empty_v<Lambda>)
	struct RuntimeGeometricFunction : KernelGeoDataStorage<Scalar> {
		using Scalar_t = Scalar;
		static constexpr uint8_t DIM = std::is_invocable_r_v<Scalar,Lambda,size_t,Scalar> ? 1 :
									   std::is_invocable_r_v<Scalar,Lambda,size_t,Scalar,Scalar> ? 2 :
									   std::is_invocable_r_v<Scalar,Lambda,size_t,Scalar,Scalar,Scalar> ? 3 : 4;
		static_assert(DIM<4);

		static constexpr KernelTag TAG = [](){
			KernelTag t{};
			t.set_rank(Rank);
			t.set_dim(DIM);
			t.set_const(false);
			t.set_needs_geo(true);
			return t;
		}();

		//point to x, y, z values in contiguous blocks
		Lambda lambda;
		
		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			GUTIL_ASSERT(this->geo_data_);
			if constexpr (DIM==1) {return lambda(component, this->geo_data_[quad_point]);}
			else if constexpr (DIM==2) {return lambda(component, this->geo_data_[quad_point], this->geo_data_[NQ+quad_point]);}
			else {return lambda(component, this->geo_data_[quad_point], this->geo_data_[NQ+quad_point], this->geo_data_[2*NQ+quad_point]);}
		}
	};



	//////////////////////////////////////////////////////////////////////////
	/// Fallback type where the user will supply the pointer to the data.
	/// Convenient for constant weights or weights the are expensive and infrequently computed
	///
	/// Note that the Const flag is per-element, not global. If it is globally constant, 
	/// you may want to use the static variant.
	///
	/// Note when using multiple StaticWeights, if the standard template parameters are
	/// the same, make sure to set a different StaticID parameter to avoid two different
	/// weights using the same data.
	//////////////////////////////////////////////////////////////////////////
	template<uint8_t Rank, uint8_t Dim, bool Const, typename Scalar=double>
	struct Weight : KernelLeafDataStorage<Scalar> {
		using Scalar_t = Scalar;
		static constexpr KernelTag TAG = [](){
			KernelTag t{};
			t.set_rank(Rank);
			t.set_dim(Dim);
			t.set_const(Const);
			t.set_needs_geo(false);
			return t;
		}();

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] constexpr auto at(size_t component, size_t quad_point) const noexcept {
			GUTIL_ASSERT(this->data_);
			if constexpr (Const) {return this->data_[component];}
			else {return this->data_[component*NQ + quad_point];}
		}
	};


	template<uint8_t Rank, uint8_t Dim, bool Const, typename Scalar=double, int StaticID=0>
	struct StaticWeight {
		using Scalar_t = Scalar;
		static constexpr KernelTag TAG = [](){
			KernelTag t{};
			t.set_rank(Rank);
			t.set_dim(Dim);
			t.set_const(Const);
			t.set_needs_geo(false);
			return t;
		}();


		static void set_data(const Scalar* ptr, KernelArgumentTag tag) noexcept {
			GUTIL_ASSERT( (TAG&tag) == tag );
			data_ = ptr;
		}
		[[nodiscard]] static constexpr const Scalar* data() noexcept {
			return data_;
		}

		GUTIL_DECLARE_SIMD()
		template<size_t NQ>
		[[nodiscard]] static constexpr auto at(size_t component, size_t quad_point) noexcept {
			GUTIL_ASSERT(data_);
			if constexpr (Const) {return data_[component];}
			else {return data_[component*NQ + quad_point];}
		}

	private:
		inline static const Scalar* data_;
	};
}}