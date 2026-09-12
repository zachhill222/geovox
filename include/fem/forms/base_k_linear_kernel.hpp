#pragma once


#include "gutil.hpp"
#include "fem/forms/util.hpp"
#include "fem/forms/weights.hpp"

namespace GV {
	template<typename Kernel>
	concept IsKLinearKernel = requires (Kernel k) {
		{Kernel::K}            -> std::convertible_to<size_t>;
		{Kernel::NEEDS_WEIGHT} -> std::convertible_to<bool>;
		{Kernel::N_WEIGHTS}    -> std::convertible_to<size_t>;	//used for building k1(u,v)*w1(x) + k2(u,v)*w2(x) as a single kernel.
		{Kernel::N_SYMMETRIC}  -> std::convertible_to<size_t>;
		{Kernel::IS_SYMMETRIC} -> std::convertible_to<bool>;
		{Kernel::NEED_VALS}    -> std::convertible_to<std::array<bool,Kernel::K>>;
		{Kernel::NEED_GRAD}    -> std::convertible_to<std::array<bool,Kernel::K>>;
		typename Kernel::WeightType;
	};


	//////////////////////////////////////////////////////////////////
	/// A base class for K-linear kernels to use in K-linear forms.
	/// Specific kernels (eg L2Linear, L2Bilinear, ...) should inherit from this
	/// so generic queries about what values need to be computed.
	/// For example, an HdivBilinear form may need the dof values of one
	/// dof and the gradient of the other.
	///
	/// Note that the first NSymmetric arguments must be interchangable.
	/// For example if the trilinear kernel is grad(phi)*grad(psi)*mu,
	/// (corresponding to a trilinear form K(u,v,w) = int_D(grad(u)*grad(v)*w)),
	/// then the phi/psi should be the first two arguments and NSymmetric should be set to 2.
	/// Note that if mu/w does not have a corresponding scalar field/dofhander,
	/// then it should be replaced with a weight and the weighted H1Bilinear
	/// kernel used instead.
	///
	/// Note that a 0-form can be used to integrate a weight function.
	///
	/// Use CRTP so that kernels can be added and scaled. Note that only unweighted
	/// kernels can be added. Examples of the Zero and Identity kernels are provided below.
	///
	/// Note for a bilinear form B(u,v) = int_D( kernel(u,v) ), we read the indices from right
	/// to left (i.e., v is index 0 and u is index 1). This allows us to keep consistent indexing
	/// through various types and have column-major storage indexing be the natural order.
	//////////////////////////////////////////////////////////////////
	template<size_t K_, typename Weight, size_t NSymmetric, typename Derived> requires (NSymmetric <= K_)
	struct KLinearKernel {
		static_assert(std::same_as<Weight,void> || IsKernelWeight<Weight>);
		//////////////////////////////////////////////////////////////////
		/// Track essential constants.
		//////////////////////////////////////////////////////////////////
		static constexpr size_t K            = K_;
		static constexpr size_t N_WEIGHTS    = WeightCountOf<Weight>();
		static constexpr bool   NEEDS_WEIGHT = N_WEIGHTS>0;
		static constexpr size_t N_SYMMETRIC  = NSymmetric;
		static constexpr bool   IS_SYMMETRIC = N_SYMMETRIC==K;
		using WeighType = Weight;
		static_assert(N_SYMMETRIC!=1, "A single symmetric argument makes no sense.");

		//////////////////////////////////////////////////////////////////
		/// Track if a form needs dof values/gradients for each argument
		//////////////////////////////////////////////////////////////////
		[[nodiscard]] static constexpr bool IsValid() noexcept {
			static_assert(std::same_as<std::array<bool,K>, std::remove_cvref_t<decltype(Derived::NEED_VALS)>>,
				"A K-linear form must state which arguments need dof values");
			static_assert(std::same_as<std::array<bool,K>, std::remove_cvref_t<decltype(Derived::NEED_GRAD)>>,
				"A K-linear form must state which arguments need dof gradients");
			return true;
		}

		//////////////////////////////////////////////////////////////////
		/// Track the evaluation signature and underlying data types.
		/// The types are templated on a quadrature rule.
		//////////////////////////////////////////////////////////////////
		template<typename QR>
		using ValueArgType = std::array<const DofValueCache<QR>*, K>;

		template<typename QR>
		using ValueArg = const std::array<const DofValueCache<QR>*, K>&;

		template<typename QR>
		using GradArgType = std::array<const DofGradCache<QR>*, K>;

		template<typename QR>
		using GradArg = const std::array<const DofGradCache<QR>*, K>&;

		template<typename QR>
		using WeightArgType = std::conditional_t<(N_WEIGHTS<=1), ScalarValueCache<QR>, std::span<const ScalarValueCache<QR>*, N_WEIGHTS>>;

		template<typename QR>
		using WeightArg = std::conditional_t<(N_WEIGHTS<=1), const ScalarValueCache<QR>*, std::span<const ScalarValueCache<QR>*, N_WEIGHTS>>;

		template<typename QR>
		[[nodiscard]] static constexpr bool IsValArgValid(ValueArg<QR> arg) noexcept {
			bool flag = true;
			for (size_t k=0; k<K; ++k) {
				if (Derived::NEED_VALS[k] && arg[k]==nullptr) {
					GUTIL_ERROR("Value argument ", k, " is nullptr but is required");
					flag = false;
				}
			}
			return flag;
		}

		template<typename QR>
		[[nodiscard]] static constexpr bool IsGradArgValid(GradArg<QR> arg) noexcept {
			bool flag = true;
			for (size_t k=0; k<K; ++k) {
				if (Derived::NEED_GRAD[k] && arg[k]==nullptr) {
					GUTIL_ERROR("Grad argument ", k, " is nullptr but is required");
					flag = false;
				}
			}
			return flag;
		}

		template<typename QR>
		[[nodiscard]] static constexpr bool IsWeightArgValid(WeightArg<QR> arg) noexcept {
			if constexpr (N_WEIGHTS==1) {return arg!=nullptr;}
			else {return true;}
		}


		//////////////////////////////////////////////////////////////////
		/// Check if the kernel can be evaluated statically. If it can,
		/// use default to static evaluation.
		//////////////////////////////////////////////////////////////////
		template<typename QR>
		[[nodiscard]] static constexpr bool HasStaticEval() noexcept requires( requires {
			Derived::template CachedEvalImpl<QR>(std::declval<ValueArg<QR>>(), std::declval<GradArg<QR>>(), std::declval<WeightArg<QR>>(), std::declval<const QR&>());
		}) {return true;}
		template<typename QR>
		[[nodiscard]] static constexpr bool HasStaticEval() noexcept requires( !requires {
			Derived::template CachedEvalImpl<QR>(std::declval<ValueArg<QR>>(), std::declval<GradArg<QR>>(), std::declval<WeightArg<QR>>(), std::declval<const QR&>());
		}) {return false;}


		//////////////////////////////////////////////////////////////////
		/// Convert to Derived at runtime if needed.
		//////////////////////////////////////////////////////////////////
		[[nodiscard]] const Derived* derived() const noexcept {return static_cast<const Derived*>(this);}
		[[nodiscard]] Derived* derived() noexcept {return static_cast<Derived*>(this);}


		//////////////////////////////////////////////////////////////////
		/// Forward evaluation to the Derived class for static evaluation.
		//////////////////////////////////////////////////////////////////
		template<typename QR>
		[[nodiscard]] static typename QR::Scalar_t CachedEval(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) noexcept requires(HasStaticEval<QR>()) {
			// GUTIL_PROFILE_FUNCTION();
			static_assert(IsValid());
			GUTIL_ASSERT(IsValArgValid(vals));
			GUTIL_ASSERT(IsGradArgValid(grad));
			GUTIL_ASSERT(IsWeightArgValid(wt_ptr));
			return Derived::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}


		//////////////////////////////////////////////////////////////////
		/// Forward evaluation to the Derived class for runtime evaluation.
		//////////////////////////////////////////////////////////////////
		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires (HasStaticEval<QR>()) {	
			// GUTIL_PROFILE_FUNCTION();
			static_assert(IsValid());
			GUTIL_ASSERT(IsValArgValid(vals));
			GUTIL_ASSERT(IsGradArgValid(grad));
			GUTIL_ASSERT(IsWeightArgValid(wt_ptr));
			return Derived::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}
		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires (!HasStaticEval<QR>()) {	
			// GUTIL_PROFILE_FUNCTION();
			static_assert(IsValid());
			GUTIL_ASSERT(IsValArgValid(vals));
			GUTIL_ASSERT(IsGradArgValid(grad));
			GUTIL_ASSERT(IsWeightArgValid(wt_ptr));
			return derived()->cached_eval_impl(vals, grad, wt_ptr, qr);
		}
	};


	//////////////////////////////////////////////////////////////////
	/// Make a few helper functions for adding and scaling kernels.
	//////////////////////////////////////////////////////////////////
	template<size_t K>
	struct ZeroKernel : public KLinearKernel<K, void, K, ZeroKernel<K>> {
		using BASE = KLinearKernel<K,void,K,ZeroKernel<K>>;
		template<typename QR>
		using ValueArgType = BASE::template ValueArgType<QR>;
		template<typename QR>
		using GradArgType = BASE::template GradArgType<QR>;
		template<typename QR>
		using WeightArgType = BASE::template WeightArgType<QR>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;
		using WeightType = void;

		using BASE::BASE;

		static constexpr std::array<bool,K> NEED_VALS = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();
		static constexpr std::array<bool,K> NEED_GRAD = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();

		template<typename QR>
		[[nodiscard]] static typename QR::Scalar_t CachedEvalImpl(ValueArg<QR>, GradArg<QR>, WeightArg<QR>, const QR&) noexcept {
			return typename QR::Scalar_t{0};
		}
	};

	template<size_t K, typename Weight=void>
	struct IdentityKernel : public KLinearKernel<K, Weight, K, IdentityKernel<K,Weight>> {
		using BASE = KLinearKernel<K,Weight,K,IdentityKernel<K,Weight>>;
		template<typename QR>
		using ValueArgType = BASE::template ValueArgType<QR>;
		template<typename QR>
		using GradArgType = BASE::template GradArgType<QR>;
		template<typename QR>
		using WeightArgType = BASE::template WeightArgType<QR>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;
		using WeightType = Weight;

		using BASE::BASE;

		static constexpr std::array<bool,K> NEED_VALS = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();
		static constexpr std::array<bool,K> NEED_GRAD = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();

		template<typename QR>
		[[nodiscard]] static typename QR::Scalar_t CachedEvalImpl(ValueArg<QR>, GradArg<QR>, WeightArg<QR>, const QR& qr) noexcept requires(!BASE::NEEDS_WEIGHT) {
			return QR::quad_w_sum() * qr.jac_det();
		}

		template<typename QR>
		[[nodiscard]] static typename QR::Scalar_t CachedEvalImpl(ValueArg<QR>, GradArg<QR>, WeightArg<QR> wt_ptr, const QR& qr) noexcept requires(BASE::NEEDS_WEIGHT) {
			GUTIL_ASSERT(wt_ptr);
			constexpr int N = QR::TOTAL_QUAD_POINTS;
			typename QR::Scalar_t val{0};
			static constexpr auto qw = QR::quad_w();
			GUTIL_SIMD(reduction(+:val))
			for (int i=0; i<N; ++i) {
				val += (*wt_ptr)[i] * qw[i];
			}
			return val * qr.jac_det();
		}
	};



	//////////////////////////////////////////////////////////////////
	/// Make a few helper functions for adding and scaling kernels.
	//////////////////////////////////////////////////////////////////
	template<size_t K>
	static constexpr std::array<bool,K> OrArrays(std::array<bool,K> a, std::array<bool,K> b) noexcept {
		std::array<bool,K> c{};
		for (size_t k=0; k<K; ++k) {c[k] = a[k] || b[k];}
		return c;
	}

	template<size_t K>
	static constexpr std::array<bool,K> AndArrays(std::array<bool,K> a, std::array<bool,K> b) noexcept {
		std::array<bool,K> c{};
		for (size_t k=0; k<K; ++k) {c[k] = a[k] && b[k];}
		return c;
	}


	//////////////////////////////////////////////////////////////////
	/// A scaled K-linear kernel
	//////////////////////////////////////////////////////////////////
	template<typename T, IsKLinearKernel Kernel>
	struct ScaledKernel : KLinearKernel<Kernel::K, typename Kernel::WeightType, Kernel::N_SYMMETRIC, ScaledKernel<T,Kernel>> {
		using BASE = KLinearKernel<Kernel::K, typename Kernel::WeightType, Kernel::N_SYMMETRIC, ScaledKernel<T,Kernel>>;
		template<typename QR>
		using ValueArgType = BASE::template ValueArgType<QR>;
		template<typename QR>
		using GradArgType = BASE::template GradArgType<QR>;
		template<typename QR>
		using WeightArgType = BASE::template WeightArgType<QR>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;
		using WeightType = typename Kernel::WeightType;

		using BASE::BASE;
		constexpr ScaledKernel(T s, Kernel k = Kernel{}) : BASE(), scale{s}, kernel{std::move(k)} {}

		static constexpr std::array<bool,Kernel::K> NEED_VALS = Kernel::NEED_VALS;
		static constexpr std::array<bool,Kernel::K> NEED_GRAD = Kernel::NEED_GRAD;
		T scale{1};
		[[no_unique_address]] Kernel kernel{};

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(Kernel::template HasStaticEval<QR>()){
			return scale * Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(!Kernel::template HasStaticEval<QR>()){
			return scale * kernel.cached_eval_impl(vals, grad, wt_ptr, qr);
		}
	};

	template<typename T, IsKLinearKernel Kernel>
	[[nodiscard]] inline constexpr ScaledKernel<T,Kernel> MakeScaledKernel(T scale, Kernel k = Kernel{}) noexcept {return {scale,std::move(k)}; }

	template<typename T, IsKLinearKernel Kernel>
	[[nodiscard]] inline constexpr ScaledKernel<T,Kernel> operator*(T scale, Kernel k) noexcept {return {scale,std::move(k)}; }


	//////////////////////////////////////////////////////////////////
	/// A negated K-linear kernel
	//////////////////////////////////////////////////////////////////
	template<IsKLinearKernel Kernel>
	struct NegatedKernel : KLinearKernel<Kernel::K, typename Kernel::WeightType, Kernel::N_SYMMETRIC, NegatedKernel<Kernel>> {
		using BASE = KLinearKernel<Kernel::K, typename Kernel::WeightType, Kernel::N_SYMMETRIC, NegatedKernel<Kernel>>;
		template<typename QR>
		using ValueArgType = BASE::template ValueArgType<QR>;
		template<typename QR>
		using GradArgType = BASE::template GradArgType<QR>;
		template<typename QR>
		using WeightArgType = BASE::template WeightArgType<QR>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;
		using WeightType = typename Kernel::WeightType;

		using BASE::BASE;
		constexpr NegatedKernel(Kernel k) : BASE(), kernel{std::move(k)} {}

		static constexpr std::array<bool,Kernel::K> NEED_VALS = Kernel::NEED_VALS;
		static constexpr std::array<bool,Kernel::K> NEED_GRAD = Kernel::NEED_GRAD;
		[[no_unique_address]] Kernel kernel{};
		
		template<typename QR>
		[[nodiscard]] static typename QR::Scalar_t CachedEvalImpl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) noexcept requires(Kernel::template HasStaticEval<QR>()){
			return -Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(Kernel::template HasStaticEval<QR>()){
			return -Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(!Kernel::template HasStaticEval<QR>()){
			return -kernel.cached_eval_impl(vals, grad, wt_ptr, qr);
		}
	};

	template<IsKLinearKernel Kernel>
	[[nodiscard]] inline constexpr NegatedKernel<Kernel> MakeNegatedKernel(Kernel k = Kernel{}) noexcept {return {std::move(k)};}

	template<IsKLinearKernel Kernel>
	[[nodiscard]] inline constexpr NegatedKernel<Kernel> operator-(Kernel k) noexcept {return {std::move(k)};}


	//////////////////////////////////////////////////////////////////
	/// A sum of two K-linear kernels.
	//////////////////////////////////////////////////////////////////
	template<IsKLinearKernel K1, IsKLinearKernel K2> requires (K1::K==K2::K && !K1::NEEDS_WEIGHT && !K2::NEEDS_WEIGHT)
	struct SummedKernel : KLinearKernel<K1::K, TupleWeight<typename K1::WeightType, typename K2::WeightType>, gutil::min(K1::N_SYMMETRIC, K2::N_SYMMETRIC), SummedKernel<K1,K2>> {
		using BASE = KLinearKernel<K1::K, TupleWeight<typename K1::WeightType, typename K2::WeightType>, gutil::min(K1::N_SYMMETRIC, K2::N_SYMMETRIC), SummedKernel<K1,K2>>;
		template<typename QR>
		using ValueArgType = BASE::template ValueArgType<QR>;
		template<typename QR>
		using GradArgType = BASE::template GradArgType<QR>;
		template<typename QR>
		using WeightArgType = BASE::template WeightArgType<QR>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;
		using WeightType = TupleWeight<typename K1::WeightType, typename K2::WeightType>;

		using BASE::BASE;
		constexpr SummedKernel(K1 k_left = K1{}, K2 k_right = K2{}) : BASE(), left{std::move(k_left)}, right{std::move(k_right)} {}

		static constexpr std::array<bool,K1::K> NEED_VALS = GV::OrArrays(K1::NEED_VALS, K2::NEED_VALS);
		static constexpr std::array<bool,K1::K> NEED_GRAD = GV::OrArrays(K1::NEED_GRAD, K2::NEED_GRAD);
		
		[[no_unique_address]] K1 left{};
		[[no_unique_address]] K2 right{};
		
		//////////////////////////////////////////////////////////////////
		/// Slice the combined weight (pointer or span) down to whatever
		/// SubKernel's own WeightArg expects, then dispatch to its own static
		/// or instance evaluation. 
		//////////////////////////////////////////////////////////////////
		template<typename SubKernel, size_t Offset, typename QR>
		[[nodiscard]] static typename QR::Scalar_t EvalSplitWeight(const SubKernel& sub, ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) noexcept {
			if constexpr (SubKernel::N_WEIGHTS == 0) {
				if constexpr (SubKernel::template HasStaticEval<QR>()) {return SubKernel::CachedEvalImpl(vals, grad, nullptr, qr);}
				else {return sub.cached_eval_impl(vals, grad, nullptr, qr);}
			}
			else if constexpr (BASE::N_WEIGHTS <= 1) {
				//this sum's own combined N_WEIGHTS is exactly 1 -- wt_ptr is already
				//the single weight pointer directly, no span involved at all
				if constexpr (SubKernel::template HasStaticEval<QR>()) {return SubKernel::CachedEvalImpl(vals, grad, wt_ptr, qr);}
				else {return sub.cached_eval_impl(vals, grad, wt_ptr, qr);}
			}
			else if constexpr (SubKernel::N_WEIGHTS == 1) {
				const auto* single = wt_ptr.template subspan<Offset,1>()[0];
				if constexpr (SubKernel::template HasStaticEval<QR>()) {return SubKernel::CachedEvalImpl(vals, grad, single, qr);}
				else {return sub.cached_eval_impl(vals, grad, single, qr);}
			}
			else {
				auto sliced = wt_ptr.template subspan<Offset, SubKernel::N_WEIGHTS>();
				if constexpr (SubKernel::template HasStaticEval<QR>()) {return SubKernel::CachedEvalImpl(vals, grad, sliced, qr);}
				else {return sub.cached_eval_impl(vals, grad, sliced, qr);}
			}
		}

		template<typename QR>
		[[nodiscard]] static typename QR::Scalar_t CachedEvalImpl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) noexcept requires(K1::HasStaticEval() && K2::HasStaticEval()) {
			// return K1::CachedEvalImpl(vals, grad, wt_ptr, qr) + K2::CachedEvalImpl(vals, grad, wt_ptr, qr);
			return EvalSplitWeight<K1,0>(K1{}, vals, grad, wt_ptr, qr) + EvalSplitWeight<K2,K1::N_WEIGHTS>(K2{}, vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(K1::HasStaticEval() && !K2::HasStaticEval()) {
			// return K1::CachedEvalImpl(vals, grad, wt_ptr, qr) + right.cached_eval_impl(vals, grad, wt_ptr, qr);
			return EvalSplitWeight<K1,0>(K1{}, vals, grad, wt_ptr, qr) + EvalSplitWeight<K2,K1::N_WEIGHTS>(right, vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(!K1::HasStaticEval() && K2::HasStaticEval()) {
			// return left.cached_eval_impl(vals, grad, wt_ptr, qr) + K2::CachedEvalImpl(vals, grad, wt_ptr, qr);
			return EvalSplitWeight<K1,0>(left, vals, grad, wt_ptr, qr) + EvalSplitWeight<K2,K1::N_WEIGHTS>(K2{}, vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(!K1::HasStaticEval() && !K2::HasStaticEval()){
			// return left.cached_eval_impl(vals, grad, wt_ptr, qr) + right.cached_eval_impl(vals, grad, wt_ptr, qr);
			return EvalSplitWeight<K1,0>(left, vals, grad, wt_ptr, qr) + EvalSplitWeight<K2,K1::N_WEIGHTS>(right, vals, grad, wt_ptr, qr);
		}
	};

	template<IsKLinearKernel K1, IsKLinearKernel K2>
	[[nodiscard]] inline constexpr SummedKernel<K1,K2> MakeSummedKernel(K1 left, K2 right) noexcept {return {std::move(left), std::move(right)};}

	template<IsKLinearKernel K1, IsKLinearKernel K2>
	[[nodiscard]] inline constexpr SummedKernel<K1,K2> operator+(K1 left, K2 right) noexcept {return {std::move(left), std::move(right)};}


	//////////////////////////////////////////////////////////////////
	/// A weighted variant of a kernel. Note that each of the final Derived classes (e.g. L2Bilinear)
	/// must be templated on NEEDS_WEIGHT and inject that into the KLinerKernel base class.
	/// This is what allows the weight to be injected here.
	//////////////////////////////////////////////////////////////////
	template<IsKLinearKernel Kernel, IsKernelWeight Weight> requires (!Kernel::NEEDS_WEIGHT)
	struct WeightedVariant : KLinearKernel<Kernel::K, Weight, Kernel::N_SYMMETRIC, WeightedVariant<Kernel,Weight>> {
		using BASE = KLinearKernel<Kernel::K, Weight, Kernel::N_SYMMETRIC, WeightedVariant<Kernel,Weight>>;
		template<typename QR>
		using ValueArgType = BASE::template ValueArgType<QR>;
		template<typename QR>
		using GradArgType = BASE::template GradArgType<QR>;
		template<typename QR>
		using WeightArgType = BASE::template WeightArgType<QR>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;
		using WeightType = Weight;

		using BASE::BASE;
		constexpr WeightedVariant(Kernel k) : BASE(), kernel{std::move(k)} {}

		static constexpr std::array<bool,Kernel::K> NEED_VALS = Kernel::NEED_VALS;
		static constexpr std::array<bool,Kernel::K> NEED_GRAD = Kernel::NEED_GRAD;
		[[no_unique_address]] Kernel kernel{};
		
		template<typename QR>
		[[nodiscard]] static typename QR::Scalar_t CachedEvalImpl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) noexcept requires(Kernel::template HasStaticEval<QR>()){
			GUTIL_ASSERT(wt_ptr);
			return Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(Kernel::template HasStaticEval<QR>()){
			GUTIL_ASSERT(wt_ptr);
			return Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QR>
		[[nodiscard]] typename QR::Scalar_t cached_eval_impl(ValueArg<QR> vals, GradArg<QR> grad, WeightArg<QR> wt_ptr, const QR& qr) const noexcept requires(!Kernel::template HasStaticEval<QR>()){
			GUTIL_ASSERT(wt_ptr);
			return kernel.cached_eval_impl(vals, grad, wt_ptr, qr);
		}
	};

	template<IsKLinearKernel Kernel, IsKernelWeight Weight> requires (!Kernel::NEEDS_WEIGHT)
	[[nodiscard]] inline constexpr WeightedVariant<Kernel,Weight> MakeWeightedVariant(Kernel k = Kernel{}) noexcept {return {std::move(k)};}
}