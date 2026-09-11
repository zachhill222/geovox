#pragma once


#include "gutil.hpp"
#include "fem/forms/util.hpp"

namespace GV {
	template<typename Kernel>
	concept IsKLinearKernel = requires (Kernel k) {
		{Kernel::K} -> std::convertible_to<size_t>;
		{Kernel::NEEDS_WEIGHT} -> std::convertible_to<bool>;
		{Kernel::N_SYMMETRIC}  -> std::convertible_to<size_t>;
		{Kernel::IS_SYMMETRIC} -> std::convertible_to<bool>;
		{Kernel::NEED_VALS}    -> std::convertible_to<std::array<bool,Kernel::K>>;
		{Kernel::NEED_GRAD}    -> std::convertible_to<std::array<bool,Kernel::K>>;
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
	template<size_t K_, bool IsWeighted, size_t NSymmetric_, typename Derived> requires (NSymmetric_ <= K_)
	struct KLinearKernel {
		//////////////////////////////////////////////////////////////////
		/// Track essential constants.
		//////////////////////////////////////////////////////////////////
		static constexpr size_t K            = K_;
		static constexpr bool   NEEDS_WEIGHT = IsWeighted;
		static constexpr size_t N_SYMMETRIC  = NSymmetric_;
		static constexpr bool   IS_SYMMETRIC = N_SYMMETRIC==K;
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
		/// pass arrays of pointers to the values, pass nullptr if that value/gradient is not needed
		//////////////////////////////////////////////////////////////////
		template<typename QuadRule>
		using ValueArg = std::array<const DofValueCache<QuadRule>*, K>;

		template<typename QuadRule>
		using GradArg = std::array<const DofGradCache<QuadRule>*, K>;

		template<typename QuadRule>
		using WeightArg = ScalarValueCache<QuadRule>;

		template<typename QuadRule>
		[[nodiscard]] static constexpr bool IsValArgValid(const ValueArg<QuadRule>& arg) noexcept {
			bool flag = true;
			for (size_t k=0; k<K; ++k) {
				if (Derived::NEED_VALS[k] && arg[k]==nullptr) {
					GUTIL_ERROR("Value argument ", k, " is nullptr but is required");
					flag = false;
				}
			}
			return flag;
		}

		template<typename QuadRule>
		[[nodiscard]] static constexpr bool IsGradArgValid(const GradArg<QuadRule>& arg) noexcept {
			bool flag = true;
			for (size_t k=0; k<K; ++k) {
				if (Derived::NEED_GRAD[k] && arg[k]==nullptr) {
					GUTIL_ERROR("Grad argument ", k, " is nullptr but is required");
					flag = false;
				}
			}
			return flag;
		}


		//////////////////////////////////////////////////////////////////
		/// Check if the kernel can be evaluated statically. If it can,
		/// use default to static evaluation.
		//////////////////////////////////////////////////////////////////
		template<typename QuadRule>
		[[nodiscard]] static constexpr bool HasStaticEval() noexcept requires( requires {
			Derived::template CachedEvalImpl<QuadRule>(std::declval<const ValueArg<QuadRule>&>(), std::declval<const GradArg<QuadRule>&>(), std::declval<const WeightArg<QuadRule>*>(), std::declval<const QuadRule&>());
		}) {return true;}
		template<typename QuadRule>
		[[nodiscard]] static constexpr bool HasStaticEval() noexcept requires( !requires {
			Derived::template CachedEvalImpl<QuadRule>(std::declval<const ValueArg<QuadRule>&>(), std::declval<const GradArg<QuadRule>&>(), std::declval<const WeightArg<QuadRule>*>(), std::declval<const QuadRule&>());
		}) {return false;}


		//////////////////////////////////////////////////////////////////
		/// Convert to Derived at runtime if needed.
		//////////////////////////////////////////////////////////////////
		[[nodiscard]] const Derived* derived() const noexcept {return static_cast<const Derived*>(this);}
		[[nodiscard]] Derived* derived() noexcept {return static_cast<Derived*>(this);}


		//////////////////////////////////////////////////////////////////
		/// Forward evaluation to the Derived class for static evaluation.
		//////////////////////////////////////////////////////////////////
		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEval(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept requires(HasStaticEval()) {
			static_assert(IsValid());
			GUTIL_ASSERT(IsValArgValid(vals));
			GUTIL_ASSERT(IsGradArgValid(grad));
			if constexpr (NEEDS_WEIGHT) {GUTIL_ASSERT(wt_ptr);}
			return Derived::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}


		//////////////////////////////////////////////////////////////////
		/// Forward evaluation to the Derived class for runtime evaluation.
		//////////////////////////////////////////////////////////////////
		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires (HasStaticEval()) {	
			static_assert(IsValid());
			GUTIL_ASSERT(IsValArgValid(vals));
			GUTIL_ASSERT(IsGradArgValid(grad));
			if constexpr (NEEDS_WEIGHT) {GUTIL_ASSERT(wt_ptr);}
			return Derived::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}
		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires (!HasStaticEval()) {	
			static_assert(IsValid());
			GUTIL_ASSERT(IsValArgValid(vals));
			GUTIL_ASSERT(IsGradArgValid(grad));
			if constexpr (NEEDS_WEIGHT) {GUTIL_ASSERT(wt_ptr);}
			return derived()->cached_eval_impl(vals, grad, wt_ptr, qr);
		}
	};


	//////////////////////////////////////////////////////////////////
	/// Make a few helper functions for adding and scaling kernels.
	//////////////////////////////////////////////////////////////////
	template<size_t K>
	struct ZeroKernel : public KLinearKernel<K, false, K, ZeroKernel<K>> {
		using BASE = KLinearKernel<K,false,K,ZeroKernel<K>>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;

		using BASE::BASE;

		static constexpr std::array<bool,K> NEED_VALS = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();
		static constexpr std::array<bool,K> NEED_GRAD = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();

		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEvalImpl(const ValueArg<QuadRule>&, const GradArg<QuadRule>&, const WeightArg<QuadRule>*, const QuadRule&) noexcept {
			return typename QuadRule::Scalar_t{0};
		}
	};

	template<size_t K, bool IsWeighted>
	struct IdentityKernel : public KLinearKernel<K, IsWeighted, K, IdentityKernel<K,IsWeighted>> {
		using BASE = KLinearKernel<K,IsWeighted,K,IdentityKernel<K,IsWeighted>>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;

		using BASE::BASE;

		static constexpr std::array<bool,K> NEED_VALS = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();
		static constexpr std::array<bool,K> NEED_GRAD = [](){std::array<bool,K> ar{}; ar.fill(false); return ar;}();

		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEvalImpl(const ValueArg<QuadRule>&, const GradArg<QuadRule>&, const WeightArg<QuadRule>*, const QuadRule& qr) noexcept requires(!IsWeighted) {
			return QuadRule::quad_w_sum() * qr.jac_det();
		}

		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEvalImpl(const ValueArg<QuadRule>&, const GradArg<QuadRule>&, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept requires(IsWeighted) {
			GUTIL_ASSERT(wt_ptr);
			constexpr int N = QuadRule::TOTAL_QUAD_POINTS;
			typename QuadRule::Scalar_t val{0};
			static constexpr auto qw = QuadRule::quad_w();
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
	struct ScaledKernel : KLinearKernel<Kernel::K, Kernel::NEEDS_WEIGHT, Kernel::N_SYMMETRIC, ScaledKernel<T,Kernel>> {
		using BASE = KLinearKernel<Kernel::K, Kernel::NEEDS_WEIGHT, Kernel::N_SYMMETRIC, ScaledKernel<T,Kernel>>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;

		using BASE::BASE;
		constexpr ScaledKernel(T s, Kernel k = Kernel{}) : BASE(), scale{s}, kernel{std::move(k)} {}

		static constexpr std::array<bool,Kernel::K> NEED_VALS = Kernel::NEED_VALS;
		static constexpr std::array<bool,Kernel::K> NEED_GRAD = Kernel::NEED_GRAD;
		T scale{1};
		[[no_unique_address]] Kernel kernel{};

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(Kernel::HasStaticEval()){
			return scale * Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(!Kernel::HasStaticEval()){
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
	struct NegatedKernel : KLinearKernel<Kernel::K, Kernel::NEEDS_WEIGHT, Kernel::N_SYMMETRIC, NegatedKernel<Kernel>> {
		using BASE = KLinearKernel<Kernel::K, Kernel::NEEDS_WEIGHT, Kernel::N_SYMMETRIC, NegatedKernel<Kernel>>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;

		using BASE::BASE;
		constexpr NegatedKernel(Kernel k) : BASE(), kernel{std::move(k)} {}

		static constexpr std::array<bool,Kernel::K> NEED_VALS = Kernel::NEED_VALS;
		static constexpr std::array<bool,Kernel::K> NEED_GRAD = Kernel::NEED_GRAD;
		[[no_unique_address]] Kernel kernel{};
		
		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEvalImpl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept requires(Kernel::HasStaticEval()){
			return -Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(Kernel::HasStaticEval()){
			return -Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(!Kernel::HasStaticEval()){
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
	struct SummedKernel : KLinearKernel<K1::K, false, gutil::min(K1::N_SYMMETRIC, K2::N_SYMMETRIC), SummedKernel<K1,K2>> {
		using BASE = KLinearKernel<K1::K, false, gutil::min(K1::N_SYMMETRIC, K2::N_SYMMETRIC), SummedKernel<K1,K2>>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;

		using BASE::BASE;
		constexpr SummedKernel(K1 k_left = K1{}, K2 k_right = K2{}) : BASE(), left{std::move(k_left)}, right{std::move(k_right)} {}

		static constexpr std::array<bool,K1::K> NEED_VALS = GV::OrArrays(K1::NEED_VALS, K2::NEED_VALS);
		static constexpr std::array<bool,K1::K> NEED_GRAD = GV::OrArrays(K1::NEED_GRAD, K2::NEED_GRAD);
		
		[[no_unique_address]] K1 left{};
		[[no_unique_address]] K2 right{};
		
		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEvalImpl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept requires(K1::HasStaticEval() && K2::HasStaticEval()) {
			return K1::CachedEvalImpl(vals, grad, wt_ptr, qr) + K2::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(K1::HasStaticEval() && !K2::HasStaticEval()) {
			return K1::CachedEvalImpl(vals, grad, wt_ptr, qr) + right.cached_eval_impl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(!K1::HasStaticEval() && K2::HasStaticEval()) {
			return left.cached_eval_impl(vals, grad, wt_ptr, qr) + K2::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(!K1::HasStaticEval() && !K2::HasStaticEval()){
			return left.cached_eval_impl(vals, grad, wt_ptr, qr) + right.cached_eval_impl(vals, grad, wt_ptr, qr);
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
	template<IsKLinearKernel Kernel> requires (!Kernel::NEEDS_WEIGHT)
	struct WeightedVariant : KLinearKernel<Kernel::K, true, Kernel::N_SYMMETRIC, WeightedVariant<Kernel>> {
		using BASE = KLinearKernel<Kernel::K, true, Kernel::N_SYMMETRIC, WeightedVariant<Kernel>>;
		template<typename QR>
		using ValueArg = BASE::template ValueArg<QR>;
		template<typename QR>
		using GradArg = BASE::template GradArg<QR>;
		template<typename QR>
		using WeightArg = BASE::template WeightArg<QR>;

		using BASE::BASE;
		constexpr WeightedVariant(Kernel k) : BASE(), kernel{std::move(k)} {}

		static constexpr std::array<bool,Kernel::K> NEED_VALS = Kernel::NEED_VALS;
		static constexpr std::array<bool,Kernel::K> NEED_GRAD = Kernel::NEED_GRAD;
		[[no_unique_address]] Kernel kernel{};
		
		template<typename QuadRule>
		[[nodiscard]] static typename QuadRule::Scalar_t CachedEvalImpl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) noexcept requires(Kernel::HasStaticEval()){
			GUTIL_ASSERT(wt_ptr);
			return Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(Kernel::HasStaticEval()){
			GUTIL_ASSERT(wt_ptr);
			return Kernel::CachedEvalImpl(vals, grad, wt_ptr, qr);
		}

		template<typename QuadRule>
		[[nodiscard]] typename QuadRule::Scalar_t cached_eval_impl(const ValueArg<QuadRule>& vals, const GradArg<QuadRule>& grad, const WeightArg<QuadRule>* wt_ptr, const QuadRule& qr) const noexcept requires(!Kernel::HasStaticEval()){
			GUTIL_ASSERT(wt_ptr);
			return kernel.cached_eval_impl(vals, grad, wt_ptr, qr);
		}
	};

	template<IsKLinearKernel Kernel> requires (!Kernel::NEEDS_WEIGHT)
	[[nodiscard]] inline constexpr WeightedVariant<Kernel> MakeWeightedVariant(Kernel k = Kernel{}) noexcept {return {std::move(k)};}
}