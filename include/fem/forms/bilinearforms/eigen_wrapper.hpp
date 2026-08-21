#pragma once
#ifdef EIGEN_MAJOR_VERSION
#include "gutil.hpp"
#include "util/util.hpp"

/////////////////////////////////////////////////////////////////////
/// Make a wrapper around a bilinear form so Eigen can use iterative
/// methods without constructing the sparse matrix. Additionally,
/// Allow the addition of such forms.
/////////////////////////////////////////////////////////////////////
namespace GV {

	/// A concept to keep any custom operators from being too broad
	template<typename T>
	concept IsEigenBaseOperator = std::derived_from<T, Eigen::EigenBase<T>>
		&& !std::derived_from<T, Eigen::MatrixBase<T>>
		&& !std::derived_from<T, Eigen::SparseMatrixBase<T>>;


	/////////////////////////////////////////////////////////////////
	/// Forward declare the GV operator types (the wrappers that will be used)
	/////////////////////////////////////////////////////////////////
	template<typename BilinearForm_t>
	struct BilinearFormOperator;

	template<typename ContainerType> requires(gutil::IsReal<typename ContainerType::value_type>)
	struct DiagonalOperator;

	template<typename ContainerType> requires(gutil::IsReal<typename ContainerType::value_type>)
	struct DiagonalPreconditioner;	//the inverse matrix of a diagonal matrix

	template<IsEigenBaseOperator Op, int K=1> requires (Op::IS_SYMMETRIC)
	struct JacobiPreconditioner;	//implements K jacobi method iterations

	template<IsEigenBaseOperator Op1, IsEigenBaseOperator Op2>
	struct SumOperator;
}

namespace Eigen {
namespace internal {


	/////////////////////////////////////////////////////////////////
	/// Inject trait information into Eigen
	/////////////////////////////////////////////////////////////////
	template<typename BilinearForm_t>
	struct traits<GV::BilinearFormOperator<BilinearForm_t>> : public Eigen::internal::traits<Eigen::SparseMatrix<typename BilinearForm_t::Scalar_t>> {};

	template<typename ContainerType> requires(gutil::IsReal<typename ContainerType::value_type>)
	struct traits<GV::DiagonalOperator<ContainerType>> : public Eigen::internal::traits<Eigen::SparseMatrix<typename ContainerType::value_type>> {};

	template<typename ContainerType> requires(gutil::IsReal<typename ContainerType::value_type>)
	struct traits<GV::DiagonalPreconditioner<ContainerType>> : public Eigen::internal::traits<Eigen::SparseMatrix<typename ContainerType::value_type>> {};

	template<GV::IsEigenBaseOperator Op, int K> requires (Op::IS_SYMMETRIC)
	struct traits<GV::JacobiPreconditioner<Op,K>> : public Eigen::internal::traits<Eigen::SparseMatrix<typename Op::Scalar>> {};

	template<GV::IsEigenBaseOperator Op1, GV::IsEigenBaseOperator Op2>
	struct traits<GV::SumOperator<Op1,Op2>> : public Eigen::internal::traits<Eigen::SparseMatrix<typename Op1::Scalar>> {};
}}

namespace GV {


	/////////////////////////////////////////////////////////////////
	/// Define the GV operators with a delayed evaluation for Eigen to use.
	/////////////////////////////////////////////////////////////////
	template<typename BilinearForm_t>
	struct BilinearFormOperator : public Eigen::EigenBase<BilinearFormOperator<BilinearForm_t>> {
		using Scalar = typename BilinearForm_t::Scalar_t;
		using RealScalar = Scalar;
		using StorageIndex = int;
		enum {ColsAtCompileTime = Eigen::Dynamic,  MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};
		
		static constexpr bool IS_SYMMETRIC = BilinearForm_t::IS_SYMMETRIC;

		Eigen::Index rows() const {return static_cast<Eigen::Index>(form.n_rows());}
		Eigen::Index cols() const {return static_cast<Eigen::Index>(form.n_cols());}

		template<typename Rhs>
		Eigen::Product<BilinearFormOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
			return Eigen::Product<BilinearFormOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
		}

		explicit BilinearFormOperator(const BilinearForm_t& f) : form(f) {}
		const BilinearForm_t& form;

		//note that construct_diagonal accumulates into diag
		void construct_diagonal(std::span<Scalar> diag) const noexcept requires(IS_SYMMETRIC) {
			if (form.mesh.is_color_sorted()) {
				form.construct_diagonal_colored(diag);
			}
			else {
				form.construct_diagonal(diag);
			}
		}

		void mat_vec_multiply_accumulate_jacobi(std::span<Scalar> Y, std::span<const Scalar> X) const noexcept requires(IS_SYMMETRIC) {
			//compute Y -= (L+U)*X
			if (form.mesh.is_color_sorted()) {
				form.mat_vec_multiply_accumulate_jacobi_colored(Y, X, Scalar{-1});
			}
			else {
				form.mat_vec_multiply_accumulate_jacobi(Y, X, Scalar{-1});
			}
		}
	};


	template<typename ContainerType> requires(gutil::IsReal<typename ContainerType::value_type>)
	struct DiagonalOperator : public Eigen::EigenBase<DiagonalOperator<ContainerType>> {
		using Scalar = typename ContainerType::value_type;
		using RealScalar = Scalar;
		using StorageIndex = int;
		enum {ColsAtCompileTime = Eigen::Dynamic,  MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};
		
		static constexpr bool IS_SYMMETRIC = true;
		static constexpr Scalar inv_cutoff = 100*gutil::Min<Scalar>::value;

		Eigen::Index rows() const {return static_cast<Eigen::Index>(diag.size());}
		Eigen::Index cols() const {return static_cast<Eigen::Index>(diag.size());}

		template<typename Rhs>
		Eigen::Product<DiagonalOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
			return Eigen::Product<DiagonalOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
		}

		explicit DiagonalOperator(ContainerType d) : diag(std::move(d)) {}
		ContainerType diag;

		DiagonalOperator() noexcept = default;
		void construct_diagonal(std::span<Scalar> other_diag) const noexcept {
			//increment to be consistent with other types
			GUTIL_ASSERT(other_diag.size()==diag.size());
			GUTIL_OMP(parallel)
			{
				OmpIndexRange range(diag.size());
				GUTIL_SIMD()
				for (auto i=range.begin; i<range.end; ++i) {
					other_diag[i] += diag[i];
				}
			}
		}

		constexpr Scalar coeff(Eigen::Index i, Eigen::Index j) const noexcept {
			GUTIL_ASSERT(0<=i && 0<=j);
			GUTIL_ASSERT(i<diag.size() && j<diag.size());
			return i==j ? diag[i] : Scalar{0};
		}

		constexpr Scalar& coeff(Eigen::Index i, Eigen::Index j) noexcept {
			GUTIL_ASSERT(i==j);
			GUTIL_ASSERT(0<=i);
			GUTIL_ASSERT(i<diag.size());
			return diag[i];
		}

		constexpr void mat_vec_multiply_accumulate_jacobi(std::span<Scalar>, std::span<const Scalar>) const noexcept {
			return;
		}

		template<typename OutType=Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, typename Rhs>
		[[nodiscard]] OutType solve(const Rhs& b) const noexcept {
			OutType result(diag.size());
			GUTIL_OMP(parallel)
			{
				GV::OmpIndexRange range(diag.size());
				GUTIL_SIMD()
				for (auto idx=range.begin; idx<range.end; ++idx) {
					GUTIL_ASSERT(diag[idx] > inv_cutoff);
					result[idx] = b[idx]/diag[idx];
				}
			}
			return result;
		}
	};

	template<typename ContainerType> requires(gutil::IsReal<typename ContainerType::value_type>)
	struct DiagonalPreconditioner : public DiagonalOperator<ContainerType> {
		using BASE = DiagonalOperator<ContainerType>;
		using Scalar = typename ContainerType::value_type;
		using RealScalar = Scalar;
		using StorageIndex = int;
		enum {ColsAtCompileTime = Eigen::Dynamic,  MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};
		
		static constexpr bool IS_SYMMETRIC = true;
		static constexpr Scalar inv_cutoff = 100*gutil::Min<Scalar>::value;

		Eigen::Index rows() const {return static_cast<Eigen::Index>(diag.size());}
		Eigen::Index cols() const {return static_cast<Eigen::Index>(diag.size());}

		DiagonalPreconditioner() noexcept = default;
		explicit DiagonalPreconditioner(ContainerType d) : BASE(std::move(d)) {
			Reciprocate(GV::as_span(diag));
		}
		using BASE::diag;	//reciprocals of the matrix this is a preconditioner for

		///Convenient interface
		static constexpr void Reciprocate(std::span<Scalar> r) noexcept {
			GUTIL_OMP(parallel)
			{
				OmpIndexRange range(r.size());
				GUTIL_SIMD()
				for (auto i=range.begin; i<range.end; ++i) {
					GUTIL_ASSERT(gutil::abs(r[i]) > inv_cutoff);
					r[i] = Scalar{1}/r[i];
				}
			}
		}

		template<typename MatType>
		DiagonalPreconditioner& analyzePattern(const MatType&) noexcept {return *this;}

		Eigen::ComputationInfo info() const {return Eigen::Success;}

		template<typename MatType>
		DiagonalPreconditioner& factorize(const MatType& mat) noexcept {
			GUTIL_ASSERT(mat.rows()==mat.cols());
			diag.resize(mat.rows());
			
			GUTIL_OMP(parallel for)
			for (Eigen::Index i=0; i<mat.rows(); ++i) {
				diag[i] = mat.coeff(i,i);
			}

			Reciprocate(GV::as_span(diag));

			return *this;
		}

		template<GV::IsEigenBaseOperator Op> requires (Op::IS_SYMMETRIC)
		DiagonalPreconditioner& factorize(const Op& op) noexcept {
			diag.assign(op.rows(), Scalar{0});
			op.construct_diagonal(GV::as_span(diag));
			Reciprocate(GV::as_span(diag));
			return *this;
		}

		template<typename OpType>
		DiagonalPreconditioner& compute(const OpType& mat) noexcept {return factorize(mat);}

		template<typename OutType = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>, typename Rhs>
		[[nodiscard]] OutType solve(const Rhs& b) const noexcept {
			OutType result(diag.size());

			GUTIL_OMP(parallel)
			{
				GV::OmpIndexRange range(diag.size());
				GUTIL_SIMD()
				for (size_t i=range.begin; i<range.end; ++i) {
					result[i] = diag[i]*b[i];
				}
			}
			return result;
		}

		template<typename Rhs, typename Guess>
		[[nodiscard]] Guess solveWithGuess(const Rhs& b, const Guess& x0) const noexcept {
			return solve<Guess>(b);
		}
	};


	template<GV::IsEigenBaseOperator Op, int K> requires(Op::IS_SYMMETRIC)
	struct JacobiPreconditioner : public Eigen::EigenBase<JacobiPreconditioner<Op,K>> {
		using Scalar = typename Op::Scalar;
		using RealScalar = Scalar;
		using StorageIndex = int;
		enum {ColsAtCompileTime = Eigen::Dynamic,  MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};
		
		using EigenVec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

		static constexpr bool IS_SYMMETRIC = Op::IS_SYMMETRIC;

		Eigen::Index rows() const {return op ? op->rows() : 0;}
		Eigen::Index cols() const {return op ? op->cols() : 0;}

		JacobiPreconditioner() noexcept = default;
		
		const Op* op{nullptr};
		DiagonalPreconditioner<std::vector<Scalar>> inv_diag{};

		template<typename MatType>
		JacobiPreconditioner& analyzePattern(const MatType&) noexcept {return *this;}

		Eigen::ComputationInfo info() const {return Eigen::Success;}

		JacobiPreconditioner& factorize(const Op& new_op) noexcept {
			op = &new_op;
			inv_diag.factorize(*op);
			return *this;
		}

		JacobiPreconditioner& compute(const Op& op) noexcept {return factorize(op);}

		template<typename Rhs>
		[[nodiscard]] EigenVec solve(const Rhs& b) const noexcept {
			EigenVec x0 = EigenVec::Zero(rows());
			return solveWithGuess(b, std::move(x0));
		}

		template<typename Rhs>
		[[nodiscard]] EigenVec solveWithGuess(const Rhs& b, EigenVec x0) const noexcept {
			GUTIL_ASSERT(op);

			for (int k=0; k<K; ++k) {
				EigenVec delta(b);
				//computes delta -= (L+U)*x0
				op->mat_vec_multiply_accumulate_jacobi(GV::as_span(delta), GV::as_span(x0));
				x0 = inv_diag.solve(delta);
			}
			return x0;
		}
	};


	template<IsEigenBaseOperator Op1, IsEigenBaseOperator Op2>
	struct SumOperator : public Eigen::EigenBase<SumOperator<Op1,Op2>> {
		using Scalar = typename Op1::Scalar;
		using RealScalar = Scalar;
		using StorageIndex = int;
		enum {ColsAtCompileTime = Eigen::Dynamic,  MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};
		
		static_assert(std::same_as<Scalar, typename Op2::Scalar>);
		static constexpr bool IS_SYMMETRIC = Op1::IS_SYMMETRIC && Op2::IS_SYMMETRIC;

		Eigen::Index rows() const {return static_cast<Eigen::Index>(left.rows());}
		Eigen::Index cols() const {return static_cast<Eigen::Index>(left.cols());}

		template<typename Rhs>
		Eigen::Product<SumOperator, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
			return Eigen::Product<SumOperator, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
		}

		explicit SumOperator(const Op1& l, const Op2& r) : left(l), right(r) {}
		const Op1& left;
		const Op2& right;

		void construct_diagonal(std::span<Scalar> diag) const noexcept requires(IS_SYMMETRIC) {
			//note that construct_diagonal accumulates into diag
			left.construct_diagonal(diag);
			right.construct_diagonal(diag);
		}

		void mat_vec_multiply_accumulate_jacobi(std::span<Scalar> Y, std::span<const Scalar> X) const noexcept requires (IS_SYMMETRIC) {
			//note that Y -= (L_l+U_l)*X followed by Y -= (L_r+U_r)*X is the same as
			// Y = Y - (L_l+U_l)*X - (L_r+U_r)*X
			left.mat_vec_multiply_accumulate_jacobi(Y,X);
			right.mat_vec_multiply_accumulate_jacobi(Y,X);
		}
	};

	template<IsEigenBaseOperator Op1, IsEigenBaseOperator Op2>
	SumOperator<Op1,Op2> operator+(const Op1& l, const Op2& r) {
		return SumOperator<Op1,Op2>(l, r);
	}
}

namespace Eigen {
namespace internal {


	/////////////////////////////////////////////////////////////////
	/// Tell Eigen what to call when evaluating the products.
	/////////////////////////////////////////////////////////////////
	template<typename Rhs, typename BilinearForm_t>
	struct generic_product_impl<GV::BilinearFormOperator<BilinearForm_t>, Rhs, SparseShape, DenseShape, GemvProduct>
		: generic_product_impl_base<GV::BilinearFormOperator<BilinearForm_t>, Rhs, generic_product_impl<GV::BilinearFormOperator<BilinearForm_t>, Rhs>>
	{
		using Operator = GV::BilinearFormOperator<BilinearForm_t>;
		using Scalar = typename Product<Operator, Rhs>::Scalar;

		template<typename Dest>
		static void scaleAndAddTo(Dest& dest, const Operator& lhs, const Rhs& rhs, const Scalar& alpha) {
			if (lhs.form.mesh.is_color_sorted()) {
				lhs.form.mat_vec_multiply_accumulate_colored(GV::as_span(dest), GV::as_span(rhs), alpha);
			}
			else {
				lhs.form.mat_vec_multiply_accumulate(GV::as_span(dest), GV::as_span(rhs), alpha);
			}
		}
	};


	template<typename Rhs, typename ContainerType> requires(gutil::IsReal<typename ContainerType::value_type>)
	struct generic_product_impl<GV::DiagonalOperator<ContainerType>, Rhs, SparseShape, DenseShape, GemvProduct>
		: generic_product_impl_base<GV::DiagonalOperator<ContainerType>, Rhs, generic_product_impl<GV::DiagonalOperator<ContainerType>, Rhs>>
	{
		using Operator = GV::DiagonalOperator<ContainerType>;
		using Scalar = typename Product<Operator, Rhs>::Scalar;

		template<typename Dest>
		static void scaleAndAddTo(Dest& dest, const Operator& lhs, const Rhs& rhs, const Scalar& alpha) {
			GUTIL_OMP(parallel)
			{
				GV::OmpIndexRange range(lhs.diag.size());
				GUTIL_SIMD()
				for (size_t i=range.begin; i<range.end; ++i) {
					dest[i] += alpha*lhs.diag[i]*rhs[i];
				}
			}
		}
	};


	template<typename Rhs, GV::IsEigenBaseOperator Op1, GV::IsEigenBaseOperator Op2>
	struct generic_product_impl<GV::SumOperator<Op1,Op2>, Rhs, SparseShape, DenseShape, GemvProduct>
		: generic_product_impl_base<GV::SumOperator<Op1,Op2>, Rhs, generic_product_impl<GV::SumOperator<Op1,Op2>, Rhs>>
	{
		using Operator = GV::SumOperator<Op1,Op2>;
		using Scalar = typename Product<Operator, Rhs>::Scalar;

		template<typename Dest>
		static void scaleAndAddTo(Dest& dest, const Operator& lhs, const Rhs& rhs, const Scalar& alpha) {
			GUTIL_ASSERT(lhs.left.rows()==lhs.right.rows());
			GUTIL_ASSERT(lhs.left.cols()==lhs.right.cols());

			dest.noalias() += alpha * (lhs.left * rhs);
			dest.noalias() += alpha * (lhs.right * rhs);
		}
	};
}}


#endif