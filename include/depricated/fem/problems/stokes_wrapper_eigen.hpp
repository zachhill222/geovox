#include "gutil.hpp"

#include "util/compatibility.hpp"

#include <span>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class Stokes;

	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class StokesOperator;

	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH, int N_STEPS=3>
	class StokesPreconditioner;
}



namespace Eigen{
namespace internal {
	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	struct traits<GV::StokesOperator<V_BC,P_BC,MAX_DEPTH>> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}
}

namespace GV
{
	//Define StokesOperator class with just a A*x method as a template expression
	//the implementation of how to evaluate A*x is delayed
	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class StokesOperator : public Eigen::EigenBase<StokesOperator<V_BC,P_BC,MAX_DEPTH>>
	{
	public:
		using Scalar 		= double;
		using RealScalar 	= double;
		using StorageIndex 	= int;
		enum {ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};

		Eigen::Index rows() const {return 3*n_vel()+n_pres();}
		Eigen::Index cols() const {return 3*n_vel()+n_pres();}

		Eigen::Index n_vel() const {return static_cast<Eigen::Index>(stokes.velocity_handler.n_dofs());}
		Eigen::Index n_pres() const {return static_cast<Eigen::Index>(stokes.pressure_handler.n_dofs());}

		//template expression for delayed vector product evaluation
		template<typename Rhs>
		Eigen::Product<StokesOperator<V_BC,P_BC,MAX_DEPTH>, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
			return Eigen::Product<StokesOperator<V_BC,P_BC,MAX_DEPTH>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
		}

		//constructor
		explicit StokesOperator(const Stokes<V_BC,P_BC,MAX_DEPTH>& problem) : stokes(problem) {}

		const Stokes<V_BC,P_BC,MAX_DEPTH>& stokes; //link to problem
	};


	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH, int N_STEPS>
	class StokesPreconditioner
	{
	public:
		using Scalar 		= double;
		using RealScalar 	= double;
		using StorageIndex 	= int;

		Stokes<V_BC,P_BC,MAX_DEPTH> const* stokes;

		StokesPreconditioner() : stokes(nullptr) {}

		//Eigen preconditioner interface
		template<typename MatType>
		StokesPreconditioner& analyzePattern(const MatType&) {return *this;}
		template<typename MatType>
		StokesPreconditioner& factorize(const MatType&) {return *this;}
		template<typename MatType>
		StokesPreconditioner& compute(const MatType&) {return *this;}
		
		StokesPreconditioner& compute(const StokesOperator<V_BC,P_BC,MAX_DEPTH>& op) {
			stokes = &op.stokes;
			return *this;
		}

		Eigen::ComputationInfo info() const {return Eigen::Success;}

		template<typename Rhs>
		Eigen::VectorXd solve(const Rhs& b) const {
			eigen_assert(stokes!=nullptr && "StokesPreconditioner - operator not set");

			Eigen::VectorXd result(b.size()); //zeroed in jacobi_preconditioner
			Eigen::VectorXd guess = Eigen::VectorXd::Zero(b.size());
			stokes->jacobi_precondition(as_span(result), as_span(guess), as_span(b), N_STEPS);

			return result;
		}

		template<typename Rhs, typename Guess>
		Eigen::VectorXd solveWithGuess(const Rhs& b, Guess& x0) const {
			eigen_assert(stokes!=nullptr && "StokesPreconditioner - operator not set");

			Eigen::VectorXd result(b.size()); //zeroed in jacobi_preconditioner
			stokes->jacobi_precondition(as_span(result), as_span(x0), as_span(b), N_STEPS);
			
			return result;
		}
	};
}

//Implementation of how to evaluate the stokes matrix product
namespace Eigen {
namespace internal {

	template<typename Rhs, uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	struct generic_product_impl<GV::StokesOperator<V_BC,P_BC,MAX_DEPTH>, Rhs, SparseShape, DenseShape, GemvProduct> 
		: generic_product_impl_base<GV::StokesOperator<V_BC,P_BC,MAX_DEPTH>, Rhs, generic_product_impl<GV::StokesOperator<V_BC,P_BC,MAX_DEPTH>, Rhs>>
	{
		using Operator = GV::StokesOperator<V_BC,P_BC,MAX_DEPTH>;
		using Scalar = typename Product<Operator, Rhs>::Scalar;

		template<typename Dest>
		static void scaleAndAddTo(Dest& dst, const Operator& lhs, const Rhs& rhs, const Scalar& alpha) {
			gutil::LogTime timer{"StokesOperator - scaleAndAddTo"};

			//this method implements dst += alpha * lhs * rhs inplace
			//we assume that alpha=1 as is the case in the iterative solvers
			eigen_assert(alpha==Scalar(1) && "scaling is not implemented");
			EIGEN_ONLY_USED_FOR_DEBUG(alpha);

			const auto N1 = 3*lhs.n_vel();
			const auto N2 = lhs.n_pres();
			eigen_assert(rhs.size() == N1+N2 && "dimension mismatch");

			std::span<const Scalar> U   = GV::as_span<Scalar>(rhs, 0,  N1);
			std::span<const Scalar> P   = GV::as_span<Scalar>(rhs, N1, N2);
			std::span<Scalar> dst_upper = GV::as_span<Scalar>(dst, 0,  N1);
			std::span<Scalar> dst_lower = GV::as_span<Scalar>(dst, N1, N2);

			//increment upper portion
			lhs.stokes.K_U(dst_upper, U);
			lhs.stokes.G_P(dst_upper, P);

			//increment lower portion (P-P block is all zeros)
			lhs.stokes.GT_U(dst_lower, U);

			//apply boundary conditions
			lhs.stokes.apply_velocity_bc(dst_upper);
			lhs.stokes.apply_pressure_bc(dst_lower);
		}
	};
}}
