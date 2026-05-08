#include "fem/problems/stokes.hpp"

#include <cstdint>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
class StokesPreconditioner;

template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
class StokesOperator;

using Eigen::SparseMatrix;

namespace Eigen{
namespace internal {
	template<>
	struct traits<StokesPreconditioner> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

	template<>
	struct traits<StokesOperator> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}
}

namespace GV
{
	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class StokesPreconditioner : public Eigen::EigenBase<StokesPreconditioner>
	{
	public:
		typedef double Scalar;
		typedef double RealScalar;
		typedef int StorageIndex;
		enum {ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};

		const Stokes<V_BC,P_BC,MAX_DEPTH>& stokes; //link to problem
		
	}
}
