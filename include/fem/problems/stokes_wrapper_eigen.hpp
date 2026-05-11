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

namespace GV
{
	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class StokesOperator;

	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class StokesPreconditioner;
}



namespace Eigen{
namespace internal {
	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	struct traits<GV::StokesPreconditioner<V_BC,P_BC,MAX_DEPTH>> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};

	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	struct traits<GV::StokesOperator<V_BC,P_BC,MAX_DEPTH>> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}
}

namespace GV
{
	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class StokesOperator : public Eigen::EigenBase<StokesOperator<V_BC,P_BC,MAX_DEPTH>>
	{
	public:

	};



	template<uint64_t V_BC, uint64_t P_BC, uint64_t MAX_DEPTH>
	class StokesPreconditioner : public Eigen::EigenBase<StokesPreconditioner<V_BC,P_BC,MAX_DEPTH>>
	{
	public:
		using Scalar 		= double;
		using RealScalar 	= double;
		using StorageIndex 	= int;
		enum {ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};

		const Stokes<V_BC,P_BC,MAX_DEPTH>& stokes; //link to problem
		Eigen::Index n_vel, n_pres, n_total;

		explicit StokesPreconditioner(const Stokes<V_BC, P_BC, MAX_DEPTH>& s) :
			stokes(s), n_vel(3*s.velocity_handler.n_dofs()), n_pres(s.pressure_handler.n_dofs()), n_total(n_vel+n_pres) {}

		StorageIndex rows() const {return static_cast<StorageIndex>(return stokes.U.size() + stokes.P.size());}
		StorageIndex cols() const {return static_cast<StorageIndex>(return stokes.U.size() + stokes.P.size());}

		template<typename Rhs>
		Eigen::Product<StokesPreconditioner, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
			return Eigen::Product<StokesPreconditioner,Rhs,Eigen::AliasFreeProduct>(*this, x.derived());
		}
	};
}
