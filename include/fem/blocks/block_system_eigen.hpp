#pragma once

#include "gutil.hpp"

#include "fem/blocks/block_system.hpp"
#include "util/compatibility.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

//Create a wrapper so that Eigen can use the block system for matrix-free methods
//Forward declare operator
namespace GV {
template<size_t NROWS, size_t NCOLS, typename... Blocks>
struct BlockSystemOperator;
}

//Tell Eigen to treat the operator as a sparse matrix
namespace Eigen {namespace internal {
template<size_t NROWS, size_t NCOLS, typename... Blocks>
struct traits<GV::BlockSystemOperator<NROWS,NCOLS,Blocks...>> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};	
}}

//Implement the interface between Eigen and the operator
namespace GV
{
	template<size_t NROWS, size_t NCOLS, typename... Blocks>
	struct BlockSystemOperator : public Eigen::EigenBase<BlockSystemOperator<NROWS,NCOLS,Blocks...>>
	{
		using Scalar 		= double;
		using RealScalar 	= double;
		using StorageIndex 	= int;
		enum {ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false};

		//Store a link to the block system
		const BlockSystem<NROWS,NCOLS,Blocks...>& system;

		//link via constructor
		BlockSystemOperator(const BlockSystem<NROWS,NCOLS,Blocks...>& block_system) : system(block_system) {}

		//necessary Eigen methods
		Eigen::Index rows() const {return static_cast<Eigen::Index>(system.total_rows());}
		Eigen::Index cols() const {return static_cast<Eigen::Index>(system.total_cols());}

		//template expression for multiplication
		template<typename Rhs>
		Eigen::Product<BlockSystemOperator<NROWS,NCOLS,Blocks...>, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
			return Eigen::Product<BlockSystemOperator<NROWS,NCOLS,Blocks...>, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
		}
	};
}




//add the the evaluation logic to Eigen
namespace Eigen {namespace internal {
	template<typename Rhs, size_t NROWS, size_t NCOLS, typename... Blocks>
	struct generic_product_impl<GV::BlockSystemOperator<NROWS,NCOLS,Blocks...>, Rhs, SparseShape, DenseShape, GemvProduct> 
		: generic_product_impl_base<GV::BlockSystemOperator<NROWS,NCOLS,Blocks...>, Rhs, generic_product_impl<GV::BlockSystemOperator<NROWS,NCOLS,Blocks...>, Rhs>>
	{
		using Operator = GV::BlockSystemOperator<NROWS,NCOLS,Blocks...>;
		using Scalar   = typename Product<Operator, Rhs>::Scalar;

		template<typename Dest>
		static void scaleAndAddTo(Dest& dst, const Operator& lhs, const Rhs& rhs, const Scalar& alpha) {
			gutil::LogTime timer{"BlockSystemOperator - scaleAndAddTo"};

			//this method implements dst += alpha * lhs * rhs inplace
			//we assume that alpha=1 as is the case in the iterative solvers
			eigen_assert(rhs.size() == lhs.cols() && "dimension mismatch");
			eigen_assert(dst.size() == lhs.rows() && "dimension mismatch");
			lhs.system.multiply_accumulate(GV::as_span(dst), GV::as_span(rhs), static_cast<double>(alpha));
		}
	};

}}

