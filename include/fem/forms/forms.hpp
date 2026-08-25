#pragma once

#include "fem/forms/util.hpp"
#include "fem/forms/weights.hpp"

#include "fem/forms/linearforms/linearform.hpp"
#include "fem/forms/linearforms/linear_kernels.hpp"

#include "fem/forms/bilinearforms/bilinearform.hpp"
#include "fem/forms/bilinearforms/bilinear_kernels.hpp"
#include "fem/forms/bilinearforms/dense_linalg.hpp"
#include "fem/forms/bilinearforms/eigen_wrapper.hpp"



namespace GV {
	///////////////////////////////////////////////////////////////
	/// A few common aliases
	///////////////////////////////////////////////////////////////
	using H1BilinearKernel_S  = GV::H1BilinearKernel<true,false>;	//symetric, non-weighted
	using H1BilinearKernel_SW = GV::H1BilinearKernel<true,true>;	//symetric, weighted
	using L2BilinearKernel_S  = GV::L2BilinearKernel<true,false>;	//symetric, non-weighted
	using L2BilinearKernel_SW = GV::L2BilinearKernel<true,true>;	//symetric, weighted

	using L2LinearKernel_W    = GV::L2LinearKernel<true>;			//weighted


	///////////////////////////////////////////////////////////////
	/// A few helpful factories
	///////////////////////////////////////////////////////////////
	template<int N, gutil::IsReal T, typename TrialHandler_t, typename TestHandler_t, IsBilinearKernel Kernel_t, IsKernelWeight Weight_t>
	[[nodiscard]] inline constexpr auto MakeBilinearForm(const TrialHandler_t& trial, const TestHandler_t& test, Kernel_t k = Kernel_t{}, Weight_t w = Weight_t{}) noexcept {
		return GV::BilinearForm<N,T,TrialHandler_t,TestHandler_t,Kernel_t,Weight_t>{trial,test,std::move(k),std::move(w)};
	}

	template<int N, gutil::IsReal T, typename TrialHandler_t, IsBilinearKernel Kernel_t, IsKernelWeight Weight_t> requires(Kernel_t::IS_SYMMETRIC)
	[[nodiscard]] inline constexpr auto MakeBilinearForm(const TrialHandler_t& trial, Kernel_t k = Kernel_t{}, Weight_t w = Weight_t{}) noexcept {
		return GV::BilinearForm<N,T,TrialHandler_t,TrialHandler_t,Kernel_t,Weight_t>{trial,std::move(k),std::move(w)};
	}

	template<int N, gutil::IsReal T, typename TestHandler_t, IsLinearKernel Kernel_t, IsKernelWeight Weight_t>
	[[nodiscard]] inline constexpr auto MakeLinearForm(const TestHandler_t& test, Kernel_t k = Kernel_t{}, Weight_t w = Weight_t{}) noexcept {
		return GV::LinearForm<N,T,TestHandler_t,Kernel_t,Weight_t>{test,std::move(k),std::move(w)};
	}
}