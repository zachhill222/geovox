#pragma once

#include "fem/forms/bilinear_form.hpp"
#include "fem/forms/bilinear_form_compute.hpp"
#include "fem/forms/bilinear_form_scatter.hpp"
#include "fem/numerics/quad_point_map.hpp"

namespace GV
{
	//a class to assemble the global "stiffness" matrix
	template<typename TestHandler_type, typename TrialHandler_type, bool IS_SYMMETRIC, typename EvalPolicy>
	struct BilinearFormAssembler : public BilinearForm<TestHandler_type,TrialHandler_type,IS_SYMMETRIC,EvalPolicy>
	{
		using BASE       = BilinearForm<TestHandler_type,TrialHandler_type,IS_SYMMETRIC,EvalPolicy>;
		using TestDOF_t  = typename BASE::TestDOF_t;
		using TrialDOF_t = typename BASE::TrialDOF_t;
		using QuadElem_t = typename BASE::QuadElem_t;

		using ComputePolicy_t = BilinearFormComputeLocalMat<TestDOF_t,TrialDOF_t,IS_SYMMETRIC,EvalPolicy>;
		using ScatterPolicy_t = BilinearFormScatterLocalMat<TestDOF_t,TrialDOF_t,ComputePolicy_t>;
		
		ComputePolicy_t compute_policy;
		ScatterPolicy_t scatter_policy;

		//constructor
		BilinearFormAssembler(const TestHandler_type& TestH, const TrialHandler_type& TrialH) : 
			BASE(TestH, TrialH), scatter_policy{compute_policy} {}

		//link to global storage
		template<typename COO>
		inline void set_global(COO& coo) {scatter_policy.set_global(coo);}

		//consistent api
		inline void prepare() {}
		inline void finalize() {}

		template<uint64_t N_QUAD_POINTS>
		inline void compute(
				const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map,
				const double Jxx, const double Jyy, const double Jzz) {
			compute_policy.compute(this->test_dofs, this->trial_dofs, q_map, Jxx, Jyy, Jzz);
		}

		inline void scatter() const {
			scatter_policy.scatter(this->test_dofs, this->trial_dofs);
		}
	};

}