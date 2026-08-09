#pragma once

#include "fem/forms/bilinear/base.hpp"
#include "fem/forms/bilinear/policy_compute.hpp"
#include "fem/forms/bilinear/policy_scatter.hpp"

#include "fem/numerics/quad_point_map.hpp"

namespace GV
{
	//a class to assemble the global "stiffness" matrix
	template<typename TestHandler_type, typename TrialHandler_type, typename EvalPolicy>
	struct BilinearFormAssembler : public BilinearForm<TestHandler_type,TrialHandler_type,EvalPolicy>
	{
		using BASE       = BilinearForm<TestHandler_type,TrialHandler_type,EvalPolicy>;
		using TestDOF_t  = typename BASE::TestDOF_t;
		using TrialDOF_t = typename BASE::TrialDOF_t;
		using QuadElem_t = typename BASE::QuadElem_t;

		using ComputePolicy_t = BilinearFormComputeLocalMat<TestDOF_t,TrialDOF_t,EvalPolicy>;
		using ScatterPolicy_t = BilinearFormScatterToGlobalMatrix<TestDOF_t,TrialDOF_t,ComputePolicy_t>;

		using MatStorage_t    = typename ScatterPolicy_t::MatStorage_t;

		ComputePolicy_t compute_policy;
		ScatterPolicy_t scatter_policy;

		//constructor
		BilinearFormAssembler(const TestHandler_type& TestH, const TrialHandler_type& TrialH) : 
			BASE(TestH, TrialH), scatter_policy{compute_policy} {}

		//link to global storage
		inline void set_global(MatStorage_t& coo) {scatter_policy.set_global(coo);}

		//consistent api
		inline void prepare() {}
		inline void finalize() {}

		template<uint64_t N_QUAD_POINTS>
		inline void compute(const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map) {
			compute_policy.compute(this->test_dofs, this->trial_dofs, q_map);
		}

		inline void scatter() {
			scatter_policy.scatter(this->test_dofs, this->trial_dofs);
		}
	};

}