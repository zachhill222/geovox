#pragma once

#include "fem/forms/linear/base.hpp"
#include "fem/forms/linear/policy_compute.hpp"
#include "fem/forms/linear/policy_scatter.hpp"

#include "fem/numerics/quad_point_map.hpp"


namespace GV
{
	//a class to the action of a linear form without assembling a global list of dof results
	template<typename TestHandler_type, typename EvalPolicy>
	struct LinearFormAssembler : public LinearForm<TestHandler_type,EvalPolicy>
	{
		using BASE       = LinearForm<TestHandler_type,EvalPolicy>;
		using TestDOF_t  = typename BASE::TestDOF_t;
		using QuadElem_t = typename BASE::QuadElem_t;

		using ComputePolicy_t = LinearFormComputeLocalVec<TestDOF_t,EvalPolicy>;
		using ScatterPolicy_t = LinearFormScatterToGlobalVector<TestDOF_t>; //scatter to rows (test dofs)
		
		ComputePolicy_t compute_policy;
		ScatterPolicy_t scatter_policy;

		//constructor
		LinearFormAssembler(const TestHandler_type& TestH) : BASE(TestH) {}
		LinearFormAssembler(const TestHandler_type& TestH, EvalPolicy eval) : BASE(TestH), compute_policy{eval} {}

		//link to global storage
		inline void set_global(std::span<double> y) {
			assert(y.size() == this->test_handler.n_dofs());
			scatter_policy.set_global(y);
		}

		template<typename ContA_t>
		inline void set_global(ContA_t& y) {set_global(as_span(y));}

		inline void prepare() {}

		template<uint64_t N_QUAD_POINTS>
		inline void compute(const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map) {
			compute_policy.compute(this->test_dofs, q_map);
		}

		inline void finalize() {}

		inline void scatter() {
			scatter_policy.set_local(compute_policy.data());
			scatter_policy.scatter(this->global_test);
		}
	};
}