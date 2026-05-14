#pragma once

#include "fem/forms/bilinear_form.hpp"
#include "fem/forms/bilinear_form_compute.hpp"
#include "fem/forms/bilinear_form_scatter.hpp"
#include "fem/forms/local_actions.hpp"
#include "fem/numerics/quad_point_map.hpp"
#include "util/compatibility.hpp"

namespace GV
{
	//a class to compute Y=A*X without forming the stiffness matrix A
	template<typename TestHandler_type, typename TrialHandler_type, typename EvalPolicy>
	struct BilinearFormMultiply : public BilinearForm<TestHandler_type,TrialHandler_type,EvalPolicy>
	{
		using BASE       = BilinearForm<TestHandler_type,TrialHandler_type,EvalPolicy>;
		using TestDOF_t  = typename BASE::TestDOF_t;
		using TrialDOF_t = typename BASE::TrialDOF_t;
		using QuadElem_t = typename BASE::QuadElem_t;

		using ComputePolicy_t = BilinearFormComputeLocalMat<TestDOF_t,TrialDOF_t,EvalPolicy>;
		using ScatterPolicy_t = BilinearFormScatterLocalVec<TestDOF_t>; //scatter to rows (test dofs)
		
		//allow multiple x and y vectors to be computed on
		struct Pair {std::span<double> y; std::span<const double> x;};
		std::vector<Pair> global_pairs;
		
		//store local Y results
		std::vector<double> Y;
		std::span<double> loc_y(const uint64_t k) {
			const uint64_t start = k * this->n_test;
			assert(k<global_pairs.size());
			assert(start + this->n_test <= Y.size());
			return as_span(Y,start,this->n_test);
		}
		void loc_x(std::span<double> result, const uint64_t k) const {
			assert(k<global_pairs.size());
			for (uint64_t j=0; j<this->m_trial; ++j) {
				const uint64_t J = this->global_trial[j];
				result[j] = global_pairs[k].x[J];
			}
		}

		ComputePolicy_t compute_policy;
		ScatterPolicy_t scatter_policy;

		//constructor
		using BASE::BASE;

		//link to global storage
		inline void set_global(std::span<double> y, std::span<const double> x) {
			assert(y.size() == this->test_handler.n_dofs());
			assert(x.size() == this->trial_handler.n_dofs());
			global_pairs.emplace_back(y,x);
		}

		template<typename ContA_t, typename ContB_t>
		inline void set_global(ContA_t& y, const ContB_t& x) {set_global(as_span(y), as_span(x));}

		inline void prepare() {
			Y.assign(global_pairs.size() * this->n_test, 0.0);
		}

		template<uint64_t N_QUAD_POINTS>
		inline void compute(
				const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map) {
			compute_policy.compute(this->test_dofs, this->trial_dofs, q_map);
		}

		inline void finalize() {
			std::vector<double> x(this->m_trial);
			for (size_t k=0; k<global_pairs.size(); ++k) {
				loc_x(x,k);
				local_multiply(loc_y(k), x, compute_policy.data());
			}
		}

		inline void scatter() {
			for (size_t k=0; k<global_pairs.size(); ++k) {
				scatter_policy.set_global(global_pairs[k].y);
				scatter_policy.set_local(loc_y(k));
				scatter_policy.scatter(this->global_test);
			}
		}
	};

}