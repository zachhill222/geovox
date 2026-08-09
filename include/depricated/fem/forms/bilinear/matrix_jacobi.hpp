#pragma once

#include "fem/forms/bilinear/base.hpp"
#include "fem/forms/bilinear/policy_compute.hpp"
#include "fem/forms/bilinear/policy_scatter.hpp"
#include "fem/forms/bilinear/local_actions.hpp"

#include "fem/numerics/quad_point_map.hpp"

namespace GV
{
	//a class to estimate X=A^-1*Y without forming the stiffness matrix A
	//The computation is done in two passes. First, the current guess of X is provided as X0,
	//Then X+= Y - (L+U)*X0 is computed (X must be zeroed before calling). Simultaneously, the diagonal D of A is formed
	//The final pass X/=D must be done by the calling class. Note that D  must be zeroed ahead of time.
	//Note that X corresponds to the trial dofs and Y to the test. Their handlers must have the same number of dofs (so that A is square), but need not be identical.
	//
	//After the loop over the elements is finished, the finalize_inverse() method can be called to do the X/=D computation.
	//The diagonal D will be replaced with its reciprocal after calling finalize_inverse()
	template<typename TestHandler_type, typename TrialHandler_type, typename EvalPolicy>
	struct BilinearFormJacobi : public BilinearForm<TestHandler_type,TrialHandler_type,EvalPolicy>
	{
		using BASE       = BilinearForm<TestHandler_type,TrialHandler_type,EvalPolicy>;
		using TestDOF_t  = typename BASE::TestDOF_t;
		using TrialDOF_t = typename BASE::TrialDOF_t;
		using QuadElem_t = typename BASE::QuadElem_t;

		using ComputePolicy_t = BilinearFormComputeLocalMat<TestDOF_t,TrialDOF_t,EvalPolicy>;
		using ScatterPolicy_t = BilinearFormScatterLocalVec<TestDOF_t>; //scatter to rows (test dofs)
		
		//allow multiple x and y vectors to be computed on
		struct Triple {std::span<double> x; std::span<const double> x0; std::span<const double> y;};
		std::vector<Triple> global_pairs;
		std::span<double> D; //global diagonal

		//store local X results
		std::vector<double> X;
		std::span<double> loc_x(const uint64_t k) {
			const uint64_t start = k * this->m_trial;
			assert(k<global_pairs.size());
			assert(start + this->m_trial <= X.size());
			return as_span(X,start,this->m_trial);
		}

		//look up provided values for the current dofs
		void loc_y(std::span<double> result, const uint64_t k) const {
			assert(k<global_pairs.size());
			#pragma omp simd
			for (uint64_t i=0; i<this->n_test; ++i) {
				const uint64_t I = this->global_test[i];
				result[i] = global_pairs[k].y[I];
			}
		}
		void loc_x0(std::span<double> result, const uint64_t k) const {
			assert(k<global_pairs.size());
			#pragma omp simd
			for (uint64_t j=0; j<this->m_trial; ++j) {
				const uint64_t J = this->global_trial[j];
				result[j] = global_pairs[k].x0[J];
			}
		}

		ComputePolicy_t compute_policy;
		ScatterPolicy_t scatter_policy;

		//constructor
		using BASE::BASE;

		//link to global storage
		template<typename ContA_t>
		inline void set_global(ContA_t& d) {set_global(as_span(d));}
		inline void set_global(std::span<double> d) {D=d;}

		template<typename ContA_t, typename ContB_t, typename ContC_t>
		inline void set_global(ContA_t& x, const ContB_t& x0, const ContC_t& y) {set_global(as_span(x), as_span(x0), as_span(y));}
		inline void set_global(std::span<double> x, std::span<const double> x0, std::span<const double> y) {
			assert(y.size() == this->test_handler.n_dofs());
			assert(x.size() == this->trial_handler.n_dofs());
			assert(x0.size() == this->trial_handler.n_dofs());
			global_pairs.emplace_back(x,x0,y);
		}

		inline void prepare() {
			X.assign(global_pairs.size() * this->n_test, 0.0);
		}

		template<uint64_t N_QUAD_POINTS>
		inline void compute(const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map) {
			assert(this->n_test == this->m_trial);
			compute_policy.compute(this->test_dofs, this->trial_dofs, q_map);
		}

		inline void finalize() {
			std::vector<double> x0(this->m_trial);
			std::vector<double> y(this->n_test);
			for (size_t k=0; k<global_pairs.size(); ++k) {
				loc_x0(x0,k);
				loc_y(y,k);
				std::span<double> x = loc_x(k); //should start at 0
				local_multiply_upper_lower(x, x0, compute_policy.data()); //x+=(L+U)*x0
				#pragma omp simd
				for (uint64_t i=0; i<this->n_test; ++i) {
					x[i] = y[i]-x[i];
				}
			}
		}

		inline void scatter() {
			//scatter each X += Y - (L+U)*X0
			for (size_t k=0; k<global_pairs.size(); ++k) {
				scatter_policy.set_global(global_pairs[k].x);
				scatter_policy.set_local(loc_x(k));
				scatter_policy.scatter(this->global_trial);
			}

			//scatter D (note D is the diagonal of A so its rows are test dofs)
			std::vector<double> d = compute_policy.diag();
			scatter_policy.set_global(D);
			scatter_policy.set_local(d);
			scatter_policy.scatter(this->global_test);
		}

		void finalize_inverse() {
			size_t K = global_pairs.size();
			size_t N = D.size();

			assert(K>0);
			//compute reciprocals and first division
			std::span<double> x = global_pairs[0].x;
			#pragma omp simd
			for (size_t i=0; i<N; ++i) {
				assert(D[i]!=0);
				D[i] = 1.0 / D[i];
				x[i] *= D[i];
			}

			//compute the rest of the "division"
			for (size_t k=1; k<K; ++k) {
				x = global_pairs[k].x;
				#pragma omp simd
				for (size_t i=0; i<N; ++i) {
					x[i] *= D[i];
				}
			}
		}
	};

}