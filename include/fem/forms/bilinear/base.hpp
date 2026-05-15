#pragma once

#include "util/compatibility.hpp"

#include <vector>
#include <span>

namespace GV
{	
	//base class for integrating bilinear forms over a mesh
	//a kernel will own a referenece to a bilinear form and call the folowing in this order:
	//		set_basis(test,trial)
	//		prepare()
	//		compute(quad_point_map)
	//		finalize()
	//		scatter()
	//
	//set_basis is exactly as it is in this class.
	//compute should be handled by a compute policy that handles any memory management and determines
	//if the full local matrix needs to be stored or just the main diagonal.
	//
	//prepare() and finalize() can be no-ops but must be present
	//scatter() passes any final result back to the primary calling method. this should involve
	// *scattering* local element contrubutions to the global.

	template<typename TestHandler_type, typename TrialHandler_type, typename EvalPolicy>
	struct BilinearForm {
		//tag to tell that this is a bilinear form (vs linear) with various compile-time mechanisms
		static constexpr bool IS_BILINEAR_FROM = true;
		static constexpr bool IS_LINEAR_FORM   = false;

		//store symmetry in the evaluation policy
		static constexpr bool IS_SYMMETRIC = EvalPolicy::IS_SYMMETRIC;

		//alias types
		using Mesh_t         = typename TestHandler_type::Mesh_t;
		using TestHandler_t  = TestHandler_type;
		using TrialHandler_t = TrialHandler_type;
		using TestDOF_t      = typename TestHandler_type::DOF_t;
		using TrialDOF_t     = typename TrialHandler_type::DOF_t;
		using QuadElem_t     = typename TrialDOF_t::QuadElem_t::NonPeriodicVariant;

		//quick sanity check
		static_assert(std::same_as<typename TestHandler_type::Mesh_t,typename TrialHandler_type::Mesh_t>,
			"BilinearForm - Test and Trial handlers must have the same mesh type.");

		static_assert(std::same_as<typename TrialDOF_t::QuadElem_t::NonPeriodicVariant, typename TestDOF_t::QuadElem_t::NonPeriodicVariant>,
			"BilinearForm - The test and trial dofs must have compatible quadrature elements.");

		static_assert(!IS_SYMMETRIC || std::same_as<TestHandler_type,TrialHandler_type>,
			"BilinearForm - A symmetric bilinear form must have the same test and trial dof handlers");

		//constructor to link dof handlers
		BilinearForm(const TestHandler_t& TestH, const TrialHandler_t& TrialH) 
			: test_handler(TestH), trial_handler(TrialH), mesh(test_handler.mesh) {}

		//store dofs on the current element
		std::span<const TestDOF_t>  test_dofs;		//local test basis functions (row dofs) (note a span is non-owning)
		std::span<const TrialDOF_t> trial_dofs; 	//local trial basis functions (column dofs)
		uint64_t n_test=0, m_trial=0;

		//store global dof information
		const TestHandler_t&		test_handler;	//link to handler for the test dofs
		const TrialHandler_t&		trial_handler;	//link to handler for the trial dofs
		std::vector<uint64_t> 		global_test; 	//track global dof numbers
		typename std::conditional_t<IS_SYMMETRIC, std::span<const uint64_t>, std::vector<uint64_t>> global_trial; //test and trial dofs are the same for a symmetric form

		//store an accessible link to the mesh
		const Mesh_t& mesh;

		//set dofs on the current element
		template<typename ContainerA_t, typename ContainerB_t>
		inline void set_basis(const ContainerA_t& test, const ContainerB_t& trial) {
			set_basis(as_span(test), as_span(trial));
		}

		void set_basis(std::span<const TestDOF_t> test, std::span<const TrialDOF_t> trial) {
			test_dofs   = test;
			trial_dofs  = trial;
			n_test      = test.size();
			m_trial     = trial.size();

			//convert local indices to global
			global_test.resize(test.size());
			for (uint64_t i=0; i<test.size(); ++i)  {global_test[i]  = test_handler.compressed_index(test[i]);}

			if constexpr (IS_SYMMETRIC) {global_trial = std::span<const uint64_t>{global_test};}
			else {
				global_trial.resize(trial.size());
				for (uint64_t i=0; i<trial.size(); ++i)  {global_trial[i]  = trial_handler.compressed_index(trial[i]);}
			}
		}
	};
}
