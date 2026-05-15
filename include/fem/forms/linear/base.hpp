#pragma once

#include "util/compatibility.hpp"

#include <vector>
#include <span>

namespace GV
{	
	//base class for integrating linear forms over a mesh
	//a kernel will own a referenece to a bilinear form and call the folowing in this order:
	//		set_basis(test)
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

	template<typename TestHandler_type, typename EvalPolicy>
	struct LinearForm {
		//alias types
		using Mesh_t         = typename TestHandler_type::Mesh_t;
		using TestHandler_t  = TestHandler_type;
		using TestDOF_t      = typename TestHandler_type::DOF_t;
		using QuadElem_t     = typename TestDOF_t::QuadElem_t::NonPeriodicVariant;

		//constructor to link dof handlers
		LinearForm(const TestHandler_t& TestH) : test_handler(TestH), mesh(test_handler.mesh) {}

		//store dofs on the current element
		std::span<const TestDOF_t>  test_dofs;		//local test basis functions (row dofs)
		uint64_t n_test=0;

		//store global dof information
		const TestHandler_t&		test_handler;	//link to handler for the test dofs
		std::vector<uint64_t> 		global_test; 	//track global dof numbers

		//store an accessible link to the mesh
		const Mesh_t& mesh;

		//set dofs on the current element
		template<typename ContainerA_t>
		inline void set_basis(const ContainerA_t& test) {
			set_basis(as_span(test));
		}

		void set_basis(std::span<const TestDOF_t> test) {
			test_dofs   = test;
			n_test      = test.size();

			//convert local indices to global
			global_test.resize(test.size());
			for (uint64_t i=0; i<test.size(); ++i)  {global_test[i]  = test_handler.compressed_index(test[i]);}
		}
	};
}
