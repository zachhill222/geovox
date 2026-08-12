#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"

#include "fem/mesh_quadrature.hpp"
#include "fem/linearforms/linear_kernels.hpp"
#include "fem/dofhandler.hpp"

namespace GV {


	//////////////////////////////////////////////////////////////////
	/// A class for coordinating quadrature of a linear form over a mesh.
	///	
	/// A linear form requires a single dofhandler plus a kernel
	///
	/// The kernel of a linear form L(v) is K s.t. 
	///				L(v) = int_D( K(v(x)) dx)
	/// where v = sum_i c[i]*test_dof[i] for some coefficients c over the dofs in its handler.
	///
	/// The kernel function object will be passed a dof and a quadrature rule
	/// and produces a scalar. The quadrature rule contians information such as
	/// quadrature points and weights on the current element as well as projections
	/// of to relevant lower depths.
	/////////////////////////////////////////////////////////////////////
	template<typename KernelType, typename TestHandlerType, int N, typename T>
	struct LinearForm {


		//////////////////////////////////////////////////////////////////
		/// Aliases and sanity checks
		//////////////////////////////////////////////////////////////////
		using Kernel_t      = KernelType;
		using TestHandler_t	= TestHandlerType;
		using QuadRule_t 	= MeshQuadratureRule<N,T>;
		using Scalar_t   	= T;
		using Mesh_t     	= UnstructuredVoxelMesh<T>;
		using MeshElem_t 	= typename Mesh_t::Elem_t;
		using TestDof_t  	= typename TestHandlerType::DOF_t;

		static_assert(std::is_invocable_r_v<T,KernelType,TestDof_t,const QuadRule_t&>, 
			"The Kernel must have the signature T(TestDof_t,const QuadRule_t&)");


		//////////////////////////////////////////////////////////////////
		/// Data and constructor
		//////////////////////////////////////////////////////////////////
		const TestHandler_t&	test_handler;
		const Mesh_t&	 		mesh;
		const Kernel_t 			kernel;

		LinearForm(const TestHandler_t& handler, KernelType&& kernel = KernelType{}) :
			test_handler(handler), mesh(test_handler.mesh), kernel(std::move(kernel)) {}


		//////////////////////////////////////////////////////////////////
		/// Implementation of evaluation methods to scatter to a global vector
		//////////////////////////////////////////////////////////////////
		void evaluate_vector_colored(std::span<T> result) noexcept {
			// compute result[i]=L(dof[i])
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(result.size() == test_handler.n_dofs());
			GUTIL_ASSERT(mesh.is_color_sorted());

			for (int i=0; i<mesh.n_colors(); ++i) {
				std::span<const MeshElem_t> quad_elems = mesh.get_color(i);

				GUTIL_OMP(parallel)
				{
					QuadRule_t 						quad_rule(mesh);
					std::vector<TestDof_t> 			test_dofs_on_elem;

					const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
					const size_t n_els_per_thread = quad_elems.size()/n_threads;
					const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);\
					const size_t start 			  = tid*n_els_per_thread;
					const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
					for (size_t j=start; j<end; ++j) {
						const MeshElem_t el = quad_elems[j];
						quad_rule.set_element(el,mesh.max_depth);
						test_dofs_on_elem = test_handler.get_active_dofs_full_hierarchical(el);

						for (TestDof_t dof : test_dofs_on_elem) {
							size_t global_n = test_handler.global_number(dof);
							GUTIL_ASSERT(global_n<result.size());
							result[global_n] += kernel(dof,quad_rule);
						}
					}
				}
			}
		}

		void evaluate_vector(std::span<T> result) noexcept {
			// compute result[i]=L(dof[i])
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(result.size() == test_handler.n_dofs());

			std::span<const MeshElem_t> quad_elems{mesh.element_begin(), mesh.element_end()};

			GUTIL_OMP(parallel)
			{
				QuadRule_t 				quad_rule(mesh);
				std::vector<TestDof_t> 	test_dofs_on_elem;

				GUTIL_OMP(for)
				for (size_t i=0; i<quad_elems.size(); ++i) {
					const MeshElem_t el = quad_elems[i];
					quad_rule.set_element(el,mesh.max_depth);
					test_dofs_on_elem = test_handler.get_active_dofs_full_hierarchical(el);

					for (TestDof_t dof : test_dofs_on_elem) {
						size_t global_n = test_handler.global_number(dof);
						GUTIL_ASSERT(global_n<result.size());
						GUTIL_OMP(atomic) result[global_n] += kernel(dof,quad_rule);
					}
				}
			}
		}

		//////////////////////////////////////////////////////////////////
		/// Implementation of evaluation methods to evaluate the linear form
		/// on a function defined via the dof handler
		//////////////////////////////////////////////////////////////////
		// T evaluate_colored(std::span<const T> coefs) noexcept {
		// 	// compute result = L( sum_i coefs[i]*dof[i] )
		// 	GUTIL_ASSERT(test_handler.is_current());
		// 	GUTIL_ASSERT(result.size() == test_handler.n_dofs());
		// 	GUTIL_ASSERT(mesh.is_color_sorted());

		// 	Scalar_t result{0};
		// 	for (int i=0; i<mesh.n_colors(); ++i) {
		// 		std::span<const Elem_t> quad_elems = mesh.get_color(i);

		// 		GUTIL_OMP(parallel)
		// 		{
		// 			std::array<T,QuadRule_t::TOTAL_QUAD_POINTS> dof_vals;
		// 			QuadRule_t 						quad_rule(mesh);
		// 			std::vector<TestDof_t> 			test_dofs_on_elem;

		// 			Scalar_t thread_result{0};

		// 			const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
		// 			const size_t n_els_per_thread = quad_elems.size()/n_threads;
		// 			const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
		// 			const size_t start 			  = tid*n_els_per_thread;
		// 			const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
		// 			for (size_t j=start; j<end; ++j) {
		// 				const MeshElem_t el = quad_elems[j];
		// 				quad_rule.set_element(el,mesh.max_depth);
		// 				test_dofs_on_elem = test_handler.get_active_dofs_full_hierarchical(el);

		// 				for (TestDof_t dof : test_dofs_on_elem) {
		// 					size_t global_n = test_handler.global_number(dof);
		// 					GUTIL_ASSERT(global_n<result.size());
		// 					thread_result += coefs[global_n]*kernel(dof,quad_rule);
		// 				}
		// 			}

		// 			GUTIL_OMP(atomic) result += thread_result;
		// 		}
		// 	}
		// 	return result;
		// }

		T evaluate_form(std::span<const T> coefs) noexcept {
			// compute result = L( sum_i coefs[i]*dof[i] )
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(coefs.size() == test_handler.n_dofs());

			std::span<const MeshElem_t> quad_elems{mesh.element_begin(), mesh.element_end()};
			Scalar_t result{0};

			GUTIL_OMP(parallel)
			{
				Scalar_t 				thread_result{0};
				QuadRule_t 				quad_rule(mesh);
				std::vector<TestDof_t> 	test_dofs_on_elem;

				GUTIL_OMP(for)
				for (size_t i=0; i<quad_elems.size(); ++i) {
					const MeshElem_t el = quad_elems[i];
					quad_rule.set_element(el,mesh.max_depth);
					test_dofs_on_elem = test_handler.get_active_dofs_full_hierarchical(el);

					for (TestDof_t dof : test_dofs_on_elem) {
						size_t global_n = test_handler.global_number(dof);
						GUTIL_ASSERT(global_n<coefs.size());
						thread_result += coefs[global_n]*kernel(dof,quad_rule);
					}
				}

				GUTIL_OMP(atomic) result += thread_result;
			}
			return result;
		}

	};
}