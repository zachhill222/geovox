#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"

#include "fem/mesh_quadrature.hpp"
#include "fem/forms/util.hpp"
#include "fem/forms/util.hpp"
#include "fem/forms/base_k_linear_form.hpp"

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
	template<int N, typename T, typename TestHandlerType, typename KernelType>
	class LinearForm : public KLinearForm<N,T,KernelType,TestHandlerType> {
	public:

		//////////////////////////////////////////////////////////////////
		/// Aliases and sanity checks
		//////////////////////////////////////////////////////////////////
		using BASE          = KLinearForm<N,T,KernelType,TestHandlerType>;
		using Kernel_t      = KernelType;
		using QuadRule_t 	= MeshQuadratureRule<N,T>;
		using Scalar_t   	= T;
		using Mesh_t     	= UnstructuredVoxelMesh<T>;
		using MeshElem_t 	= typename Mesh_t::Elem_t;
		using TestHandler_t	= TestHandlerType;
		using TestDof_t  	= typename TestHandlerType::DOF_t;

		static_assert(std::same_as<TestDof_t, typename BASE::template GetDofType<0>>);


		//////////////////////////////////////////////////////////////////
		/// Constructor
		//////////////////////////////////////////////////////////////////
		LinearForm(const TestHandler_t& handler, KernelType kernel = KernelType{}) :
			BASE(handler.mesh, std::move(kernel)) {BASE::set_handlers(handler);}


		//////////////////////////////////////////////////////////////////
		/// A few convenience methods
		//////////////////////////////////////////////////////////////////
		const TestHandler_t& test_handler() const noexcept {return this->template get_handler<0>();}
		static const auto& get_test_cache(const typename BASE::KernelEval& k_eval) noexcept {return k_eval.template get_cache<0>();}
		[[nodiscard]] size_t n_dofs() const noexcept {return test_handler().n_dofs();}


		//////////////////////////////////////////////////////////////////
		/// A thin wrapper preserving the established action-first calling convention,
		/// on top of KLinearForm::for_each_element (init-first).
		//////////////////////////////////////////////////////////////////
		template<bool Colored=false, typename Action, typename Init=std::nullptr_t, typename Finalize=std::nullptr_t>
		void for_each_element(Action&& action, Init&& init=nullptr, Finalize&& finalize=nullptr) const noexcept {
			GUTIL_ASSERT(this->template get_handler<0>().is_current());
			BASE::template for_each_element<Colored>(std::forward<Action>(action), std::forward<Init>(init), std::forward<Finalize>(finalize));
		}

		//////////////////////////////////////////////////////////////////
		/// Implementation of evaluation methods to scatter to a global vector
		//////////////////////////////////////////////////////////////////
		void evaluate_vector_colored(std::span<T> result) const noexcept {
			// compute result[i]+=L(dof[i])
			GUTIL_ASSERT(test_handler().is_current());
			GUTIL_ASSERT(result.size() == test_handler().n_dofs());
			GUTIL_ASSERT(this->mesh().is_color_sorted());

			for_each_element<true>(
			[&](auto& k_eval, const auto&) {
				const auto& test_cache = get_test_cache(k_eval);
				const size_t n = test_cache.dofs.size();
				for (size_t i=0; i<n; ++i) {
					result[test_cache.global_idx[i]] += k_eval({i});
				}
			});
		}

		void evaluate_vector(std::span<T> result) const noexcept {
			// compute result[i]+=L(dof[i])
			GUTIL_ASSERT(test_handler().is_current());
			GUTIL_ASSERT(result.size() == test_handler().n_dofs());

			for_each_element(
			[&](auto& k_eval, const auto&) {
				const auto& test_cache = get_test_cache(k_eval);
				const size_t n = test_cache.dofs.size();
				for (size_t i=0; i<n; ++i) {
					GUTIL_OMP(atomic) result[test_cache.global_idx[i]] += k_eval({i});
				}
			});
		}

		//////////////////////////////////////////////////////////////////
		/// Implementation of evaluation methods to evaluate the linear form
		/// on a function defined via the dof handler
		//////////////////////////////////////////////////////////////////
		T evaluate_form(std::span<const T> coefs) const noexcept {
			GUTIL_ASSERT(test_handler().is_current());
			GUTIL_ASSERT(coefs.size() == test_handler().n_dofs());
			return BASE::evaluate(coefs);
		}
	};
}