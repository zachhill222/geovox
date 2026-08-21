#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"

#include "fem/dofhandler.hpp"

#include "fem/mesh_quadrature.hpp"
#include "fem/forms/linearforms/linear_kernels.hpp"

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
	template<int N, typename T, typename TestHandlerType, typename KernelType, typename KernelWeightType=IdentityKernelWeight>
	struct LinearForm {


		//////////////////////////////////////////////////////////////////
		/// Aliases and sanity checks
		//////////////////////////////////////////////////////////////////
		using Kernel_t      = KernelType;
		using Weight_t   	= KernelWeightType;
		using TestHandler_t	= TestHandlerType;
		using QuadRule_t 	= MeshQuadratureRule<N,T>;
		using Scalar_t   	= T;
		using Mesh_t     	= UnstructuredVoxelMesh<T>;
		using MeshElem_t 	= typename Mesh_t::Elem_t;
		using TestDof_t  	= typename TestHandlerType::DOF_t;

		using ElementTestCache_t = ElementDofCache<Kernel_t,TestHandler_t,QuadRule_t>;
		using WeightCache_t      = ScalarValueCache<QuadRule_t>;

		struct KernelEval {
			const Kernel_t& 			kernel;
			const ElementTestCache_t& 	cache;
			const QuadRule_t& 			qr;

			KernelEval(const Kernel_t& k, const ElementTestCache_t& c, const QuadRule_t& q) : kernel(k), cache(c), qr(q) {}

			[[nodiscard]] Scalar_t operator()(size_t i, const WeightCache_t* wt = nullptr) const noexcept {
				const DofValueCache<QuadRule_t>* vals{nullptr};
				const DofGradCache<QuadRule_t>*  grad{nullptr};
				if constexpr (Kernel_t::NEEDS_DOF_VALS) {vals = &cache.vals[i];}
				if constexpr (Kernel_t::NEEDS_DOF_GRAD) {grad = &cache.grad[i];}
				return kernel.cached_eval(vals, grad, wt, qr);
			}
		};


		//////////////////////////////////////////////////////////////////
		/// Data and constructor
		//////////////////////////////////////////////////////////////////
		const TestHandler_t&	test_handler;
		const Mesh_t&	 		mesh;
		const Kernel_t 			kernel;
		const Weight_t 			weight;

		LinearForm(const TestHandler_t& handler, KernelType kernel = KernelType{}, KernelWeightType weight = KernelWeightType{}) :
			test_handler(handler), mesh(test_handler.mesh), kernel(std::move(kernel)), weight(std::move(weight)) {}


		//////////////////////////////////////////////////////////////////
		/// Implementation of evaluation methods to scatter to a global vector
		//////////////////////////////////////////////////////////////////
		void evaluate_vector_colored(std::span<T> result) noexcept {
			// compute result[i]+=L(dof[i])
			static_assert(!Weight_t::NEEDS_SCALAR_VALS);
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(result.size() == test_handler.n_dofs());
			GUTIL_ASSERT(mesh.is_color_sorted());

			for (int i=0; i<mesh.n_colors(); ++i) {
				std::span<const MeshElem_t> quad_elems = mesh.get_color(i);

				GUTIL_OMP(parallel)
				{
					QuadRule_t 						quad_rule(mesh);
					ElementTestCache_t				test_cache(test_handler,quad_rule);	
					KernelEval						k_eval(kernel, test_cache, quad_rule);
					WeightCache_t					wt;

					const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
					const size_t n_els_per_thread = quad_elems.size()/n_threads;
					const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
					const size_t start 			  = tid*n_els_per_thread;
					const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
					for (size_t j=start; j<end; ++j) {
						const MeshElem_t el = quad_elems[j];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						test_cache.gather_qh();
						if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
							wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
						}

						for (size_t i=0; i<test_cache.size(); ++i) {
							result[test_cache.global_idx[i]] += k_eval(i, &wt);
						}
					}
				}
			}
		}

		void evaluate_vector(std::span<T> result) noexcept {
			// compute result[i]+=L(dof[i])
			static_assert(!Weight_t::NEEDS_SCALAR_VALS);
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(result.size() == test_handler.n_dofs());

			std::span<const MeshElem_t> quad_elems{mesh.element_begin(), mesh.element_end()};

			GUTIL_OMP(parallel)
			{
				QuadRule_t 					quad_rule(mesh);
				ElementTestCache_t			test_cache(test_handler,quad_rule);
				KernelEval					k_eval(kernel, test_cache, quad_rule);
				WeightCache_t				wt;

				GUTIL_OMP(for)
				for (size_t i=0; i<quad_elems.size(); ++i) {
					const MeshElem_t el = quad_elems[i];
					
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					test_cache.gather_qh();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
					}
					
					for (size_t i=0; i<test_cache.size(); ++i) {
						GUTIL_OMP(atomic) result[test_cache.global_idx[i]] += k_eval(i, &wt);
					}
				}
			}
		}

		//////////////////////////////////////////////////////////////////
		/// Implementation of evaluation methods to evaluate the linear form
		/// on a function defined via the dof handler
		//////////////////////////////////////////////////////////////////
		T evaluate_form(std::span<const T> coefs) noexcept {
			// compute result = L( sum_i coefs[i]*dof[i] )
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(coefs.size() == test_handler.n_dofs());

			std::span<const MeshElem_t> quad_elems{mesh.element_begin(), mesh.element_end()};
			Scalar_t result{0};

			GUTIL_OMP(parallel)
			{
				Scalar_t 					thread_result{0};
				QuadRule_t 					quad_rule(mesh);
				ElementTestCache_t			test_cache(test_handler,quad_rule);	
				KernelEval					k_eval(kernel, test_cache, quad_rule);
				WeightCache_t				wt;

				GUTIL_OMP(for)
				for (size_t i=0; i<quad_elems.size(); ++i) {
					const MeshElem_t el = quad_elems[i];
					
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					test_cache.gather_qh();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
					}
					
					for (size_t i=0; i<test_cache.size(); ++i) {
						thread_result += coefs[test_cache.global_idx[i]] * k_eval(i,&wt);
					}
				}

				GUTIL_OMP(atomic) result += thread_result;
			}
			return result;
		}


		//////////////////////////////////////////////////////////////////
		/// For computing errors, it is convenient to just integrate the weight.
		/// We assume that when using this function, any required scalar field is
		/// from the test handler.
		///
		/// TODO: maybe move out of linearform.
		//////////////////////////////////////////////////////////////////
		T integrate_weight(std::span<const Scalar_t> coefs) const noexcept {
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(coefs.size() == test_handler.n_dofs());

			Scalar_t result{0};

			GUTIL_OMP(parallel)
			{
				Scalar_t 						thread_result{0};
				QuadRule_t 						quad_rule(mesh);
				ElementTestCache_t				test_cache(test_handler,quad_rule);	
				ScalarValueCache<QuadRule_t>	field_cache;
				WeightCache_t					wt;

				OmpIteratorRange				range(mesh.element_begin(), mesh.element_end());

				GUTIL_OMP(for)
				for (auto it=range.begin; it!=range.end; ++it) {
					const MeshElem_t el = *it;
					
					quad_rule.set_element(el,2);
					test_cache.gather_qh();

					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					if constexpr (Weight_t::NEEDS_SCALAR_VALS) {field_cache = test_cache.reconstruct_field(coefs);}

					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(&field_cache, quad_rule);
					}
					
					Scalar_t el_result{0};
					auto q_wt = quad_rule.quad_w();
					for (size_t i=0; i<test_cache.size(); ++i) {
						el_result += wt[i]*q_wt[i];
					}
					thread_result += el_result*quad_rule.jacobian_det[el.depth()];
				}

				GUTIL_OMP(atomic) result += thread_result;
			}
			return result;
		}
	};
}