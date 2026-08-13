#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"

#include "fem/dofhandler.hpp"

#include "fem/mesh_quadrature.hpp"
#include "fem/bilinearforms/bilinear_kernels.hpp"
#include "fem/bilinearforms/dense_linalg.hpp"

namespace GV {


	//////////////////////////////////////////////////////////////////
	/// A class for coordinating quadrature of a bilinear form over a mesh.
	///	
	/// A bilinear form requires two dofhandlers plus a kernel
	///
	/// The kernel of a bilinear form B(u,v) is K s.t. 
	///				L(v) = int_D( K(v(x)) dx)
	/// where v = sum_i c[i]*test_dof[i] for some coefficients c over the dofs in its handler.
	///
	/// The kernel function object will be passed a dof and a quadrature rule
	/// and produces a scalar. The quadrature rule contians information such as
	/// quadrature points and weights on the current element as well as projections
	/// of to relevant lower depths.
	/////////////////////////////////////////////////////////////////////
	template<typename TrialHandlerType, typename TestHandlerType, typename KernelType, int N, typename T>
	struct BilinearForm {


		//////////////////////////////////////////////////////////////////
		/// Aliases and sanity checks
		//////////////////////////////////////////////////////////////////
		using Kernel_t      	= KernelType;
		using TrialHandler_t	= TrialHandlerType;
		using TestHandler_t		= TestHandlerType;
		using QuadRule_t 		= MeshQuadratureRule<N,T>;
		using Scalar_t   		= T;
		using Mesh_t     		= UnstructuredVoxelMesh<T>;
		using MeshElem_t 		= typename Mesh_t::Elem_t;
		using TrialDof_t  		= typename TrialHandlerType::DOF_t;
		using TestDof_t  		= typename TestHandlerType::DOF_t;

		static_assert(std::is_invocable_r_v<T,KernelType,TrialDof_t,TestDof_t,const QuadRule_t&>, 
			"The Kernel must have the signature T(TrialDof_t,TestDof_t,const QuadRule_t&)");

		static_assert(!Kernel_t::IS_SYMMETRIC || std::same_as<TrialHandler_t,TestHandler_t>);

		//////////////////////////////////////////////////////////////////
		/// Data and constructor
		//////////////////////////////////////////////////////////////////
		const TrialHandler_t&	trial_handler;
		const TestHandler_t&	test_handler;
		const Mesh_t&	 		mesh;
		const Kernel_t 			kernel;

		BilinearForm(const TrialHandler_t& u_handler, const TestHandler_t& v_handler, KernelType&& kernel = KernelType{}) :
			trial_handler(u_handler), test_handler(v_handler), mesh(test_handler.mesh), kernel(std::move(kernel)) {}



		//////////////////////////////////////////////////////////////////
		/// Given trial coefficients X, compute Y += alpha*A*X
		//////////////////////////////////////////////////////////////////
		void mat_vec_multiply_accumulate_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) noexcept requires(!Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(X.size() == trial_handler.n_dofs());
			GUTIL_ASSERT(Y.size() == test_handler.n_dofs());
			GUTIL_ASSERT(mesh.is_color_sorted());

			for (int i=0; i<mesh.n_colors(); ++i) {
				std::span<const MeshElem_t> quad_elems = mesh.get_color(i);

				GUTIL_OMP(parallel)
				{
					QuadRule_t 						quad_rule(mesh);
					std::vector<TestDof_t> 			v_dofs;
					std::vector<TrialDof_t>			u_dofs;
					std::vector<size_t>				v_global;
					std::vector<size_t>				u_global;
					
					std::vector<Scalar_t>			local_matrix;
					std::vector<Scalar_t>			local_y;
					std::vector<Scalar_t>			local_x;

					const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
					const size_t n_els_per_thread = quad_elems.size()/n_threads;
					const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
					const size_t start 			  = tid*n_els_per_thread;
					const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
					for (size_t j=start; j<end; ++j) {
						const MeshElem_t el = quad_elems[j];
						quad_rule.set_element(el,2);
						if constexpr (Kernel_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						v_dofs = test_handler.get_active_dofs_quasi_hierarchical(el);
						u_dofs = trial_handler.get_active_dofs_quasi_hierarchical(el);

						// GUTIL_LOG("q_el=", el, "dofs[0]=", v_dofs[0], " n_dofs=", v_dofs.size());

						const size_t v_size = v_dofs.size();
						const size_t u_size = u_dofs.size();

						//get global indices for the test and trial dofs
						//cache the x values needed locally
						local_y.assign(v_size, 0);
						local_x.assign(u_size, 0);
						local_matrix.assign(u_size*v_size, 0);

						v_global.assign(v_size, 0);
						u_global.assign(u_size, 0);
						for (size_t i=0; i<v_size; ++i) {
							v_global[i] = test_handler.global_number(v_dofs[i]);
							GUTIL_ASSERT(v_global[i] < test_handler.n_dofs());
							GUTIL_ASSERT(v_dofs[i].depth()<=el.depth()+1);
						}
						for (size_t j=0; j<u_size; ++j) {
							u_global[j] = trial_handler.global_number(u_dofs[j]);
							local_x[j]  = X[u_global[j]];
							GUTIL_ASSERT(u_global[j] < trial_handler.n_dofs());
							GUTIL_ASSERT(u_dofs[j].depth()<=el.depth()+1);
						}

						//construct local matrix (row major)
						// size_t idx_start = 0;
						// for (size_t i=0; i<v_size; ++i, idx_start+=u_size) {
						// 	for (size_t j=0; j<u_size; ++j) {
						// 		local_matrix[idx_start + j] = kernel(u_dofs[j],v_dofs[i],quad_rule);
						// 		GUTIL_ASSERT(kernel(u_dofs[j], v_dofs[i], quad_rule) > Scalar_t{0});
						// 	}
						// }

						//construct local matrix (col major)
						size_t idx_start = 0;
						for (size_t j=0; j<u_size; ++j, idx_start+=v_size) {
							for (size_t i=0; i<v_size; ++i) {
								local_matrix[idx_start + i] = kernel(u_dofs[j],v_dofs[i],quad_rule);
							}
						}

						//multiply local matrix
						// germ_mv(local_y.data(), v_size, local_x.data(), u_size, local_matrix.data());
						gecm_mv(local_y.data(), v_size, local_x.data(), u_size, local_matrix.data());

						//scatter local result
						for (size_t i=0; i<v_size; ++i) {
							Y[v_global[i]] += alpha*local_y[i];
						}
					}
				}
			}
		}

		void mat_vec_multiply_accumulate_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(X.size() == trial_handler.n_dofs());
			GUTIL_ASSERT(Y.size() == test_handler.n_dofs());
			GUTIL_ASSERT(mesh.is_color_sorted());
			GUTIL_ASSERT(&trial_handler == &test_handler);

			//note v/u dofs are the same
			for (int i=0; i<mesh.n_colors(); ++i) {
				std::span<const MeshElem_t> quad_elems = mesh.get_color(i);

				GUTIL_OMP(parallel)
				{
					QuadRule_t 						quad_rule(mesh);
					std::vector<TestDof_t> 			v_dofs;
					std::vector<size_t>				v_global;
					
					std::vector<Scalar_t>			local_matrix;
					std::vector<Scalar_t>			local_y;
					std::vector<Scalar_t>			local_x;

					const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
					const size_t n_els_per_thread = quad_elems.size()/n_threads;
					const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
					const size_t start 			  = tid*n_els_per_thread;
					const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
					for (size_t j=start; j<end; ++j) {
						const MeshElem_t el = quad_elems[j];
						quad_rule.set_element(el,2);
						if constexpr (Kernel_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						v_dofs = test_handler.get_active_dofs_quasi_hierarchical(el);
						
						const size_t v_size = v_dofs.size();

						//get global indices for the test and trial dofs
						//cache the x values needed locally
						local_y.assign(v_size, 0);
						local_x.assign(v_size, 0);
						local_matrix.assign(v_size*v_size, 0);

						v_global.assign(v_size, 0);
						for (size_t i=0; i<v_size; ++i) {
							v_global[i] = test_handler.global_number(v_dofs[i]);
							local_x[i]  = X[v_global[i]];
							GUTIL_ASSERT(v_global[i] < test_handler.n_dofs());
							GUTIL_ASSERT(v_dofs[i].depth()<=el.depth()+1);
						}

						//construct local matrix (row major)
						// size_t idx_start = 0;
						// for (size_t i=0; i<v_size; ++i, idx_start+=u_size) {
						// 	for (size_t j=0; j<u_size; ++j) {
						// 		local_matrix[idx_start + j] = kernel(u_dofs[j],v_dofs[i],quad_rule);
						// 		GUTIL_ASSERT(kernel(u_dofs[j], v_dofs[i], quad_rule) > Scalar_t{0});
						// 	}
						// }

						//construct local matrix (col major)
						for (size_t j=0; j<v_size; ++j) {
							local_matrix[j + v_size*j] = kernel(v_dofs[j], v_dofs[j],quad_rule);
							for (size_t i=j+1; i<v_size; ++i) {
								Scalar_t val = kernel(v_dofs[j],v_dofs[i],quad_rule);
								local_matrix[i + v_size*j] = val;
								local_matrix[j + v_size*i] = val;
							}
						}

						//multiply local matrix
						// germ_mv(local_y.data(), v_size, local_x.data(), u_size, local_matrix.data());
						gecm_mv(local_y.data(), v_size, local_x.data(), v_size, local_matrix.data());

						//scatter local result
						for (size_t i=0; i<v_size; ++i) {
							Y[v_global[i]] += alpha*local_y[i];
						}
					}
				}
			}
		}
	};
}