#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"

#include "fem/dofhandler.hpp"

#include "fem/mesh_quadrature.hpp"
#include "fem/forms/bilinearforms/bilinear_kernels.hpp"
#include "fem/forms/bilinearforms/dense_linalg.hpp"

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
	template<int N, typename T, typename TrialHandlerType, typename TestHandlerType, typename KernelType, typename KernelWeightType=NoKernelWeight>
	struct BilinearForm {


		//////////////////////////////////////////////////////////////////
		/// Aliases and sanity checks
		//////////////////////////////////////////////////////////////////
		using Kernel_t      	= KernelType;
		using Weight_t  		= KernelWeightType;
		using TrialHandler_t	= TrialHandlerType;
		using TestHandler_t		= TestHandlerType;
		using QuadRule_t 		= MeshQuadratureRule<N,T>;
		using Scalar_t   		= T;
		using Mesh_t     		= UnstructuredVoxelMesh<T>;
		using MeshElem_t 		= typename Mesh_t::Elem_t;
		using TrialDof_t  		= typename TrialHandlerType::DOF_t;
		using TestDof_t  		= typename TestHandlerType::DOF_t;

		using ElementTrialCache_t = ElementDofCache<Kernel_t,TrialHandler_t,QuadRule_t>;
		using ElementTestCache_t = ElementDofCache<Kernel_t,TestHandler_t,QuadRule_t>;
		using WeightCache_t      = ScalarValueCache<QuadRule_t>;

		static_assert(!Kernel_t::IS_SYMMETRIC || std::same_as<TrialHandler_t,TestHandler_t>,
			"A symmetric kernel must have the same dofhandlers for the test and trial spaces");


		struct KernelEval {
			const Kernel_t& 			kernel;
			const ElementTrialCache_t& 	trial_cache;
			const ElementTestCache_t& 	test_cache;
			const QuadRule_t& 			qr;

			KernelEval(const Kernel_t& k, const ElementTrialCache_t& trial, const ElementTestCache_t& test, const QuadRule_t& q) : 
					kernel(k), trial_cache(trial), test_cache(test), qr(q) {}

			[[nodiscard]] Scalar_t operator()(size_t i, size_t j, const WeightCache_t* wt = nullptr) const noexcept {
				const DofValueCache<QuadRule_t>* trial_vals{nullptr};
				const DofGradCache<QuadRule_t>*  trial_grad{nullptr};
				const DofValueCache<QuadRule_t>* test_vals{nullptr};
				const DofGradCache<QuadRule_t>*  test_grad{nullptr};
				
				if constexpr (Kernel_t::NEEDS_DOF_VALS) {
					trial_vals = &trial_cache.vals[i];
					test_vals  = &test_cache.vals[j];
				}
				if constexpr (Kernel_t::NEEDS_DOF_GRAD) {
					trial_grad = &trial_cache.grad[i];
					test_grad  = &test_cache.grad[j];
				}
				return kernel.cached_eval(trial_vals, trial_grad, test_vals, test_grad, wt, qr);
			}
		};



		//////////////////////////////////////////////////////////////////
		/// Data and constructor
		//////////////////////////////////////////////////////////////////
		const TrialHandler_t&	trial_handler;
		const TestHandler_t&	test_handler;
		const Mesh_t&	 		mesh;
		const Kernel_t 			kernel;

		BilinearForm(const TrialHandler_t& u_handler, const TestHandler_t& v_handler, KernelType kernel = KernelType{}) :
			trial_handler(u_handler), test_handler(v_handler), mesh(test_handler.mesh), kernel(std::move(kernel)) {}

		BilinearForm(const TrialHandler_t& sym_handler, KernelType kernel=KernelType{}) requires(Kernel_t::IS_SYMMETRIC) :
			trial_handler(sym_handler), test_handler(sym_handler), mesh(sym_handler.mesh), kernel(std::move(kernel)) {}


		//////////////////////////////////////////////////////////////////
		/// A few convenience methods
		//////////////////////////////////////////////////////////////////
		[[nodiscard]] size_t n_rows() const noexcept {return test_handler.n_dofs();}
		[[nodiscard]] size_t n_cols() const noexcept {return trial_handler.n_dofs();}


		//////////////////////////////////////////////////////////////////
		/// Given trial coefficients X, compute Y += alpha*A*X
		//////////////////////////////////////////////////////////////////
		void mat_vec_multiply_accumulate_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(!Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(X.size() == trial_handler.n_dofs());
			GUTIL_ASSERT(Y.size() == test_handler.n_dofs());
			GUTIL_ASSERT(mesh.is_color_sorted());

			for (int clr=0; clr<mesh.n_colors(); ++clr) {
				std::span<const MeshElem_t> quad_elems = mesh.get_color(clr);

				GUTIL_OMP(parallel)
				{
					QuadRule_t 						quad_rule(mesh);
					ElementTrialCache_t				trial_cache(trial_handler,quad_rule);
					ElementTestCache_t				test_cache(test_handler,quad_rule);
					KernelEval						k_eval(kernel, trial_cache, test_cache, quad_rule);
					WeightCache_t					wt;

					std::vector<Scalar_t>			local_matrix;
					std::vector<Scalar_t>			local_y;
					std::vector<Scalar_t>			local_x;

					const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
					const size_t n_els_per_thread = quad_elems.size()/n_threads;
					const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
					const size_t start 			  = tid*n_els_per_thread;
					const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
					for (size_t q=start; q<end; ++q) {
						const MeshElem_t el = quad_elems[q];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						trial_cache.gather_qh();	const size_t u_size = trial_cache.size();
						test_cache.gather_qh();		const size_t v_size = test_cache.size();
						if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
							wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
						}

						local_y.assign(v_size, 0);
						local_x.assign(u_size, 0);
						local_matrix.assign(u_size*v_size, 0);

						for (size_t j=0; j<u_size; ++j) {local_x[j]  = X[trial_cache.global_idx[j]];}

						//construct local matrix (col major)
						size_t idx_start = 0;
						for (size_t j=0; j<u_size; ++j, idx_start+=v_size) {
							for (size_t i=0; i<v_size; ++i) {
								local_matrix[idx_start + i] = k_eval(j,i,&wt);
							}
						}

						//multiply local matrix
						gecm_mv(local_y.data(), v_size, local_x.data(), u_size, local_matrix.data());

						//scatter local result
						for (size_t i=0; i<v_size; ++i) {
							Y[trial_cache.global_idx[i]] += alpha*local_y[i];
						}
					}
				}
			}
		}

		void mat_vec_multiply_accumulate_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(X.size() == trial_handler.n_dofs());
			GUTIL_ASSERT(Y.size() == test_handler.n_dofs());
			GUTIL_ASSERT(mesh.is_color_sorted());
			GUTIL_ASSERT(&trial_handler == &test_handler);

			//note v/u dofs are the same
			for (int clr=0; clr<mesh.n_colors(); ++clr) {
				std::span<const MeshElem_t> quad_elems = mesh.get_color(clr);

				GUTIL_OMP(parallel)
				{
					QuadRule_t 						quad_rule(mesh);
					ElementTestCache_t				sym_cache(test_handler,quad_rule);
					KernelEval						k_eval(kernel, sym_cache, sym_cache, quad_rule);
					WeightCache_t					wt;
					
					std::vector<Scalar_t>			local_matrix;
					std::vector<Scalar_t>			local_y;
					std::vector<Scalar_t>			local_x;

					const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
					const size_t n_els_per_thread = quad_elems.size()/n_threads;
					const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
					const size_t start 			  = tid*n_els_per_thread;
					const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
					for (size_t q=start; q<end; ++q) {
						const MeshElem_t el = quad_elems[q];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
						if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
							wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
						}

						local_y.assign(v_size, 0);
						local_x.assign(v_size, 0);
						local_matrix.assign(v_size*v_size, 0);

						for (size_t i=0; i<v_size; ++i) {local_x[i]  = X[sym_cache.global_idx[i]]; }

						//construct local matrix (col major)
						for (size_t j=0; j<v_size; ++j) {
							local_matrix[j + v_size*j] = k_eval(j,j,&wt);
							for (size_t i=j+1; i<v_size; ++i) {
								Scalar_t val = k_eval(j,i,&wt);
								local_matrix[i + v_size*j] = val;
								local_matrix[j + v_size*i] = val;
							}
						}

						//multiply local matrix
						gecm_mv(local_y.data(), v_size, local_x.data(), v_size, local_matrix.data());

						//scatter local result
						for (size_t i=0; i<v_size; ++i) {
							Y[sym_cache.global_idx[i]] += alpha*local_y[i];
						}
					}
				}
			}
		}

		void mat_vec_multiply_accumulate(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(!Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(X.size() == trial_handler.n_dofs());
			GUTIL_ASSERT(Y.size() == test_handler.n_dofs());
			
			GUTIL_OMP(parallel)
			{
				QuadRule_t 						quad_rule(mesh);
				ElementTrialCache_t				trial_cache(trial_handler,quad_rule);
				ElementTestCache_t				test_cache(test_handler,quad_rule);
				KernelEval						k_eval(kernel, trial_cache, test_cache, quad_rule);
				WeightCache_t					wt;
				
				std::vector<Scalar_t>			local_matrix;
				std::vector<Scalar_t>			local_y;
				std::vector<Scalar_t>			local_x;

				const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
				const size_t n_els_per_thread = mesh.n_elements()/n_threads;
				const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
				const size_t start 			  = tid*n_els_per_thread;
				const size_t end 			  = (tid==n_threads-1) ? mesh.n_elements() : start + n_els_per_thread;
				
				std::span<const MeshElem_t> quad_elems(mesh.element_begin()+start, mesh.element_begin()+end);

				for (size_t q=0; q<quad_elems.size(); ++q) {
					const MeshElem_t el = quad_elems[q];
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					trial_cache.gather_qh();	const size_t u_size = trial_cache.size();
					test_cache.gather_qh();		const size_t v_size = test_cache.size();
					if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
						wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					local_y.assign(v_size, 0);
					local_x.assign(u_size, 0);
					local_matrix.assign(u_size*v_size, 0);

					for (size_t j=0; j<u_size; ++j) {local_x[j]  = X[trial_cache.global_idx[j]];}

					//construct local matrix (col major)
					size_t idx_start = 0;
					for (size_t j=0; j<u_size; ++j, idx_start+=v_size) {
						for (size_t i=0; i<v_size; ++i) {
							local_matrix[idx_start + i] = k_eval(j,i,&wt);
						}
					}

					//multiply local matrix
					gecm_mv(local_y.data(), v_size, local_x.data(), u_size, local_matrix.data());

					//scatter local result
					for (size_t i=0; i<v_size; ++i) {
						GUTIL_OMP(atomic) Y[trial_cache.global_idx[i]] += alpha*local_y[i];
					}
				}
			}
		}

		void mat_vec_multiply_accumulate(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(X.size() == trial_handler.n_dofs());
			GUTIL_ASSERT(Y.size() == test_handler.n_dofs());
			GUTIL_ASSERT(&trial_handler == &test_handler);

			GUTIL_OMP(parallel)
			{
				QuadRule_t 						quad_rule(mesh);
				ElementTestCache_t				sym_cache(test_handler,quad_rule);
				KernelEval						k_eval(kernel, sym_cache, sym_cache, quad_rule);
				WeightCache_t					wt;
				
				std::vector<Scalar_t>			local_matrix;
				std::vector<Scalar_t>			local_y;
				std::vector<Scalar_t>			local_x;

				const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
				const size_t n_els_per_thread = mesh.n_elements()/n_threads;
				const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
				const size_t start 			  = tid*n_els_per_thread;
				const size_t end 			  = (tid==n_threads-1) ? mesh.n_elements() : start + n_els_per_thread;
				
				std::span<const MeshElem_t> quad_elems(mesh.element_begin()+start, mesh.element_begin()+end);

				for (size_t q=0; q<quad_elems.size(); ++q) {
					const MeshElem_t el = quad_elems[q];
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
					if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
						wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					local_y.assign(v_size, 0);
					local_x.assign(v_size, 0);
					local_matrix.assign(v_size*v_size, 0);

					for (size_t i=0; i<v_size; ++i) {local_x[i]  = X[sym_cache.global_idx[i]]; }

					//construct local matrix (col major)
					for (size_t j=0; j<v_size; ++j) {
						local_matrix[j + v_size*j] = k_eval(j,j,&wt);
						for (size_t i=j+1; i<v_size; ++i) {
							Scalar_t val = k_eval(j,i,&wt);
							local_matrix[i + v_size*j] = val;
							local_matrix[j + v_size*i] = val;
						}
					}

					//multiply local matrix
					gecm_mv(local_y.data(), v_size, local_x.data(), v_size, local_matrix.data());

					//scatter local result
					for (size_t i=0; i<v_size; ++i) {
						GUTIL_OMP(atomic) Y[sym_cache.global_idx[i]] += alpha*local_y[i];
					}
				}
			}
		}

		/////////////////////////////////////////////////////////////////////
		/// Construct diagonal of the matrix for conditioning
		/////////////////////////////////////////////////////////////////////
		void construct_diagonal_colored(std::span<T> D) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(mesh.is_color_sorted());
			GUTIL_ASSERT(&trial_handler == &test_handler);

			//note v/u dofs are the same
			for (int clr=0; clr<mesh.n_colors(); ++clr) {
				std::span<const MeshElem_t> quad_elems = mesh.get_color(clr);

				GUTIL_OMP(parallel)
				{
					QuadRule_t 						quad_rule(mesh);
					ElementTestCache_t				sym_cache(test_handler,quad_rule);
					KernelEval						k_eval(kernel, sym_cache, sym_cache, quad_rule);
					WeightCache_t					wt;
					
					const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
					const size_t n_els_per_thread = quad_elems.size()/n_threads;
					const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
					const size_t start 			  = tid*n_els_per_thread;
					const size_t end 			  = (tid==n_threads-1) ? quad_elems.size() : start + n_els_per_thread;
					
					for (size_t q=start; q<end; ++q) {
						const MeshElem_t el = quad_elems[q];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
						if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
							wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
						}

						//scatter local result
						for (size_t i=0; i<v_size; ++i) {
							D[sym_cache.global_idx[i]] += k_eval(i,i,&wt);
						}
					}
				}
			}
		}

		void construct_diagonal(std::span<T> D) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(&trial_handler == &test_handler);

			GUTIL_OMP(parallel)
			{
				QuadRule_t 						quad_rule(mesh);
				ElementTestCache_t				sym_cache(test_handler,quad_rule);
				KernelEval						k_eval(kernel, sym_cache, sym_cache, quad_rule);
				WeightCache_t					wt;
				
				const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
				const size_t n_els_per_thread = mesh.n_elements()/n_threads;
				const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
				const size_t start 			  = tid*n_els_per_thread;
				const size_t end 			  = (tid==n_threads-1) ? mesh.n_elements() : start + n_els_per_thread;
				
				std::span<const MeshElem_t> quad_elems(mesh.element_begin()+start, mesh.element_begin()+end);

				for (size_t q=0; q<quad_elems.size(); ++q) {
					const MeshElem_t el = quad_elems[q];
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
					if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
						wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					//scatter local result
					for (size_t i=0; i<v_size; ++i) {
						GUTIL_OMP(atomic) D[sym_cache.global_idx[i]] += k_eval(i,i,&wt);
					}
				}
			}
		}


		///////////////////////////////////////////////////////////////
		/// Construct triples for sparese matrix construction
		///////////////////////////////////////////////////////////////
		template<typename Triplet_t>
		void build_triplets(std::vector<Triplet_t>& triplets, const size_t row_offset=0, const size_t col_offset=0) const noexcept requires(!Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			
			size_t triplet_size{0};

			GUTIL_OMP(parallel)
			{
				QuadRule_t 						quad_rule(mesh);
				ElementTrialCache_t				trial_cache(trial_handler,quad_rule);
				ElementTestCache_t				test_cache(test_handler,quad_rule);
				KernelEval						k_eval(kernel, trial_cache, test_cache, quad_rule);
				WeightCache_t					wt;
				
				std::vector<Triplet_t>			thread_triplets;

				const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
				const size_t n_els_per_thread = mesh.n_elements()/n_threads;
				const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
				const size_t start 			  = tid*n_els_per_thread;
				const size_t end 			  = (tid==n_threads-1) ? mesh.n_elements() : start + n_els_per_thread;
				
				std::span<const MeshElem_t> quad_elems(mesh.element_begin()+start, mesh.element_begin()+end);
				thread_triplets.reserve(quad_elems.size() * TrialDof_t::N_DOF_PER_ELEM * TestDof_t::N_DOF_PER_ELEM);


				for (size_t q=0; q<quad_elems.size(); ++q) {
					const MeshElem_t el = quad_elems[q];
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					trial_cache.gather_qh();	const size_t u_size = trial_cache.size();
					test_cache.gather_qh();		const size_t v_size = test_cache.size();
					if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
						wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					//add triplets
					for (size_t j=0; j<u_size; ++j) {
						for (size_t i=0; i<v_size; ++i) {
							thread_triplets.emplace_back(
								row_offset + test_cache.global_idx[i],
								col_offset + trial_cache.global_idx[j],
								k_eval(j,i,&wt));
						}
					}
				}

				GUTIL_OMP(atomic) triplet_size += thread_triplets.size();
				GUTIL_OMP(barrier)

				GUTIL_OMP(single)
				{
					triplets.reserve(triplets.size() + triplet_size);
				}

				GUTIL_OMP(critical)
				{
					triplets.insert(triplets.end(),
						std::make_move_iterator(thread_triplets.begin()),
						std::make_move_iterator(thread_triplets.end()));
				}
			}
		}

		template<typename Triplet_t>
		void build_triplets(std::vector<Triplet_t>& triplets, const size_t row_offset=0, const size_t col_offset=0) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(&trial_handler == &test_handler);

			size_t triplet_size{0};

			GUTIL_OMP(parallel)
			{
				QuadRule_t 						quad_rule(mesh);
				ElementTestCache_t				sym_cache(test_handler,quad_rule);
				KernelEval						k_eval(kernel, sym_cache, sym_cache, quad_rule);
				WeightCache_t					wt;
				
				std::vector<Triplet_t>			thread_triplets;

				const size_t n_threads 		  = GUTIL_OMP_TERNARY(omp_get_num_threads(), 1);
				const size_t n_els_per_thread = mesh.n_elements()/n_threads;
				const size_t tid 			  = GUTIL_OMP_TERNARY(omp_get_thread_num(), 0);
				const size_t start 			  = tid*n_els_per_thread;
				const size_t end 			  = (tid==n_threads-1) ? mesh.n_elements() : start + n_els_per_thread;
				
				std::span<const MeshElem_t> quad_elems(mesh.element_begin()+start, mesh.element_begin()+end);
				thread_triplets.reserve(quad_elems.size() * TrialDof_t::N_DOF_PER_ELEM * TestDof_t::N_DOF_PER_ELEM);


				for (size_t q=0; q<quad_elems.size(); ++q) {
					const MeshElem_t el = quad_elems[q];
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
					if constexpr (!std::same_as<Weight_t,NoKernelWeight>) {
						wt = Weight_t::template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					//add triplets
					for (size_t j=0; j<v_size; ++j) {
						thread_triplets.emplace_back(
							row_offset + sym_cache.global_idx[j],
							col_offset + sym_cache.global_idx[j],
							k_eval(j,j,&wt));
						
						for (size_t i=j+1; i<v_size; ++i) {
							const Scalar_t val = k_eval(j,i,&wt);
							thread_triplets.emplace_back(
								row_offset + sym_cache.global_idx[i],
								col_offset + sym_cache.global_idx[j],
								val);
							thread_triplets.emplace_back(
								row_offset + sym_cache.global_idx[j],
								col_offset + sym_cache.global_idx[i],
								val);
						}
					}
				}

				GUTIL_OMP(atomic) triplet_size += thread_triplets.size();
				GUTIL_OMP(barrier)

				GUTIL_OMP(single)
				{
					triplets.reserve(triplets.size() + triplet_size);
				}

				GUTIL_OMP(critical)
				{
					triplets.insert(triplets.end(),
						std::make_move_iterator(thread_triplets.begin()),
						std::make_move_iterator(thread_triplets.end()));
				}
			}
		}
	};
}