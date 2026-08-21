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
	/// When building a sparse matrix, it is usually most convenient to 
	/// build it from triplets. We make a custom triplet type to help
	/// de-duplicate entries before passing to e.g., Eigen.
	///
	/// Set StorageOrder to 0 for ColMajor and 1 for RowMajor.
	/// The sorting order is outer index then inner index (eg. col then row for ColMajor)
	/// This allows us to use Eigen's setFromSortedTriplets direcly from a vector
	/// of GV::Triplet.
	//////////////////////////////////////////////////////////////////
	template<typename Scalar, typename StorageIndex=size_t, int StorageOrder=0> requires (StorageOrder==0 || StorageOrder==1)
	struct Triplet {
		//add an api compatible with Eigen
		StorageIndex row() const {return i;}
		StorageIndex col() const {return j;}
		Scalar       value() const {return val;}

		StorageIndex i, j;
		Scalar val;

		constexpr Triplet(StorageIndex i, StorageIndex j, Scalar v) noexcept : i(i), j(j), val(v) {}
		constexpr Triplet() : i(0), j(0), val(0) {}
		static constexpr Triplet None() noexcept {return Triplet{StorageIndex(-1), StorageIndex(-1), Scalar{}};}

		bool operator<(const Triplet& other) const noexcept requires (StorageOrder==0) {
			if (j<other.j) {return true;}
			if (j==other.j && i < other.i) {return true;}
			return false;
		}

		bool operator<(const Triplet& other) const noexcept requires (StorageOrder==1) {
			if (i<other.i) {return true;}
			if (i==other.i && j < other.j) {return true;}
			return false;
		}

		bool operator==(const Triplet& other) const noexcept {
			return i==other.i && j==other.j;
		}

		static void Compress(std::vector<Triplet>& list) {
			//We assume that we are already in a multithreaded region
			if (list.empty()) {return;}

			std::sort(list.begin(), list.end());
			auto accumulate_into = list.begin();

			for (auto it=list.begin()+1; it!=list.end(); ++it) {
				if (*it == *accumulate_into) {
					accumulate_into->val += it->val;
					*it = None();
				}
				else {
					//keep the block to keep contiguous at the start
					//of the array
					++accumulate_into;
					*accumulate_into = *it;
				}
			}

			//remove the unneeded entries.
			++accumulate_into;
			list.erase(accumulate_into, list.end());
		}

		static std::vector<Triplet> Merge(std::vector<Triplet>& left, std::vector<Triplet>& right) {
			//take two vectors and merge the right into the left, then compress
			//for fastest results, compress left and right ahead of time
			left.insert(left.end(), std::make_move_iterator(right.begin()), std::make_move_iterator(right.end()));
			Compress(left);
			
			right.clear();
			right.shrink_to_fit();

			return left;
		}
	};


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
	template<int N, typename T, typename TrialHandlerType, typename TestHandlerType, typename KernelType, typename KernelWeightType=IdentityKernelWeight>
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

		static constexpr bool IS_SYMMETRIC = Kernel_t::IS_SYMMETRIC;
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
		const Weight_t 			weight;

		BilinearForm(const TrialHandler_t& u_handler, const TestHandler_t& v_handler, KernelType kernel = KernelType{}, KernelWeightType weight = KernelWeightType{}) :
			trial_handler(u_handler), test_handler(v_handler), mesh(test_handler.mesh), kernel(std::move(kernel)), weight(std::move(weight)) {}

		BilinearForm(const TrialHandler_t& sym_handler, KernelType kernel=KernelType{}, KernelWeightType weight = KernelWeightType{}) requires(Kernel_t::IS_SYMMETRIC) :
			trial_handler(sym_handler), test_handler(sym_handler), mesh(sym_handler.mesh), kernel(std::move(kernel)), weight(std::move(weight)) {}


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

					OmpIndexRange 					range(quad_elems.size());

					for (size_t q=range.begin; q<range.end; ++q) {
						const MeshElem_t el = quad_elems[q];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						trial_cache.gather_qh();	const size_t u_size = trial_cache.size();
						test_cache.gather_qh();		const size_t v_size = test_cache.size();
						if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
							wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
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

					OmpIndexRange 					range(quad_elems.size());
					for (size_t q=range.begin; q<range.end; ++q) {
						const MeshElem_t el = quad_elems[q];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
						if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
							wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
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
				QuadRule_t 				quad_rule(mesh);
				ElementTrialCache_t		trial_cache(trial_handler,quad_rule);
				ElementTestCache_t		test_cache(test_handler,quad_rule);
				KernelEval				k_eval(kernel, trial_cache, test_cache, quad_rule);
				WeightCache_t			wt;
				
				std::vector<Scalar_t>	local_matrix;
				std::vector<Scalar_t>	local_y;
				std::vector<Scalar_t>	local_x;

				OmpIteratorRange		range(mesh.element_begin(), mesh.element_end());
				for (auto it=range.begin; it!=range.end; ++it) {
					const MeshElem_t el = *it;
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					trial_cache.gather_qh();	const size_t u_size = trial_cache.size();
					test_cache.gather_qh();		const size_t v_size = test_cache.size();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
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

				OmpIteratorRange				range(mesh.element_begin(), mesh.element_end());
				for (auto it=range.begin; it!=range.end; ++it) {
					const MeshElem_t el = *it;
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
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
		/// Multiply and increment by the non-diagonal entries for jacobi preconditioning.
		/// Same as Y += alpha*(L+U)*X
		/////////////////////////////////////////////////////////////////////
		void mat_vec_multiply_accumulate_jacobi_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
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

					OmpIndexRange 					range(quad_elems.size());
					for (size_t q=range.begin; q<range.end; ++q) {
						const MeshElem_t el = quad_elems[q];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
						if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
							wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
						}

						local_y.assign(v_size, 0);
						local_x.assign(v_size, 0);
						local_matrix.assign(v_size*v_size, 0);

						for (size_t i=0; i<v_size; ++i) {local_x[i]  = X[sym_cache.global_idx[i]]; }

						//construct local matrix (col major)
						for (size_t j=0; j<v_size; ++j) {
							//ignore the diagonal entry
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

		void mat_vec_multiply_accumulate_jacobi(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
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

				OmpIteratorRange	range(mesh.element_begin(), mesh.element_end());
				for (auto it=range.begin; it!=range.end; ++it) {
					const MeshElem_t el = *it;
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					local_y.assign(v_size, 0);
					local_x.assign(v_size, 0);
					local_matrix.assign(v_size*v_size, 0);

					for (size_t i=0; i<v_size; ++i) {local_x[i]  = X[sym_cache.global_idx[i]]; }

					//construct local matrix (col major)
					for (size_t j=0; j<v_size; ++j) {
						//ignore the diagonal entry
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
					
					OmpIndexRange 					range(quad_elems.size());
					for (size_t q=range.begin; q<range.end; ++q) {
						const MeshElem_t el = quad_elems[q];
						
						quad_rule.set_element(el,2);
						if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
						
						sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
						if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
							wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
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
				
				OmpIteratorRange				range(mesh.element_begin(), mesh.element_end());
				for (auto it=range.begin; it!=range.end; ++it) {
					const MeshElem_t el = *it;

					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
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
			std::vector<std::vector<Triplet_t>> 	all_thread_triplets;

			GUTIL_OMP(parallel)
			{
				QuadRule_t 							quad_rule(mesh);
				ElementTrialCache_t					trial_cache(trial_handler,quad_rule);
				ElementTestCache_t					test_cache(test_handler,quad_rule);
				KernelEval							k_eval(kernel, trial_cache, test_cache, quad_rule);
				WeightCache_t						wt;
				
				OmpIteratorRange					range(mesh.element_begin(), mesh.element_end());

				GUTIL_OMP(single)
				{
					all_thread_triplets.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)

				std::vector<Triplet_t>&	thread_triplet = all_thread_triplets[range.tid];
				thread_triplet.reserve(range.count * TrialDof_t::N_DOF_PER_ELEM * TestDof_t::N_DOF_PER_ELEM);
				
				for (auto it=range.begin; it!=range.end; ++it) {
					const MeshElem_t el = *it;
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					trial_cache.gather_qh();	const size_t u_size = trial_cache.size();
					test_cache.gather_qh();		const size_t v_size = test_cache.size();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					//add triplets
					for (size_t j=0; j<u_size; ++j) {
						for (size_t i=0; i<v_size; ++i) {
							thread_triplet.emplace_back(
								row_offset + test_cache.global_idx[i],
								col_offset + trial_cache.global_idx[j],
								k_eval(j,i,&wt));
						}
					}
				}

				//de-duplicate triplets and convert to the desired format (e.g, Eigen::Triplet<T>)
				Triplet_t::Compress(thread_triplet);
				thread_triplet.shrink_to_fit();

				//join the results from each thread
				GUTIL_OMP(atomic) triplet_size += thread_triplet.size();
				GUTIL_OMP(barrier)


				for (size_t stride=1; stride<range.n_threads; stride*=2) {
					if (range.tid % (2*stride) == 0 && range.tid+stride < range.n_threads) {
						Triplet_t::Merge(all_thread_triplets[range.tid], all_thread_triplets[range.tid+stride]);
					}
					GUTIL_OMP(barrier)
				}


				GUTIL_OMP(single)
				{
					triplets.reserve(triplets.size() + triplet_size);
				}
			}

			triplets.insert(triplets.end(),
				std::make_move_iterator(all_thread_triplets[0].begin()),
				std::make_move_iterator(all_thread_triplets[0].end()));
		}

		template<typename Triplet_t>
		void build_triplets(std::vector<Triplet_t>& triplets, const size_t row_offset=0, const size_t col_offset=0) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			GUTIL_ASSERT(&trial_handler == &test_handler);

			size_t triplet_size{0};
			std::vector<std::vector<Triplet_t>> 	all_thread_triplets;

			GUTIL_OMP(parallel)
			{
				QuadRule_t 							quad_rule(mesh);
				ElementTestCache_t					sym_cache(test_handler,quad_rule);
				KernelEval							k_eval(kernel, sym_cache, sym_cache, quad_rule);
				WeightCache_t						wt;
				
				OmpIteratorRange					range(mesh.element_begin(), mesh.element_end());

				GUTIL_OMP(single)
				{
					all_thread_triplets.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)

				std::vector<Triplet_t>&	thread_triplet = all_thread_triplets[range.tid];
				thread_triplet.reserve(range.count * TrialDof_t::N_DOF_PER_ELEM * TestDof_t::N_DOF_PER_ELEM);
				
				for (auto it=range.begin; it!=range.end; ++it) {
					const MeshElem_t el = *it;
						
					quad_rule.set_element(el,2);
					if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
					
					sym_cache.gather_qh();		const size_t v_size = sym_cache.size();
					if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
						wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
					}

					//add triplets
					for (size_t j=0; j<v_size; ++j) {
						thread_triplet.emplace_back(
							row_offset + sym_cache.global_idx[j],
							col_offset + sym_cache.global_idx[j],
							k_eval(j,j,&wt));
						
						for (size_t i=j+1; i<v_size; ++i) {
							const Scalar_t val = k_eval(j,i,&wt);
							thread_triplet.emplace_back(
								row_offset + sym_cache.global_idx[i],
								col_offset + sym_cache.global_idx[j],
								val);
							thread_triplet.emplace_back(
								row_offset + sym_cache.global_idx[j],
								col_offset + sym_cache.global_idx[i],
								val);
						}
					}
				}

				//de-duplicate triplets and convert to the desired format (e.g, Eigen::Triplet<T>)
				Triplet_t::Compress(thread_triplet);
				thread_triplet.shrink_to_fit();

				//join the results from each thread
				GUTIL_OMP(atomic) triplet_size += thread_triplet.size();
				GUTIL_OMP(barrier)

				for (size_t stride=1; stride<range.n_threads; stride*=2) {
					if (range.tid % (2*stride) == 0 && range.tid+stride < range.n_threads) {
						Triplet_t::Merge(all_thread_triplets[range.tid], all_thread_triplets[range.tid+stride]);
					}
					GUTIL_OMP(barrier)
				}


				GUTIL_OMP(single)
				{
					triplets.reserve(triplets.size() + triplet_size);
				}
			}

			triplets.insert(triplets.end(),
				std::make_move_iterator(all_thread_triplets[0].begin()),
				std::make_move_iterator(all_thread_triplets[0].end()));
		}
	};
}


