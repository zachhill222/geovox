#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"

#include "fem/mesh_quadrature.hpp"
#include "fem/forms/util.hpp"
#include "fem/forms/bilinearforms/triplet.hpp"
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

		using ElementTrialCache_t = ElementTrialCache<Kernel_t,TrialHandler_t,QuadRule_t>;
		using ElementTestCache_t  = std::conditional_t<Kernel_t::IS_SYMMETRIC, ElementTrialCache_t, ElementTestCache<Kernel_t,TestHandler_t,QuadRule_t>>;
		using WeightCache_t       = ScalarValueCache<QuadRule_t>;

		static constexpr bool IS_SYMMETRIC = Kernel_t::IS_SYMMETRIC;
		static_assert(!Kernel_t::IS_SYMMETRIC || std::same_as<TrialHandler_t,TestHandler_t>,
			"A symmetric kernel must have the same dofhandlers for the test and trial spaces");


		//////////////////////////////////////////////////////////////////
		/// Helper types
		//////////////////////////////////////////////////////////////////
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
				
				if constexpr (Kernel_t::TRIAL_DOF_VALS) {
					trial_vals = &trial_cache.vals[i];
				}
				if constexpr (Kernel_t::TEST_DOF_VALS) {
					test_vals  = &test_cache.vals[j];
				}
				if constexpr (Kernel_t::TRIAL_DOF_GRAD) {
					trial_grad = &trial_cache.grad[i];
				}
				if constexpr (Kernel_t::TEST_DOF_GRAD) {
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
		/// A few methods for handling the looping over elements in parallel
		/// either as colored or non-colored. Note that action should have the signature
		/// void(KernelEval, TrialCache_t, TestCache_t, WeightCache_t*) for both symmetric
		/// and non-symmetric problems (for consistency). Optionally, the range may be passed
		/// to the action as well. 
		/// For symmetric kernels, the correct symmetric loop is chosen at compile time.
		///
		/// Extra arguments may be forwarded to the action as well, but will need a custom for_each_element function.
		///
		/// Note that OmpIteratorRange contains:
		///		begin     - iterator to the beginning of the current thread data
		///		end       - iterator to the end of the current thread data
		///     count     - the number of current data in the current thread (same as std::distance(begin,end))
		///		tid       - the number of the current thread in the OpenMP thread group/pool
		/// 	n_threads - the total number of threads in the current OpenMP thread group/pool
		///
		/// Note that raw pointers satisfy the std::random_access_iterator concept.
		//////////////////////////////////////////////////////////////////
		template<typename Action, std::random_access_iterator I, typename... Args> requires(std::same_as<std::iter_value_t<I>,MeshElem_t>)
		void apply_action_over_range(const gutil::OmpIteratorRange<I>& range, Action&& action, Args&&... args) const noexcept requires (!Kernel_t::IS_SYMMETRIC){
			GUTIL_PROFILE_FUNCTION();
			
			QuadRule_t				quad_rule(mesh);
			ElementTrialCache_t 	trial_cache(trial_handler, quad_rule);
			ElementTestCache_t		test_cache(test_handler, quad_rule);
			KernelEval  			k_eval(kernel, trial_cache, test_cache, quad_rule);
			WeightCache_t  			wt;
			const auto 				quad_project_depth = std::max(test_handler.max_depth_distance, trial_handler.max_depth_distance);

			for (auto it=range.begin; it!=range.end; ++it) {
				quad_rule.set_element(*it, quad_project_depth);
				if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
				trial_cache.gather();
				test_cache.gather();
				if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
					//TODO: add a method to evaluate/cache the scalar field
					wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
				}

				//after everything is cached on this element, do the action
				constexpr bool ACTION_NEEDS_RANGE = std::is_invocable_r_v<void, Action&, KernelEval&,
						ElementTrialCache_t&, ElementTestCache_t&, WeightCache_t*, gutil::OmpIteratorRange<I>&>;
				if constexpr (ACTION_NEEDS_RANGE) {
					action(k_eval, trial_cache, test_cache, &wt, range, std::forward<Args>(args)...);
				}
				else {
					action(k_eval, trial_cache, test_cache, &wt, std::forward<Args>(args)...);
				}
			}
		}

		template<typename Action, std::random_access_iterator I, typename... Args> requires(std::same_as<std::iter_value_t<I>,MeshElem_t>)
		void apply_action_over_range(const gutil::OmpIteratorRange<I>& range, Action&& action, Args&&... args) const noexcept requires (Kernel_t::IS_SYMMETRIC){
			GUTIL_PROFILE_FUNCTION();

			QuadRule_t 				quad_rule(mesh);
			ElementTestCache_t		sym_cache(test_handler,quad_rule);
			KernelEval				k_eval(kernel, sym_cache, sym_cache, quad_rule);
			WeightCache_t			wt;
			const auto 				quad_project_depth = std::max(test_handler.max_depth_distance, trial_handler.max_depth_distance);

			constexpr bool ACTION_NEEDS_RANGE = std::is_invocable_r_v<void, Action&, KernelEval&,
				ElementTrialCache_t&, ElementTestCache_t&, WeightCache_t*, gutil::OmpIteratorRange<I>&,
				Args&&...>;

			for (auto it=range.begin; it!=range.end; ++it) {
				const MeshElem_t el = *it;
				quad_rule.set_element(el, quad_project_depth);
				if constexpr (Weight_t::NEEDS_GEO_POINTS) {quad_rule.build_geometric_coords();}
				sym_cache.gather();
				if constexpr (!std::same_as<Weight_t,IdentityKernelWeight>) {
					//TODO: add a method to evaluate/cache the scalar field
					wt = weight.template build_weights<QuadRule_t>(nullptr, quad_rule);
				}

				//after everything is cached on this element, do the action
				if constexpr (ACTION_NEEDS_RANGE) {
					action(k_eval, sym_cache, sym_cache, &wt, range, std::forward<Args>(args)...);
				}
				else {
					action(k_eval, sym_cache, sym_cache, &wt, std::forward<Args>(args)...);
				}
			}
		}


		template<bool Colored=false, typename Action, typename Init=std::nullptr_t, typename Finalize=std::nullptr_t>
		void for_each_element(Action&& action, Init&& init=nullptr, Finalize&& finalize=nullptr) const noexcept {
			GUTIL_ASSERT(trial_handler.is_current());
			GUTIL_ASSERT(test_handler.is_current());
			if constexpr (Colored) {GUTIL_ASSERT(mesh.is_color_sorted());}
			if constexpr (Kernel_t::IS_SYMMETRIC) {GUTIL_ASSERT(&test_handler == &trial_handler);}

			//use init and finalize to manage extra thread resources.
			//for example, init may return a vector that gets passed to action for each thread to accumulate into
			//then finalize is responsible for merging the results of each thread.
			//note that init and finalize are called in the parallel block before/after the action is applied and the
			//same OmpIteratorRange (see above) that is passed to action is passed to these methods.
			//To handle per-thread resources using this we could do somthing like:
			//
			// main() {
			// std::vector<Data_t> thread_data;
			// auto init = [&thread_data](const auto& range) {
			//	GUTIL_OMP(single)
			//  {
			//     thread_data.resize(range.n_threads);
			//  }
			//  GUTIL_OMP(barrier)
			// };
			//
			// auto finalize = [&thread_data](const auto& range) {
			//	  GUTIL_OMP(critical/barrier/ect..)
			//    {
			//      do_somthing(thread_data[range.tid]);
			//    }
			// };
			// ....}

			//dispatch the loop using either colored or non-colored element loop
			if constexpr (Colored) {
				for (int clr=0; clr<mesh.n_colors(); ++clr) {
					std::span<const MeshElem_t> quad_elems = mesh.get_color(clr);
					GUTIL_OMP(parallel)
					{
						//note that action requires less/no synchronization here
						//dof-agnostic mesh coloring will likely not work for True Hierarchical dof_hadlers
						const gutil::OmpIteratorRange range{quad_elems.begin(), quad_elems.end()};
						if constexpr (!std::same_as<Init,std::nullptr_t>) {init(range);}
						apply_action_over_range(range, action);
						if constexpr (!std::same_as<Finalize,std::nullptr_t>) {finalize(range);}
					}
				}
			}
			else {
				GUTIL_OMP(parallel)
				{
					//note that action requires more synchronization here
					const gutil::OmpIteratorRange range{mesh.element_begin(), mesh.element_end()};
					if constexpr (!std::same_as<Init,std::nullptr_t>) {init(range);}
					apply_action_over_range(range, action);
					if constexpr (!std::same_as<Finalize,std::nullptr_t>) {finalize(range);}
				}
			}
		}


		//////////////////////////////////////////////////////////////////
		/// A few building blocks for the actions
		//////////////////////////////////////////////////////////////////
		template<typename Cache_t>
		static void GatherLocalVector(std::span<const Scalar_t> X, const Cache_t& dof_cache, std::vector<Scalar_t>& local_x) noexcept {
			//dof_cache has the dof values and global numbers for test/trial dofs whose support overlaps the current element
			//for allowing simd operations and better memory caching, it is often best to copy/gather the (spread out)
			//X-values into contiguous values.
			const size_t n = dof_cache.size();
			local_x.assign(n,Scalar_t{0});
			for (size_t i=0; i<n; ++i) {local_x[i] = X[dof_cache.global_idx[i]];}
		}

		template<bool Atomic, typename Cache_t>
		static void ScatterLocalVector(std::span<Scalar_t> Y, const Cache_t& dof_cache, std::span<const Scalar_t> local_y, Scalar_t alpha) noexcept {
			//increment Y by alpha*y_local with the local to global index conversion supplied by the cache.
			GUTIL_ASSERT(local_y.size()==dof_cache.size());
			const size_t n = dof_cache.size();
			if constexpr (Atomic) {
				for (size_t i=0; i<n; ++i) { GUTIL_OMP(atomic) Y[dof_cache.global_idx[i]] += alpha*local_y[i];}
			}
			else {
				for (size_t i=0; i<n; ++i) {Y[dof_cache.global_idx[i]] += alpha*local_y[i];}
			}
		}

		template<int LDU_FLag = 0b111, typename KernelEval_t, typename WeightCache_t>
		static void ConstructLocalMatrix(const KernelEval_t& k_eval, size_t u_size, size_t v_size, 
			const WeightCache_t* wt_ptr, std::vector<Scalar_t>& local_mat) noexcept requires (!Kernel_t::IS_SYMMETRIC) {
			//assemble the local matrix in column major format
			local_mat.resize(u_size*v_size);
			size_t idx = 0; //index (i,j) in column major
			for (size_t j=0; j<u_size; ++j) {
				for (size_t i=0; i<v_size; ++i) {
					local_mat[idx++] = k_eval(j,i,wt_ptr);
				}
			}
		}

		template<int LDU_Flag = 0b111, typename KernelEval_t, typename WeightCache_t>
		static void ConstructLocalMatrix(const KernelEval_t& k_eval, size_t u_size, size_t v_size, 
			const WeightCache_t* wt_ptr, std::vector<Scalar_t>& local_mat) noexcept requires (Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(u_size==v_size);
			//assemble the local matrix in column major format
			local_mat.resize(v_size*v_size);
			for (size_t j=0; j<v_size; ++j) {
				if constexpr (LDU_Flag&0b010) { local_mat[j+v_size*j] = k_eval(j,j,wt_ptr);}
				else {local_mat[j+v_size*j] = Scalar_t{0};}
				for (size_t i=j+1; i<v_size; ++i) {
					Scalar_t val = k_eval(j,i,wt_ptr);
					if constexpr (LDU_Flag&0b100) {local_mat[i + v_size*j] = val;}
					else {local_mat[i + v_size*j] = Scalar_t{0};}

					if constexpr (LDU_Flag&0b001) {local_mat[j + v_size*i] = val;}
					else { local_mat[j + v_size*i] = Scalar_t{0};}
				}
			}
		}


		//////////////////////////////////////////////////////////////////
		/// Given trial coefficients X, compute Y += alpha*A*X
		//////////////////////////////////////////////////////////////////
		template<int LDU_Flag=0b111> requires (Kernel_t::IS_SYMMETRIC || LDU_Flag==0b111)
		void mat_vec_multiply_accumulate_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept {
			GUTIL_ASSERT(X.size()==trial_handler.n_dofs() && Y.size()==test_handler.n_dofs());
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_x, t_local_y, t_local_mat;

			for_each_element<true>(
			//action
			[&](auto& k_eval, auto& trial_cache, auto& test_cache, auto wt_ptr, const auto& range){
				auto& local_x=t_local_x[range.tid];
				auto& local_y=t_local_y[range.tid];
				auto& local_mat=t_local_mat[range.tid];
				GatherLocalVector(X, trial_cache, local_x);
				ConstructLocalMatrix<LDU_Flag>(k_eval,  trial_cache.size(), test_cache.size(), wt_ptr, local_mat);
				local_y.assign(test_cache.size(), Scalar_t{0});
				GV::gecm_mv(local_y.data(), local_y.size(), local_x.data(), local_x.size(), local_mat.data());
				ScatterLocalVector<false>(Y, test_cache, local_y, alpha);	//don't need an atomic scatter
			},
			//init
			[&t_local_x, &t_local_y, &t_local_mat](const auto& range) {
				GUTIL_OMP(single)
				{
					t_local_x.resize(range.n_threads);
					t_local_y.resize(range.n_threads);
					t_local_mat.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)
			});
		}

		
		template<int LDU_Flag=0b111> requires (Kernel_t::IS_SYMMETRIC || LDU_Flag==0b111)
		void mat_vec_multiply_accumulate(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept {
			GUTIL_ASSERT(X.size()==trial_handler.n_dofs() && Y.size()==test_handler.n_dofs());
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_x, t_local_y, t_local_mat;

			for_each_element<false>(
			//action
			[&](auto& k_eval, auto& trial_cache, auto& test_cache, auto wt_ptr, const auto& range){ //false=not colored
				auto& local_x=t_local_x[range.tid];
				auto& local_y=t_local_y[range.tid];
				auto& local_mat=t_local_mat[range.tid];
				GatherLocalVector(X, trial_cache, local_x);
				ConstructLocalMatrix<LDU_Flag>(k_eval,  trial_cache.size(), test_cache.size(), wt_ptr, local_mat);
				local_y.assign(test_cache.size(), Scalar_t{0});
				GV::gecm_mv(local_y.data(), local_y.size(), local_x.data(), local_x.size(), local_mat.data());
				ScatterLocalVector<true>(Y, test_cache, local_y, alpha);	//need an atomic scatter
			},
			//init
			[&t_local_x, &t_local_y, &t_local_mat](const auto& range) {
				GUTIL_OMP(single)
				{
					t_local_x.resize(range.n_threads);
					t_local_y.resize(range.n_threads);
					t_local_mat.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)
			});
		}

		
		/////////////////////////////////////////////////////////////////////
		/// Multiply and increment by the non-diagonal entries for jacobi preconditioning.
		/// Same as Y += alpha*(L+U)*X
		/////////////////////////////////////////////////////////////////////
		void mat_vec_multiply_accumulate_jacobi_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			mat_vec_multiply_accumulate_colored<0b101>(Y, X, alpha);
		}

		void mat_vec_multiply_accumulate_jacobi(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			mat_vec_multiply_accumulate<0b101>(Y, X, alpha);
		}


		/////////////////////////////////////////////////////////////////////
		/// Evaluate the quadratic form x^t * M * x when M is symmetric
		/////////////////////////////////////////////////////////////////////
		[[nodiscard]] T evaluate_quadratic_form(std::span<const T> X) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(X.size()==trial_handler.n_dofs());
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_x, t_local_mat;
			std::vector<T> t_val;

			for_each_element(
			//action
			[&](auto& k_eval, auto& trial_cache, auto& test_cache, auto wt_ptr, const auto& range){
				auto& local_x=t_local_x[range.tid];
				auto& local_mat=t_local_mat[range.tid];
				GatherLocalVector(X, trial_cache, local_x);
				ConstructLocalMatrix(k_eval, trial_cache.size(), test_cache.size(), wt_ptr, local_mat);
				//handle the local xMx evaluation
				t_val[range.tid] += GV::gecm_vmv(local_x.data(), local_x.size(), local_x.data(), local_x.size(), local_mat.data());},
			//init
			[&](const auto& range){
				GUTIL_OMP(single)
				{
					t_local_x.resize(range.n_threads);
					t_local_mat.resize(range.n_threads);
					t_val.assign(range.n_threads, T{0});
				}
				GUTIL_OMP(barrier)
			});
			
			T val{0};
			for (T v : t_val) {val += v;}
			return val;
		}


		[[nodiscard]] T evaluate_form(std::span<const T> Y, std::span<const T> X) const noexcept {
			GUTIL_ASSERT(X.size()==trial_handler.n_dofs() && Y.size()==test_handler.n_dofs());
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_x, t_local_y, t_local_mat;
			std::vector<T> t_val;

			for_each_element<false>(
			//action
			[&](auto& k_eval, auto& trial_cache, auto& test_cache, auto wt_ptr, const auto& range){
				auto& local_x=t_local_x[range.tid];
				auto& local_y=t_local_y[range.tid];
				auto& local_mat=t_local_mat[range.tid];
				GatherLocalVector(X, trial_cache, local_x);
				GatherLocalVector(Y, test_cache,  local_y);
				ConstructLocalMatrix(k_eval, trial_cache.size(), test_cache.size(), wt_ptr, local_mat);
				//handle the local yMx evaluation
				t_val[range.tid] += GV::gecm_vmv(local_y.data(), local_y.size(), local_x.data(), local_x.size(), local_mat.data());},
			//init
			[&](const auto& range){
				GUTIL_OMP(single)
				{
					t_local_x.resize(range.n_threads);
					t_local_y.resize(range.n_threads);
					t_local_mat.resize(range.n_threads);
					t_val.assign(range.n_threads, T{0});
				}
				GUTIL_OMP(barrier)
			});
			
			T val{0};
			for (T v : t_val) {val += v;}
			return val;
		}

		
		////////////////////////////////////////////////////////////////////
		/// Construct diagonal of the matrix for conditioning
		/////////////////////////////////////////////////////////////////////
		template<bool Colored=false>
		void construct_diagonal(std::span<T> D) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_mat;

			for_each_element<Colored>(
			//action
			[&,D](auto& k_eval, auto& trial_cache, auto& test_cache, auto wt_ptr, const auto& range){
				auto& local_mat = t_local_mat[range.tid];
				const size_t n = trial_cache.size();
				ConstructLocalMatrix<0b010>(k_eval, n, n, wt_ptr, local_mat);
				//accumulate the diagonal entries into D
				if constexpr (Colored) {
					for (size_t i=0; i<n; ++i) { D[trial_cache.global_idx[i]] += local_mat[(n+1)*i];}
				}
				else {
					for (size_t i=0; i<n; ++i) { GUTIL_OMP(atomic) D[trial_cache.global_idx[i]] += local_mat[(n+1)*i];}
				}
			},
			//init
			[&t_local_mat](const auto& range){
				GUTIL_OMP(single)
				{
					t_local_mat.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)
			});
		}

		template<bool Colored=false>
		void construct_lumped_diagonal(std::span<T> D) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_mat, t_row_sum;

			for_each_element<Colored>(
			//action
			[&,D](auto& k_eval, auto& trial_cache, auto& test_cache, auto wt_ptr, const auto& range){
				auto& local_mat = t_local_mat[range.tid];
				auto& row_sum   = t_row_sum[range.tid];
				const size_t n  = trial_cache.size();
				ConstructLocalMatrix(k_eval, n, n, wt_ptr, local_mat);
				//compute the row sums and scatter to the diagonal
				row_sum.assign(n, T{0});
				for (size_t j=0; j<n; ++j) {
					GUTIL_SIMD()
					for (size_t i=0; i<n; ++i) {
						row_sum[i] += local_mat[i + j*n];
					}
				}


				if constexpr (Colored) {
					for (size_t i=0; i<n; ++i) { D[trial_cache.global_idx[i]] += row_sum[i];}
				}
				else {
					for (size_t i=0; i<n; ++i) { GUTIL_OMP(atomic) D[trial_cache.global_idx[i]] += row_sum[i];}
				}
			},
			//init
			[&t_local_mat, &t_row_sum](const auto& range){
				GUTIL_OMP(single)
				{
					t_local_mat.resize(range.n_threads);
					t_row_sum.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)
			});
		}


		///////////////////////////////////////////////////////////////
		/// Construct triples for sparese matrix construction (any upper,lower,diagonal combinations if it is symmetric)
		///////////////////////////////////////////////////////////////
		template<int LDU_Flag=0b111, typename Triplet_t> requires (LDU_Flag==0b111 || Kernel_t::IS_SYMMETRIC)
		void build_triplets(std::vector<Triplet_t>& triplets, const size_t row_offset=0, const size_t col_offset=0) const noexcept {
			GUTIL_PROFILE_FUNCTION();
			
			std::vector<std::vector<T>> t_local_mat;
			std::vector<std::vector<Triplet_t>> t_coo;
			
			for_each_element<false>(
			//action
			[&](auto& k_eval, auto& trial_cache, auto& test_cache, auto wt_ptr, const auto& range){
				auto& local_mat = t_local_mat[range.tid];
				auto& coo       = t_coo[range.tid];

				const size_t u_size=trial_cache.size(), v_size=test_cache.size();
				ConstructLocalMatrix<LDU_Flag>(k_eval, u_size, v_size, wt_ptr, local_mat);
				
				//add triplets, only loop throught the lower block of indices
				for (size_t j=0; j<u_size; ++j) {
					if constexpr (LDU_Flag&0b010) {
						coo.emplace_back(
							row_offset + test_cache.global_idx[j],
							col_offset + trial_cache.global_idx[j],
							local_mat[j + j*u_size]);
					}

					for (size_t i=j+1; i<v_size; ++i) {
						if constexpr (LDU_Flag&0b100) {
							coo.emplace_back(
								row_offset + test_cache.global_idx[i],
								col_offset + trial_cache.global_idx[j],
								local_mat[i + j*u_size]);
						}
						if constexpr (LDU_Flag&0b001) {
							//upper block, transpose the indices
							coo.emplace_back(
								row_offset + test_cache.global_idx[j],
								col_offset + trial_cache.global_idx[i],
								local_mat[j + i*u_size]);
						}
					}
				}
			},
			//init
			[&t_local_mat, &t_coo](const auto& range){
				GUTIL_OMP(single)
				{
					t_local_mat.resize(range.n_threads);
					t_coo.resize(range.n_threads);
				}
				GUTIL_OMP(barrier)
			},
			//finalize
			[&t_coo](const auto& range){
				//compress the per-thread coo lists (also de-duplicates)
				Triplet_t::Compress(t_coo[range.tid]);
				t_coo[range.tid].shrink_to_fit();

				//merge the per-thread coo lists to thread 0 (also de-duplicates)
				GUTIL_OMP(barrier)
				for (size_t stride=1; stride<range.n_threads; stride*=2) {
					if (range.tid % (2*stride) == 0 && range.tid+stride < range.n_threads) {
						Triplet_t::Merge(t_coo[range.tid], t_coo[range.tid+stride]);
					}
					GUTIL_OMP(barrier)
				}
			});

			//move the new coo values to the provided vector
			triplets.insert(triplets.end(),
				std::make_move_iterator(t_coo[0].begin()),
				std::make_move_iterator(t_coo[0].end()));
		}
	};
}


