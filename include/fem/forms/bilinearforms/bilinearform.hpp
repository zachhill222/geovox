#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"

#include "fem/mesh_quadrature.hpp"
#include "fem/forms/util.hpp"
#include "fem/forms/bilinearforms/triplet.hpp"
#include "fem/forms/bilinearforms/bilinear_kernels.hpp"
#include "fem/forms/bilinearforms/dense_linalg.hpp"
#include "fem/forms/base_k_linear_form.hpp"


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
	///
	/// This class is built entirely on top of KLinearForm<N,T,KernelType,KernelWeightType,TrialHandlerType,TestHandlerType>
	/// (K=2). All per-element kernel evaluation, dof caching, and looping now goes through the
	/// inherited KernelEval/for_each_element rather than a separate, BilinearForm-specific mechanism.
	/////////////////////////////////////////////////////////////////////
	template<int N, typename T, typename TrialHandlerType, typename TestHandlerType, typename KernelType, typename KernelWeightType=IdentityKernelWeight>
	class BilinearForm : public KLinearForm<N,T,KernelType,KernelWeightType,TrialHandlerType,TestHandlerType> {
	public:

		//////////////////////////////////////////////////////////////////
		/// Aliases and sanity checks
		//////////////////////////////////////////////////////////////////
		using Base           = KLinearForm<N,T,KernelType,KernelWeightType,TrialHandlerType,TestHandlerType>;
		using Kernel_t       = typename Base::Kernel_t;
		using Weight_t       = typename Base::Weight_t;
		using QuadRule_t     = typename Base::QuadRule_t;
		using Scalar_t       = typename Base::Scalar_t;
		using Mesh_t         = typename Base::Mesh_t;
		using MeshElem_t     = typename Base::MeshElem_t;
		using TrialHandler_t = TrialHandlerType;
		using TestHandler_t  = TestHandlerType;
		using TrialDof_t     = typename TrialHandlerType::DOF_t;
		using TestDof_t      = typename TestHandlerType::DOF_t;

		using TrialCache_t = typename Base::template ElemCache_t<0>;
		using TestCache_t  = typename Base::template ElemCache_t<1>;

		static constexpr bool IS_SYMMETRIC = Kernel_t::IS_SYMMETRIC;
		static_assert(!Kernel_t::IS_SYMMETRIC || std::same_as<TrialHandler_t,TestHandler_t>,
			"A symmetric kernel must have the same dofhandlers for the test and trial spaces");


		//////////////////////////////////////////////////////////////////
		/// Constructors -- delegate to KLinearForm's own handler constructor,
		/// deriving the mesh from the test handler (matching the original's own choice).
		//////////////////////////////////////////////////////////////////
		BilinearForm(const TrialHandler_t& u_handler, const TestHandler_t& v_handler, KernelType kernel = KernelType{}, KernelWeightType weight = KernelWeightType{}) :
			Base(v_handler.mesh, std::move(kernel), std::move(weight), u_handler, v_handler) {}

		BilinearForm(const TrialHandler_t& sym_handler, KernelType kernel=KernelType{}, KernelWeightType weight = KernelWeightType{}) requires(Kernel_t::IS_SYMMETRIC) :
			Base(sym_handler.mesh, std::move(kernel), std::move(weight), sym_handler, sym_handler) {}


		//////////////////////////////////////////////////////////////////
		/// A few convenience methods
		//////////////////////////////////////////////////////////////////
		[[nodiscard]] size_t n_rows() const noexcept {return this->template get_handler<1>().n_dofs();}
		[[nodiscard]] size_t n_cols() const noexcept {return this->template get_handler<0>().n_dofs();}


		//////////////////////////////////////////////////////////////////
		/// A thin wrapper preserving the established action-first calling convention
		/// used throughout this class, on top of KLinearForm::for_each_element (init-first).
		///
		/// Action must have the signature void(KernelEval&) or void(KernelEval&, OmpIteratorRange&).
		/// Init/Finalize must have the signature void(size_t n_threads, size_t tid) or they will not
		/// be called. Use lambda captures for any per-thread resources, e.g.:
		///
		/// std::vector<Data_t> thread_data;
		/// auto init = [&thread_data](size_t n_threads, size_t tid) {
		///		GUTIL_OMP(single)
		///  	{
		///   	  thread_data.resize(n_threads);
		///  	}
		/// 	 GUTIL_OMP(barrier)
		/// };
		//////////////////////////////////////////////////////////////////
		template<bool Colored=false, typename Action, typename Init=std::nullptr_t, typename Finalize=std::nullptr_t>
		void for_each_element(Action&& action, Init&& init=nullptr, Finalize&& finalize=nullptr) const noexcept {
			GUTIL_ASSERT(this->template get_handler<0>().is_current());
			GUTIL_ASSERT(this->template get_handler<1>().is_current());
			Base::template for_each_element<Colored>(std::forward<Init>(init), std::forward<Action>(action), std::forward<Finalize>(finalize));
		}


		//////////////////////////////////////////////////////////////////
		/// A few building blocks for the actions
		//////////////////////////////////////////////////////////////////
		//ConstructLocalMatrix: k_eval({j,i}) replaces the old k_eval(j,i,wt_ptr) -- the weight is
		//now handled internally by KernelEval::operator(), not passed explicitly by the caller.
		template<int LDU_Flag = 0b111, typename KernelEval_t>
		static void ConstructLocalMatrix(const KernelEval_t& k_eval, size_t u_size, size_t v_size,
			std::vector<Scalar_t>& local_mat) noexcept requires (!Kernel_t::IS_SYMMETRIC) {
			//assemble the local matrix in column major format
			local_mat.resize(u_size*v_size);
			size_t idx = 0; //index (i,j) in column major
			for (size_t j=0; j<u_size; ++j) {
				for (size_t i=0; i<v_size; ++i) {
					local_mat[idx++] = k_eval({j,i});
				}
			}
		}

		template<int LDU_Flag = 0b111, typename KernelEval_t>
		static void ConstructLocalMatrix(const KernelEval_t& k_eval, size_t u_size, size_t v_size,
			std::vector<Scalar_t>& local_mat) noexcept requires (Kernel_t::IS_SYMMETRIC) {
			GUTIL_ASSERT(u_size==v_size);
			//assemble the local matrix in column major format
			local_mat.resize(v_size*v_size);
			for (size_t j=0; j<v_size; ++j) {
				if constexpr (LDU_Flag&0b010) { local_mat[j+v_size*j] = k_eval({j,j});}
				else {local_mat[j+v_size*j] = Scalar_t{0};}
				for (size_t i=j+1; i<v_size; ++i) {
					Scalar_t val = k_eval({j,i});
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
		template<int LDU_Flag=0b111, bool Colored=false> requires (Kernel_t::IS_SYMMETRIC || LDU_Flag==0b111)
		void mat_vec_multiply_accumulate(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept {
			GUTIL_ASSERT(X.size()==this->template get_handler<0>().n_dofs() && Y.size()==this->template get_handler<1>().n_dofs());
			if constexpr (Colored) {GUTIL_ASSERT(this->mesh().is_color_sorted());}
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_x, t_local_y, t_local_mat;

			for_each_element<Colored>(
			//action
			[&](auto& k_eval, const auto& range){
				auto& local_x=t_local_x[range.tid];
				auto& local_y=t_local_y[range.tid];
				auto& local_mat=t_local_mat[range.tid];
				auto& trial_cache = std::get<0>(k_eval.dof_caches);
				auto& test_cache  = std::get<1>(k_eval.dof_caches);
				const size_t u_size = trial_cache.dofs.size(), v_size = test_cache.dofs.size();
				BASE::GatherLocalVector(X, trial_cache, local_x);
				ConstructLocalMatrix<LDU_Flag>(k_eval, u_size, v_size, local_mat);
				local_y.assign(v_size, Scalar_t{0});
				GV::gecm_mv(local_y.data(), local_y.size(), local_x.data(), local_x.size(), local_mat.data());
				//colored: elements of the same color are dof-disjoint, so no atomic scatter needed;
				//non-colored: different threads' elements may share dofs, so an atomic scatter is required.
				if constexpr (Colored) {
					if (alpha!=T{1}) {BASE::ScatterLocalVector<false>(Y, test_cache, local_y, alpha);}
					else {BASE::ScatterLocalVector<false>(Y, test_cache, local_y);}
				}
				else {
					if (alpha!=T{1}) {BASE::ScatterLocalVector<true>(Y, test_cache, local_y, alpha);}
					else {BASE::ScatterLocalVector<true>(Y, test_cache, local_y);}
				}
			},
			//init
			[&t_local_x, &t_local_y, &t_local_mat](size_t n_threads, size_t tid) {
				GUTIL_OMP(single)
				{
					t_local_x.resize(n_threads);
					t_local_y.resize(n_threads);
					t_local_mat.resize(n_threads);
				}
				GUTIL_OMP(barrier)
			});
		}


		/////////////////////////////////////////////////////////////////////
		/// Multiply and increment by the non-diagonal entries for jacobi preconditioning.
		/// Same as Y += alpha*(L+U)*X
		/////////////////////////////////////////////////////////////////////
		void mat_vec_multiply_accumulate_jacobi_colored(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			mat_vec_multiply_accumulate<0b101,true>(Y, X, alpha);
		}

		void mat_vec_multiply_accumulate_jacobi(std::span<T> Y, std::span<const T> X, const T alpha = T{1}) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			mat_vec_multiply_accumulate<0b101,false>(Y, X, alpha);
		}


		/////////////////////////////////////////////////////////////////////
		/// Evaluate the quadratic form x^t * M * x when M is symmetric
		/////////////////////////////////////////////////////////////////////
		[[nodiscard]] T evaluate_quadratic_form(std::span<const T> X) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_PROFILE_FUNCTION();
			GUTIL_ASSERT(X.size()==this->template get_handler<0>().n_dofs());
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_x, t_local_mat;
			std::vector<T> t_val;

			for_each_element(
			//action
			[&](auto& k_eval, const auto& range){
				auto& local_x=t_local_x[range.tid];
				auto& local_mat=t_local_mat[range.tid];
				auto& trial_cache = std::get<0>(k_eval.dof_caches);
				auto& test_cache  = std::get<1>(k_eval.dof_caches);
				const size_t u_size = trial_cache.dofs.size(), v_size = test_cache.dofs.size();
				BASE::GatherLocalVector(X, trial_cache, local_x);
				ConstructLocalMatrix(k_eval, u_size, v_size, local_mat);
				//handle the local xMx evaluation
				t_val[range.tid] += GV::gecm_vmv(local_x.data(), local_x.size(), local_x.data(), local_x.size(), local_mat.data());},
			//init
			[&](size_t n_threads, size_t tid){
				GUTIL_OMP(single)
				{
					t_local_x.resize(n_threads);
					t_local_mat.resize(n_threads);
					t_val.assign(n_threads, T{0});
				}
				GUTIL_OMP(barrier)
			});
			
			T val{0};
			for (T v : t_val) {val += v;}
			return val;
		}


		[[nodiscard]] T evaluate_form(std::span<const T> Y, std::span<const T> X) const noexcept {
			GUTIL_PROFILE_FUNCTION();
			GUTIL_ASSERT(X.size()==this->template get_handler<0>().n_dofs() && Y.size()==this->template get_handler<1>().n_dofs());
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_x, t_local_y, t_local_mat;
			std::vector<T> t_val;

			for_each_element(
			//action
			[&](auto& k_eval, const auto& range){
				auto& local_x=t_local_x[range.tid];
				auto& local_y=t_local_y[range.tid];
				auto& local_mat=t_local_mat[range.tid];
				auto& trial_cache = std::get<0>(k_eval.dof_caches);
				auto& test_cache  = std::get<1>(k_eval.dof_caches);
				const size_t u_size = trial_cache.dofs.size(), v_size = test_cache.dofs.size();
				BASE::GatherLocalVector(X, trial_cache, local_x);
				BASE::GatherLocalVector(Y, test_cache,  local_y);
				ConstructLocalMatrix(k_eval, u_size, v_size, local_mat);
				//handle the local yMx evaluation
				t_val[range.tid] += GV::gecm_vmv(local_y.data(), local_y.size(), local_x.data(), local_x.size(), local_mat.data());},
			//init
			[&](size_t n_threads, size_t tid){
				GUTIL_OMP(single)
				{
					t_local_x.resize(n_threads);
					t_local_y.resize(n_threads);
					t_local_mat.resize(n_threads);
					t_val.assign(n_threads, T{0});
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
			GUTIL_PROFILE_FUNCTION();
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_mat;

			for_each_element<Colored>(
			//action
			[&,D](auto& k_eval, const auto& range){
				auto& local_mat = t_local_mat[range.tid];
				auto& trial_cache = std::get<0>(k_eval.dof_caches);
				const size_t n = trial_cache.dofs.size();
				ConstructLocalMatrix<0b010>(k_eval, n, n, local_mat);
				//accumulate the diagonal entries into D
				if constexpr (Colored) {
					for (size_t i=0; i<n; ++i) { D[trial_cache.global_idx[i]] += local_mat[(n+1)*i];}
				}
				else {
					for (size_t i=0; i<n; ++i) { GUTIL_OMP(atomic) D[trial_cache.global_idx[i]] += local_mat[(n+1)*i];}
				}
			},
			//init
			[&t_local_mat](size_t n_threads, size_t tid){
				GUTIL_OMP(single)
				{
					t_local_mat.resize(n_threads);
				}
				GUTIL_OMP(barrier)
			});
		}

		template<bool Colored=false>
		void construct_lumped_diagonal(std::span<T> D) const noexcept requires(Kernel_t::IS_SYMMETRIC) {
			GUTIL_PROFILE_FUNCTION();
			using ThreadData = std::vector<std::vector<T>>;
			ThreadData t_local_mat, t_row_sum;

			for_each_element<Colored>(
			//action
			[&,D](auto& k_eval, const auto& range){
				auto& local_mat = t_local_mat[range.tid];
				auto& row_sum   = t_row_sum[range.tid];
				auto& trial_cache = std::get<0>(k_eval.dof_caches);
				const size_t n  = trial_cache.dofs.size();
				ConstructLocalMatrix(k_eval, n, n, local_mat);
				//compute the row sums and scatter to the diagonal
				row_sum.assign(n, T{0});
				for (size_t j=0; j<n; ++j) {
					GUTIL_SIMD()
					for (size_t i=0; i<n; ++i) {
						row_sum[i] += local_mat[i + j*n];
					}
				}

				if constexpr (Colored) {for (size_t i=0; i<n; ++i) { D[trial_cache.global_idx[i]] += row_sum[i];} }
				else {for (size_t i=0; i<n; ++i) { GUTIL_OMP(atomic) D[trial_cache.global_idx[i]] += row_sum[i];} }
			},
			//init
			[&t_local_mat, &t_row_sum](size_t n_threads, size_t tid){
				GUTIL_OMP(single)
				{
					t_local_mat.resize(n_threads);
					t_row_sum.resize(n_threads);
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
			
			for_each_element(
			//action
			[&](auto& k_eval, const auto& range){
				auto& local_mat = t_local_mat[range.tid];
				auto& coo       = t_coo[range.tid];
				auto& trial_cache = std::get<0>(k_eval.dof_caches);
				auto& test_cache  = std::get<1>(k_eval.dof_caches);

				const size_t u_size=trial_cache.dofs.size(), v_size=test_cache.dofs.size();
				ConstructLocalMatrix<LDU_Flag>(k_eval, u_size, v_size, local_mat);
				
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
			[&t_local_mat, &t_coo](size_t n_threads, size_t tid){
				GUTIL_OMP(single)
				{
					t_local_mat.resize(n_threads);
					t_coo.resize(n_threads);
				}
				GUTIL_OMP(barrier)
			},
			//finalize
			[&t_coo](size_t n_threads, size_t tid){
				//compress the per-thread coo lists (also de-duplicates)
				Triplet_t::Compress(t_coo[tid]);
				t_coo[tid].shrink_to_fit();

				//merge the per-thread coo lists to thread 0 (also de-duplicates)
				GUTIL_OMP(barrier)
				for (size_t stride=1; stride<n_threads; stride*=2) {
					if (tid % (2*stride) == 0 && tid+stride < n_threads) {
						Triplet_t::Merge(t_coo[tid], t_coo[tid+stride]);
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