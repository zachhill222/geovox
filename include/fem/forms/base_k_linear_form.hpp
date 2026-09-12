#pragma once


#include "gutil.hpp"

#include "util/util.hpp"
#include "fem/forms/base_k_linear_kernel.hpp"

namespace GV {
	//////////////////////////////////////////////////////////////////
	/// A base class for K-linear forms. This class stores the shared logic
	/// between these forms, but it may be best to have biliinear form inherit
	/// from KLinearForm<N,T,K, H1, H2> and similar for other specialized types.
	///
	/// It may be useful to have a 0-form to simply integrate a weight over the mesh.
	///
	/// Note for a bilinear form B(u,v) = int_D( kernel(u,v) ), we read the indices from right
	/// to left (i.e., v is index 0 and u is index 1). This allows us to keep consistent indexing
	/// through various types and have column-major storage indexing be the natural order.
	///
	/// Thus, specify dofhandlers from the right to left. To use symmetry, the symmetric group must be the rightmost.
	//////////////////////////////////////////////////////////////////
	template<int N, typename T, IsKLinearKernel KernelType, typename KernelWeightType, typename... HandlerTypes>
	class KLinearForm {
	public:
		//////////////////////////////////////////////////////////////////
		/// Aliases and constants for tracking DofHandler types.
		/// We can support a few symmetric arguments for the kernel, but only one group
		/// and they must be the first arguments. Any symmetric arguments must use
		/// the same dofhandler.
		//////////////////////////////////////////////////////////////////
		static constexpr size_t K = KernelType::K;
		static constexpr size_t N_SYMMETRIC = KernelType::N_SYMMETRIC;
		static constexpr size_t N_HANDLERS  = (N_SYMMETRIC==0) ? K : (K - N_SYMMETRIC + 1);	//number of required handlers

		//The KLinearForm type must be set using the full dofhandler types
		//but the first N_SYMMETRIC must be identical and only one handler
		//will be stored for those arguments.
		using HandlerTuple = std::tuple<HandlerTypes...>;
		
		template<size_t Index> requires (Index<K)
		using HandlerType = std::tuple_element_t<Index, HandlerTuple>;
		
		static_assert(KernelType::K == sizeof...(HandlerTypes), 
			"Must supply exactly K handler types (one per kernel argument)");
		
		static_assert([]<size_t... Is>(std::index_sequence<Is...>) {
			return (std::same_as<HandlerType<Is>, HandlerType<0>> && ...);
		}(std::make_index_sequence<N_SYMMETRIC>{}),
			"Symmetric slots must all use the same handler type.");


		static constexpr size_t GetHandlerIndex(size_t k) noexcept {
			if constexpr (N_SYMMETRIC==0) {return k;}
			else {return (k<N_SYMMETRIC) ? 0 : k + 1 - N_SYMMETRIC;}
		}

		static constexpr size_t GetFormPosition(size_t index) noexcept {
			if constexpr (N_SYMMETRIC==0) {return index;}
			else {return (index==0) ? 0 : index + N_SYMMETRIC - 1;}
		}

		template<size_t k> requires (k < K)
		using GetHandlerType = HandlerType<GetHandlerIndex(k)>;

		template<size_t k> requires (k < K)
		using GetDofType = typename HandlerType<GetHandlerIndex(k)>::DOF_t;


		//////////////////////////////////////////////////////////////////
		/// Other aliases and constants
		//////////////////////////////////////////////////////////////////
		using Kernel_t   = KernelType;
		using Weight_t   = std::conditional_t<KernelType::NEEDS_WEIGHT, KernelWeightType, int>;
		using QuadRule_t = MeshQuadratureRule<N,T>;
		using Scalar_t   = T;
		using Mesh_t     = UnstructuredVoxelMesh<T>;
		using MeshElem_t = typename Mesh_t::Elem_t;

		template<size_t Index>
		using ElemCache_t   = ElementDofCacheNew<Kernel_t, QuadRule_t, typename HandlerType<Index>::DOF_t>;
		using WeightCache_t = std::conditional_t<KernelType::NEEDS_WEIGHT, ScalarValueCache<QuadRule_t>, int>;


		//////////////////////////////////////////////////////////////////
		/// A helper type to evaluate the kernel and manage the per-thread
		/// caches that need to be updated per-element. Implemented below.
		//////////////////////////////////////////////////////////////////
		struct KernelEval;


	protected:
		const Mesh_t* 						mesh_ptr{nullptr};
		Kernel_t      						kernel{};
		[[no_unique_address]] Weight_t		weight{};
		std::array<const void*, N_HANDLERS> handlers{};


	public:
		//////////////////////////////////////////////////////////////////
		/// Constructors
		//////////////////////////////////////////////////////////////////
		KLinearForm(const Mesh_t& m, KernelType k, Weight_t w, const HandlerTypes&... hs) :
			mesh_ptr(&m), kernel{std::move(k)}, weight{std::move(w)} {set_handlers(hs...);}

		KLinearForm(const Mesh_t& m, KernelType k = KernelType{}, Weight_t w = Weight_t{}) :
			mesh_ptr(&m), kernel{std::move(k)}, weight{std::move(w)}, handlers{} {}

		KLinearForm()=default;
		KLinearForm(const KLinearForm&)=default;
		KLinearForm(KLinearForm&&)=default;
		KLinearForm& operator=(const KLinearForm&)=default;
		KLinearForm& operator=(KLinearForm&&)=default;


		//////////////////////////////////////////////////////////////////
		/// Setters and getters
		//////////////////////////////////////////////////////////////////
		void set_handlers(const HandlerTypes&... hs) {
			std::array<const void*,K> full_handlers{static_cast<const void*>(&hs)...};

			for (size_t k=0; k<N_SYMMETRIC; ++k) {
				if (full_handlers[k]!=full_handlers[0]) {GUTIL_ABORT("symmetric block has different dofhandlers");}
			}

			[&]<size_t... Is>(std::index_sequence<Is...>) {
				((handlers[Is] = full_handlers[GetFormPosition(Is)]), ...);
			}(std::make_index_sequence<N_HANDLERS>{});
		}


		//set the handlers with one handler per variable
		//symmetric components must use the same handler
		template<typename... Handlers> requires (
			sizeof...(Handlers)==N_HANDLERS && N_HANDLERS<K &&
			[]<size_t... Is>(std::index_sequence<Is...>) {
				return (std::same_as<std::tuple_element_t<Is, std::tuple<Handlers...>>, HandlerType<GetFormPosition(Is)>> && ...);
			}(std::make_index_sequence<N_HANDLERS>{})
		)
		void set_handlers(const Handlers&... hs) {
			handlers = std::array<const void*,N_HANDLERS>{static_cast<const void*>(&hs)...};
		}


		[[nodiscard]] const Mesh_t& mesh() const noexcept {GUTIL_ASSERT(mesh_ptr); return *mesh_ptr;}
		void set_mesh(const Mesh_t& m) noexcept {mesh_ptr = &m;}

		template<size_t k> requires (k < K)
		[[nodiscard]] const HandlerType<k>& get_handler() const noexcept {
			static constexpr size_t Index = GetHandlerIndex(k);
			static_assert(std::same_as<HandlerType<k>, GetHandlerType<Index>>);
			return *reinterpret_cast<const HandlerType<k>*>(handlers[Index]);
		}

	protected:
		//////////////////////////////////////////////////////////////////
		/// Primary loops to apply some sort of action per element.
		/// For example, the action may be to compute the local stiffness matrix
		/// and then either build coo triplets or concatenate along some specified index.
		///
		/// Action must have the signature
		///   void(KernelEval&) or void(KernelEval&, OmpIteratorRange&)
		///
		/// Init must have the signature void(size_t n_threads, size_t thread_id) or it will not be called
		/// Finalize must have the signature void(size_t n_threads, size_t thread_id) or it will not be called
		///
		/// These can be used to set up per-thread resources (e.g., local vectors/matrices)
		/// so they don't have to be re-allocated per-element. It may be needed to put
		/// GUTIL_OMP(barrier) at the end of init and the beginning of finalize. Use lambda captures
		/// to manipulate the per-thread resources.
		///
		/// Init may look something like:
		/// std::vector<Data_t> thread_data;
		/// auto init = [&thread_data](size_t n_threads, size_t tid) {
		///		GUTIL_OMP(single)
		///  	{
		///   	  thread_data.resize(n_threads);
		///  	}
		/// 	 GUTIL_OMP(barrier)
		/// };
		///
		/// It may be desirable to pass an int for init and/or finalize if they are not needed.
		///
		/// Note that OmpIteratorRange contains:
		///		begin     - iterator to the beginning of the current thread data
		///		end       - iterator to the end of the current thread data
		///     count     - the number of current data in the current thread (same as std::distance(begin,end))
		///		tid       - the number of the current thread in the OpenMP thread group/pool
		/// 	n_threads - the total number of threads in the current OpenMP thread group/pool
		///
		/// Note for the snapshot variant, passing which=0 should be the same as the standard variant
		/// but is likely to be slower.
		//////////////////////////////////////////////////////////////////
		template<typename OmpIteratorRange_t, typename Action>
		static void ApplyActionOverRange(const OmpIteratorRange_t& range, KernelEval& k_eval, Action&& action) noexcept {
			static constexpr bool ACTION_NEEDS_RANGE = std::is_invocable_r_v<void, Action&, KernelEval&, const OmpIteratorRange_t&>;
			static_assert(ACTION_NEEDS_RANGE || std::is_invocable_r_v<void, Action&, KernelEval&>, "Action signature is invalid");
			for (auto it=range.begin; it!=range.end; ++it) {
				k_eval.set_element(*it);
				if constexpr (ACTION_NEEDS_RANGE) {action(k_eval, range);}
				else {action(k_eval);}
			}
		}

		template<bool Colored=false, typename Action, typename Init = std::nullptr_t,  typename Finalize= std::nullptr_t>
		void for_each_element(Action&& action, Init&& init=nullptr, Finalize&& finalize=nullptr) const noexcept {
			static constexpr bool HAS_INIT = std::is_invocable_r_v<void, Init&, size_t, size_t>;
			static constexpr bool HAS_FINALIZE = std::is_invocable_r_v<void, Finalize&, size_t, size_t>;

			if constexpr (Colored) {
				GUTIL_OMP(parallel)
				{
					KernelEval k_eval(*this);
					const gutil::OmpIndexRange dummy_range(size_t{0});//get n_threads and tid
					if constexpr (HAS_INIT) {init(dummy_range.n_threads, dummy_range.tid);}
					for (int clr=0; clr<mesh().n_colors(); ++clr) {
						auto quad_elems = mesh().get_color(clr);
						const gutil::OmpIteratorRange range(quad_elems.begin(), quad_elems.end());
						ApplyActionOverRange(range, k_eval, action);
						GUTIL_OMP(barrier)   // colors are only dof-disjoint within themselves, not across colors
					}
					if constexpr (HAS_FINALIZE) {finalize(dummy_range.n_threads, dummy_range.tid);}
				}
			}
			else {
				GUTIL_OMP(parallel)
				{
					KernelEval k_eval(*this);
					const gutil::OmpIteratorRange range(mesh().element_begin(), mesh().element_end());
					if constexpr (HAS_INIT) {init(range.n_threads, range.tid);}
					ApplyActionOverRange(range, k_eval, action);
					if constexpr (HAS_FINALIZE) {finalize(range.n_threads, range.tid);}
				}
			}
		}


		template<bool Colored=false, typename Action, typename Init = std::nullptr_t,  typename Finalize= std::nullptr_t>
		void for_each_element(uint8_t which, const gutil::BinSortVector<typename HandlerTypes::DOF_t>&... bin_sorts, 
				Action&& action, Init&& init=nullptr, Finalize&& finalize=nullptr) const noexcept {
			static constexpr bool HAS_INIT = std::is_invocable_r_v<void, Init&, size_t, size_t>;
			static constexpr bool HAS_FINALIZE = std::is_invocable_r_v<void, Finalize&, size_t, size_t>;

			if constexpr (Colored) {
				GUTIL_OMP(parallel)
				{
					KernelEval k_eval(*this, which, bin_sorts...);
					const gutil::OmpIndexRange dummy_range(size_t{0});//get n_threads and tid
					if constexpr (HAS_INIT) {init(dummy_range.n_threads, dummy_range.tid);}
					for (size_t clr=0; clr<mesh().n_colors(); ++clr) {
						auto quad_elems = mesh().get_color(clr);
						const gutil::OmpIteratorRange range(quad_elems.begin(), quad_elems.end());
						ApplyActionOverRange(range, k_eval, action);
						GUTIL_OMP(barrier)   // colors are only dof-disjoint within themselves, not across colors
					}
					if constexpr (HAS_FINALIZE) {finalize(dummy_range.n_threads, dummy_range.tid);}
				}
			}
			else {
				GUTIL_OMP(parallel)
				{
					KernelEval k_eval(*this, which, bin_sorts...);
					const gutil::OmpIteratorRange range(mesh().element_begin(), mesh().element_end());
					if constexpr (HAS_INIT) {init(range.n_threads, range.tid);}
					ApplyActionOverRange(range, k_eval, action);
					if constexpr (HAS_FINALIZE) {finalize(range.n_threads, range.tid);}
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
			const size_t n = dof_cache.global_idx.size();
			local_x.assign(n,Scalar_t{0});
			for (size_t i=0; i<n; ++i) {local_x[i] = X[dof_cache.global_idx[i]];}
		}

		template<typename Cache_t, typename VecX>
		static void GatherLocalVector(std::span<const Scalar_t> X, const Cache_t& dof_cache, VecX&& local_x) noexcept {
			//dof_cache has the dof values and global numbers for test/trial dofs whose support overlaps the current element
			//for allowing simd operations and better memory caching, it is often best to copy/gather the (spread out)
			//X-values into contiguous values.
			const size_t n = dof_cache.global_idx.size();
			//assume local_x is correctly sized
			for (size_t i=0; i<n; ++i) {local_x[i] = X[dof_cache.global_idx[i]];}
		}

		template<bool Atomic, typename Cache_t>
		static void ScatterLocalVector(std::span<Scalar_t> Y, const Cache_t& dof_cache, std::span<const Scalar_t> local_y, Scalar_t alpha) noexcept {
			//increment Y by alpha*y_local with the local to global index conversion supplied by the cache.
			GUTIL_ASSERT(local_y.size()==dof_cache.global_idx.size());
			const size_t n = dof_cache.global_idx.size();
			if constexpr (Atomic) {
				for (size_t i=0; i<n; ++i) { GUTIL_OMP(atomic) Y[dof_cache.global_idx[i]] += alpha*local_y[i];}
			}
			else {
				for (size_t i=0; i<n; ++i) {Y[dof_cache.global_idx[i]] += alpha*local_y[i];}
			}
		}

		template<bool Atomic, typename Cache_t>
		static void ScatterLocalVector(std::span<Scalar_t> Y, const Cache_t& dof_cache, std::span<const Scalar_t> local_y) noexcept {
			//increment Y by y_local with the local to global index conversion supplied by the cache.
			GUTIL_ASSERT(local_y.size()==dof_cache.global_idx.size());
			const size_t n = dof_cache.global_idx.size();
			if constexpr (Atomic) {
				for (size_t i=0; i<n; ++i) { GUTIL_OMP(atomic) Y[dof_cache.global_idx[i]] += local_y[i];}
			}
			else {
				for (size_t i=0; i<n; ++i) {Y[dof_cache.global_idx[i]] += local_y[i];}
			}
		}


		//////////////////////////////////////////////////////////////////
		/// A few fallback/generic methods.
		/// Note that coefficients are indexed right to left.
		//////////////////////////////////////////////////////////////////
		template<typename... Spans> requires (sizeof...(Spans)==K && AllArgsSameAs<std::span<const Scalar_t>, Spans...>)
		[[nodiscard]] Scalar_t evaluate(Spans... x_spans) const noexcept {
			GUTIL_PROFILE_FUNCTION();
			std::array<std::span<const Scalar_t>, K> X{x_spans...};

			std::vector<std::vector<Scalar_t>> t_thread_resource;
			std::vector<DynamicVectorValueCache<Scalar_t,K>> t_local_x;
			std::vector<Scalar_t> t_val;

			this->for_each_element(
			//action
			[&](auto& k_eval, const auto& range) {
				auto& thread_resource = t_thread_resource[range.tid];
				auto& local_x = t_local_x[range.tid];

				std::array<size_t,K> sizes;
				[&]<size_t... Slot>(std::index_sequence<Slot...>) {
					((sizes[Slot] = std::get<GetHandlerIndex(Slot)>(k_eval.dof_caches).global_idx.size()), ...);
				}(std::make_index_sequence<K>{});
				local_x.resize(sizes);

				[&]<size_t... Slot>(std::index_sequence<Slot...>) {
					(GatherLocalVector(X[Slot], std::get<GetHandlerIndex(Slot)>(k_eval.dof_caches), local_x[Slot]), ...);
				}(std::make_index_sequence<K>{});

				k_eval.make_local_tensor(thread_resource);
				t_val[range.tid] += gutil::Tensor<T,K>::EvaluateKFormConsume(thread_resource, local_x.to_spans(), std::move(sizes));
			},
			//init
			[&](size_t n_threads, size_t tid) {
				GUTIL_OMP(single)
				{
					t_thread_resource.resize(n_threads);
					t_local_x.resize(n_threads);
					t_val.assign(n_threads, Scalar_t{0});
				}
				GUTIL_OMP(barrier)
			});

			Scalar_t total{0};
			for (Scalar_t v : t_val) {total += v;}
			return total;
		}


		///////////////////////////////////////////////////////////////////////////
		/// Methods to build triplets for the sparse matrix. Wrappers designate to the *_impl method.
		///////////////////////////////////////////////////////////////////////////
		template<size_t RowSlot=0, size_t ColSlot=1, typename Triplet_t, typename LPred=std::nullptr_t, typename GPred=std::nullptr_t> requires (K==2)
		void build_triplets(std::vector<Triplet_t>& triplets, size_t row_offset=0, size_t col_offset=0, LPred&& l_pred=nullptr, GPred&& g_pred=nullptr) const noexcept {
			build_triplets_impl<RowSlot,ColSlot,Triplet_t,LPred,GPred>(triplets, row_offset, col_offset, std::forward<LPred>(l_pred), std::forward<GPred>(g_pred));
		}

	protected:
		//build a sparse matrix in COO format. allow specifying which variable to use for rows/columns
		//and supplying variables of the other fields to contract with.
		template<size_t RowSlot, size_t ColSlot, typename Triplet_t, typename LocalIndexPredicate, typename GlobalIndexPredicate, typename... ExtraSpans> requires (
			RowSlot<K && ColSlot<K && RowSlot!=ColSlot &&
			sizeof...(ExtraSpans)==K-2 &&
			(std::same_as<ExtraSpans,std::span<const Scalar_t>> && ...)
		)
		void build_triplets_impl(std::vector<Triplet_t>& triplets, size_t row_offset, size_t col_offset,
			LocalIndexPredicate&& l_pred, GlobalIndexPredicate&& g_pred, ExtraSpans... extra_spans) const noexcept {
			
			static_assert(std::same_as<LocalIndexPredicate,std::nullptr_t> || std::is_invocable_r_v<bool, LocalIndexPredicate, size_t, size_t>);
			static_assert(std::same_as<GlobalIndexPredicate,std::nullptr_t> || std::is_invocable_r_v<bool, GlobalIndexPredicate, size_t, size_t>);
			
			GUTIL_PROFILE_FUNCTION();
			std::array<std::span<const Scalar_t>, K-2> X_extra{extra_spans...};

			[[maybe_unused]] static constexpr std::array<size_t,K-2> ExtraSlots = []() {
				std::array<size_t,K-2> result{};
				size_t pos = 0;
				for (size_t s=0; s<K; ++s) {
					if (s!=RowSlot && s!=ColSlot) {result[pos++] = s;}
				}
				return result;
			}();

			std::vector<std::vector<Scalar_t>> t_thread_resource;
			std::vector<DynamicVectorValueCache<Scalar_t,K-2>> t_local_extra;
			std::vector<std::vector<Triplet_t>> t_coo;

			const auto local_pred = [l_pred = std::forward<LocalIndexPredicate>(l_pred)](size_t row, size_t col) {
				if constexpr (!std::same_as<LocalIndexPredicate,std::nullptr_t>) {return l_pred(row,col);}
				else {return true;}
			};
			const auto global_pred = [g_pred = std::forward<GlobalIndexPredicate>(g_pred)](size_t row, size_t col) {
				if constexpr (!std::same_as<GlobalIndexPredicate,std::nullptr_t>) {return g_pred(row,col);}
				else {return true;}
			};

			this->for_each_element(
			[&](auto& k_eval, const auto& range) {
				auto& thread_resource = t_thread_resource[range.tid];
				auto& local_extra = t_local_extra[range.tid];
				auto& coo = t_coo[range.tid];

				std::array<size_t,K-2> extra_sizes;
				[&]<size_t... Idx>(std::index_sequence<Idx...>) {
					((extra_sizes[Idx] = std::get<GetHandlerIndex(ExtraSlots[Idx])>(k_eval.dof_caches).global_idx.size()), ...);
				}(std::make_index_sequence<K-2>{});
				local_extra.resize(extra_sizes);

				[&]<size_t... Idx>(std::index_sequence<Idx...>) {
					(GatherLocalVector(X_extra[Idx], std::get<GetHandlerIndex(ExtraSlots[Idx])>(k_eval.dof_caches), local_extra[Idx]), ...);
				}(std::make_index_sequence<K-2>{});

				//make_local_tensor now returns void and mutates thread_resource in place --
				//construct a local, non-owning view over the already-populated buffer.
				k_eval.make_local_tensor(thread_resource);
				gutil::RawTensorBase<Scalar_t,K,false> tensor(thread_resource.data(), k_eval.make_local_tensor_size());

				auto extra_spans_view = local_extra.to_spans();
				for (size_t idx=0; idx<K-2; ++idx) {
					tensor.contract_axis_vector(ExtraSlots[idx], extra_spans_view[idx].data());
				}

				const auto& row_cache = std::get<GetHandlerIndex(RowSlot)>(k_eval.dof_caches);
				const auto& col_cache = std::get<GetHandlerIndex(ColSlot)>(k_eval.dof_caches);
				const size_t v_size = row_cache.dofs.size(), u_size = col_cache.dofs.size();

				std::array<size_t,K> idx{};
				for (size_t j=0; j<u_size; ++j) {
					idx[ColSlot] = j;
					for (size_t i=0; i<v_size; ++i) {
						idx[RowSlot] = i;
						if (!local_pred(i,j)) {continue;}
						size_t row = row_offset + row_cache.global_idx[i];
						size_t col = col_offset + col_cache.global_idx[j];
						if (!global_pred(row,col)) {continue;}
						coo.emplace_back(row, col, tensor(idx));
					}
				}
			},
			[&](size_t n_threads, size_t tid) {
				GUTIL_OMP(single)
				{
					t_thread_resource.resize(n_threads);
					t_local_extra.resize(n_threads);
					t_coo.resize(n_threads);
				}
				GUTIL_OMP(barrier)
			},
			[&](size_t n_threads, size_t tid) {
				Triplet_t::Compress(t_coo[tid]);
				t_coo[tid].shrink_to_fit();
				GUTIL_OMP(barrier)
				for (size_t stride=1; stride<n_threads; stride*=2) {
					if (tid % (2*stride) == 0 && tid+stride < n_threads) {
						Triplet_t::Merge(t_coo[tid], t_coo[tid+stride]);
					}
					GUTIL_OMP(barrier)
				}
			});

			triplets.insert(triplets.end(),
				std::make_move_iterator(t_coo[0].begin()),
				std::make_move_iterator(t_coo[0].end()));
		}
	};













	//////////////////////////////////////////////////////////////////
	/// A helper type to evaluate the kernel and manage the per-thread
	/// caches that need to be updated per-element.
	//////////////////////////////////////////////////////////////////
	template<int N, typename T, IsKLinearKernel KernelType, typename KernelWeightType, typename... HandlerTypes>
	struct KLinearForm<N,T,KernelType,KernelWeightType,HandlerTypes...>::KernelEval {
		//////////////////////////////////////////////////////////////
		/// Aliases and helper classes
		//////////////////////////////////////////////////////////////
		using ValueArg  = typename Kernel_t::template ValueArgType<QuadRule_t>;  //array of pointers to DofValueCache
		using GradArg   = typename Kernel_t::template GradArgType<QuadRule_t>;	 //array of pointers to DofGradCache
		using WeightArg = typename Kernel_t::template WeightArgType<QuadRule_t>; //pointer to this

		//helper class to store the element cache per dofhandler
		template<typename Seq> struct CacheTupleHelper;
		template<size_t... Indices> struct CacheTupleHelper<std::index_sequence<Indices...>> {
			using type = std::tuple<ElemCache_t<Indices>...>;
		};
		
		using CacheTuple_t = typename CacheTupleHelper<std::make_index_sequence<N_HANDLERS>>::type;

		//helper to store type-erased pointers to gutil::BinSortVector<DOF_t> for snapshots
		//and reinterpret them with the correct type.
		template<size_t Index>
		using DofType = typename HandlerType<Index>::DOF_t;

		template<size_t Index>
		const auto& get_bin_sort_vector() const noexcept {
			return *reinterpret_cast<const gutil::BinSortVector<DofType<Index>>*>(bin_sort_vectors[Index]);
		}

		//helper to store element to dof lookup lambda functions
		//these set the std::vector<DOF_t> dofs in each ElementCache directly.
		template<size_t... Is>
		[[nodiscard]] std::array<std::function<void(MeshElem_t)>, N_HANDLERS> make_getters_impl(uint8_t which, std::index_sequence<Is...>) noexcept {
			return {kform.template get_handler<GetFormPosition(Is)>().template make_dof_getter<MeshElem_t>(which, std::get<Is>(dof_caches).dofs)... };
		}


		//////////////////////////////////////////////////////////////
		/// Access caches by form index
		//////////////////////////////////////////////////////////////
		template<size_t k> requires (k<K)
		const auto& get_cache() const noexcept {
			return std::get<GetHandlerIndex(k)>(dof_caches);
		}


		//////////////////////////////////////////////////////////////
		/// Per-thread and per-element resources
		//////////////////////////////////////////////////////////////
		const KLinearForm& kform;
		QuadRule_t qr;
		uint8_t qr_depth{};
		Kernel_t kernel{};
		[[no_unique_address]] Weight_t weight{};

		CacheTuple_t dof_caches{};
		[[no_unique_address]] WeightArg wt_cache{};

		//data only used for snapshots
		[[no_unique_address]] std::array<const void*, N_HANDLERS> bin_sort_vectors{};
		[[no_unique_address]] std::array<std::function<void(MeshElem_t)>, N_HANDLERS> getters{};
		const bool using_snapshot{false};


		//////////////////////////////////////////////////////////////
		/// Per-thread constructors (before element loop)
		//////////////////////////////////////////////////////////////
		KernelEval(const KLinearForm& kf) noexcept : kform{kf}, qr{kf.mesh()}, kernel{kform.kernel}, weight{kform.weight} {
			if constexpr (K==0) {qr_depth = 0;}
			else {
				qr_depth = [&]<size_t... Is>(std::index_sequence<Is...>) {
					return std::min((uint8_t)kform.mesh().max_depth, gutil::max(kform.template get_handler<Is>().max_depth_distance...));
				}(std::make_index_sequence<K>{});
			}
		}

		KernelEval(const KLinearForm& kf, uint8_t which, const gutil::BinSortVector<typename HandlerTypes::DOF_t>&... bin_sorts) noexcept :
			kform{kf}, qr{kf.mesh()}, kernel{kform.kernel}, weight{kform.weight},
			bin_sort_vectors{static_cast<const void*>(&bin_sorts)...}, using_snapshot{true} {
			if constexpr (K==0) {qr_depth = 0;}
			else {
				qr_depth = [&]<size_t... Is>(std::index_sequence<Is...>) {
					return std::min((uint8_t)kform.mesh().max_depth, gutil::max(kform.template get_handler<Is>().max_depth_distance()...));
				}(std::make_index_sequence<K>{});
			}
			getters = make_getters_impl(which, std::make_index_sequence<N_HANDLERS>{});
		}



		//////////////////////////////////////////////////////////////
		/// Per-element update methods
		//////////////////////////////////////////////////////////////
		void set_element(MeshElem_t el) noexcept {
			qr.set_element(el, qr_depth);
			if constexpr (Kernel_t::NEEDS_WEIGHT) {
				if constexpr (Weight_t::NEEDS_GEO_POINTS) {qr.build_geometric_coords();}
				wt_cache = weight.template build_weights<QuadRule_t>(nullptr, qr);
			}
			build_caches(el);
		}


		//////////////////////////////////////////////////////////////
		/// Helper methods to build the caches and select arguments from
		/// the caches. Note that these methods take advantage of the
		/// symmetry so that other methods can be fairly generic and efficient.
		//////////////////////////////////////////////////////////////
		void build_caches(MeshElem_t el) noexcept {
			GUTIL_PROFILE_FUNCTION(N_HANDLERS);
			if (!using_snapshot) {
				//call the cache updates using the dof_handlers
				[&]<size_t... Is>(std::index_sequence<Is...>) {
					(ElemCache_t<Is>::template Update<GetFormPosition(Is)>(&std::get<Is>(dof_caches), kform.template get_handler<GetFormPosition(Is)>(), qr), ...);
				}(std::make_index_sequence<N_HANDLERS>{});
			} else {
				//call the cache updates using the lookup functions
				[&]<size_t... Is>(std::index_sequence<Is...>) {
					( ( getters[Is](el),
						ElemCache_t<Is>::template Update<GetFormPosition(Is)>(&std::get<Is>(dof_caches), get_bin_sort_vector<Is>(), qr) )
					, ...);
				}(std::make_index_sequence<N_HANDLERS>{});
			}
		}

		[[nodiscard]] ValueArg build_value_arg(std::array<size_t,K> local_idx) const noexcept {
			ValueArg vals{};
			[&]<size_t... Ks>(std::index_sequence<Ks...>) {
				((void)([&]{
					if constexpr (Kernel_t::NEED_VALS[Ks]) {
						vals[Ks] = &std::get<GetHandlerIndex(Ks)>(dof_caches).vals[local_idx[Ks]];
					}
				}()), ...);
			}(std::make_index_sequence<K>{});
			return vals;
		}

		[[nodiscard]] GradArg build_grad_arg(std::array<size_t,K> local_idx) const noexcept {
			GradArg grads{};
			[&]<size_t... Ks>(std::index_sequence<Ks...>) {
				((void)([&]{
					if constexpr (Kernel_t::NEED_GRAD[Ks]) {
						grads[Ks] = &std::get<GetHandlerIndex(Ks)>(dof_caches).grad[local_idx[Ks]];
					}
				}()), ...);
			}(std::make_index_sequence<K>{});
			return grads;
		}


		//////////////////////////////////////////////////////////////
		/// Evaluate the kernel for some combination of local dof indices.
		//////////////////////////////////////////////////////////////
		[[nodiscard]] Scalar_t operator()(std::array<size_t,K> local_idx) const noexcept {
			const WeightArg* wt_ptr{nullptr};
			if constexpr (Kernel_t::NEEDS_WEIGHT) {wt_ptr = &wt_cache;}

			if constexpr (Kernel_t::template HasStaticEval<QuadRule_t>()) {
				return Kernel_t::CachedEval(build_value_arg(local_idx), build_grad_arg(local_idx), wt_ptr, qr);
			}
			else {
				return kernel.cached_eval(build_value_arg(local_idx), build_grad_arg(local_idx), wt_ptr, qr);
			}
		}

		[[nodiscard]] Scalar_t operator()() const noexcept requires (K==0) {
			return operator()({});
		}


		//////////////////////////////////////////////////////////////
		/// Assemble the local 'stiffness' matrix/tensor using
		/// a provided per-thread resource vector. This does not exploit
		/// symmetry and should only be used as a fallback/temporary implementation.
		//////////////////////////////////////////////////////////////
		[[nodiscard]] std::array<size_t,K> make_local_tensor_size() const noexcept {
			return [&]<size_t... Is>(std::index_sequence<Is...>) {
				return std::array<size_t,K>{std::get<GetHandlerIndex(Is)>(dof_caches).dofs.size()...};
			}(std::make_index_sequence<K>{});
		}

		void make_local_tensor(std::vector<Scalar_t>& thread_resource) const noexcept requires (K==2 && N_SYMMETRIC==2) {
			GUTIL_PROFILE_FUNCTION("SymmetricBilinearForm");
			const size_t n = std::get<0>(dof_caches).dofs.size();
			thread_resource.resize(n*n);
			for (size_t j=0; j<n; ++j) {
				thread_resource[j+j*n] = (*this)({j,j});
				for (size_t i=j+1; i<n; ++i) {
					const Scalar_t val = (*this)({i,j});
					thread_resource[i+j*n] = val;
					thread_resource[j+i*n] = val;
				}
			}
		}

		void make_local_tensor(std::vector<Scalar_t>& thread_resource) const noexcept requires (!(K==2 && N_SYMMETRIC==2)) {
			GUTIL_PROFILE_FUNCTION(N_SYMMETRIC);
			const std::array<size_t,K> dims = make_local_tensor_size();
			const size_t total = gutil::product_reduce(std::span<const size_t,K>{dims});
			thread_resource.resize(total);

			using Tensor = gutil::Tensor<T,K>;
			// gutil::TensorWrapper<Scalar_t,K> tensor(thread_resource, make_local_tensor_size());

			// Scalar_t* data = thread_resource.data();
			std::array<size_t,K> local_idx;

			if constexpr (N_SYMMETRIC==0) {
				for (size_t flat=0; flat<total; ++flat) {
					// tensor.flat_to_tensor_index(flat, local_idx);
					Tensor::FlatToTensorIndex(flat, local_idx, dims);
					thread_resource[flat] = (*this)(local_idx);
				}
			}
			else {
				//canonicalize: sorts the first N_SYMMETRIC entries of idx into the
				//minimum-flat-index order -- descending, since column-major storage
				//weights earlier slots by smaller strides, so the smallest flat index
				//comes from putting the largest values there. Returns whether idx was
				//already in that order. note that dimensions of the tensor along each
				//symmetric axis is constant.
				auto canonicalize = [](std::array<size_t,K>& idx) noexcept -> bool {
					if constexpr (N_SYMMETRIC==2) {
						//by far the most common case (symmetric bilinear forms) --
						//a direct compare-and-swap, no loop overhead at all.
						if (idx[0] >= idx[1]) {return true;}
						std::swap(idx[0], idx[1]);
						return false;
					}
					else {
						//general fallback: single-pass insertion sort.
						bool was_canonical = true;
						for (size_t a=1; a<N_SYMMETRIC; ++a) {
							const size_t val = idx[a];
							size_t b = a;
							while (b>0 && idx[b-1]<val) {
								idx[b] = idx[b-1];
								--b;
								was_canonical = false;
							}
							idx[b] = val;
						}
						return was_canonical;
					}
				};

				for (size_t flat=0; flat<total; ++flat) {
					Tensor::FlatToTensorIndex(flat, local_idx, dims);
					// tensor.flat_to_tensor_index(flat, local_idx);

					if (canonicalize(local_idx)) {
						thread_resource[flat] = (*this)(local_idx);
					}
					else {
						//local_idx is now the canonical version; its flat index is <= the
						//current one, so it's guaranteed already computed by a prior iteration.
						// thread_resource[flat] = thread_resource[tensor.flat_index(local_idx)];
						thread_resource[flat] = thread_resource[Tensor::FlatIndex(flat, local_idx, dims)];
					}
				}
			}

			// return tensor;
		}
	};
}