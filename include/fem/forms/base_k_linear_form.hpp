#pragma once


#include "gutil.hpp"

#include "fem/forms/base_k_linear_kernel.hpp"

namespace GV {
	//////////////////////////////////////////////////////////////////
	/// A base class for K-linear forms. This class stores the shared logic
	/// between these forms, but it may be best to have biliinear form inherit
	/// from KLinearForm<N,T,K, H1, H2> and similar for other specialized types.
	///
	/// It may be useful to have a 0-form to simply integrate a weight over the mesh.
	//////////////////////////////////////////////////////////////////
	template<int N, typename T, IsKLinearKernel KernelType, typename KernelWeightType=void, typename... HandlerTypes>
	class KLinearForm {
	public:
		//////////////////////////////////////////////////////////////////
		/// Aliases and constants for tracking DofHandler types.
		/// We can support a few symmetric arguments for the kernel, but only one group
		/// and they must be the first arguments. Any symmetric arguments must use
		/// the same dofhandler.
		//////////////////////////////////////////////////////////////////
		static constexpr size_t K = KernelType::K;
		static constexpr size_t N_HANDLERS = sizeof...(HandlerTypes);
		static constexpr size_t N_SYMMETRIC = KernelType::N_SYMMETRIC;

		static_assert(KernelType::K == 0 || (N_SYMMETRIC==0 ? (N_HANDLERS==KernelType::K) : 
											(KernelType::K == (N_HANDLERS + N_SYMMETRIC) - 1)),
			"Kernel and handlers must have compatible sizes");
		using HandlerTuple = std::tuple<HandlerTypes...>;

		template<size_t Index>
		using HandlerType = std::tuple_element_t<Index, HandlerTuple>;

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

		private:
		template<typename Seq> struct CacheTupleHelper;
		template<size_t... Indices> struct CacheTupleHelper<std::index_sequence<Indices...>> {
			using type = std::tuple<ElemCache_t<Indices>...>;
		};
		public:
		using CacheTuple_t = typename CacheTupleHelper<std::make_index_sequence<N_HANDLERS>>::type;

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
			mesh_ptr(&m), kernel{std::move(k)}, weight{std::move(w)}, handlers{static_cast<const void*>(&hs)...} {}

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
		[[nodiscard]] const Mesh_t& mesh() const noexcept {GUTIL_ASSERT(mesh_ptr); return *mesh_ptr;}
		void set_mesh(const Mesh_t& m) noexcept {mesh_ptr = &m;}

		template<size_t k> requires (k < K)
		[[nodiscard]] const GetHandlerType<k>& get_handler() const noexcept {
			static constexpr size_t Index = GetHandlerIndex(k);
			return *reinterpret_cast<const HandlerType<Index>*>(handlers[Index]);
		}

		template<size_t k> requires (k < K)
		void set_handler(const GetHandlerType<k>& h) noexcept {
			static constexpr size_t Index = GetHandlerIndex(k);
			handlers[Index] = &h;
		}

		[[nodiscard]] const auto& handler0() const noexcept requires (K>0) {return get_handler<0>();}
		[[nodiscard]] const auto& handler1() const noexcept requires (K>1) {return get_handler<1>();}
		[[nodiscard]] const auto& handler2() const noexcept requires (K>2) {return get_handler<2>();}
		[[nodiscard]] const auto& handler3() const noexcept requires (K>3) {return get_handler<3>();}

		void set_handler0(const GetHandlerType<0>& h) noexcept requires (K>0) {set_handler<0>(h);}
		void set_handler1(const GetHandlerType<1>& h) noexcept requires (K>1) {set_handler<1>(h);}
		void set_handler2(const GetHandlerType<2>& h) noexcept requires (K>2) {set_handler<2>(h);}
		void set_handler3(const GetHandlerType<3>& h) noexcept requires (K>3) {set_handler<3>(h);}

	protected:
		//////////////////////////////////////////////////////////////////
		/// Primary loops to apply some sort of action per element.
		/// For example, the action may be to compute the local stiffness matrix
		/// and then either build coo triplets or concatenate along some specified index.
		///
		/// Action must have the signature
		///   void(KernelEval&, Args&&...) or void(KernelEval&, OmpIteratorRange&, Args&&...)
		///
		/// Init must have the signature void(OmpIteratorRange&, Args&&...) or it will not be called
		/// Finalize must have the signature void(OmpIteratorRange&, Args&&...) or it will not be called
		///
		/// These can be used to set up per-thread resources (e.g., local vectors/matrices)
		/// so they don't have to be re-allocated per-element. It may be needed to put
		/// GUTIL_OMP(barrier) at the end of init and the beginning of finalize.
		///
		/// It may be desirable to pass an int for init and/or finalize if they are not needed.
		///
		/// If the mesh is colored, some scatters (in finalize) may be lock free.
		///
		/// Note that OmpIteratorRange contains:
		///		begin     - iterator to the beginning of the current thread data
		///		end       - iterator to the end of the current thread data
		///     count     - the number of current data in the current thread (same as std::distance(begin,end))
		///		tid       - the number of the current thread in the OpenMP thread group/pool
		/// 	n_threads - the total number of threads in the current OpenMP thread group/pool
		//////////////////////////////////////////////////////////////////
		template<typename Init, typename Action, typename Finalize, typename... Args>
		void for_each_element(Init&& init, Action&& action, Finalize&& finalize, Args&&... args) const noexcept {
			GUTIL_PROFILE_FUNCTION();
			using OmpIteratorRange = gutil::OmpIteratorRange<decltype(mesh().element_begin())>;

			static constexpr bool ACTION_NEEDS_RANGE = std::is_invocable_r_v<void, Action&, 
					KernelEval&, OmpIteratorRange&, Args&&...>;
			static_assert(ACTION_NEEDS_RANGE || std::is_invocable_r_v<void, Action&, KernelEval&, Args&&...>,
					"Action signature is invalid");

			static constexpr bool HAS_INIT = std::is_invocable_r_v<void, Init&, OmpIteratorRange&, Args&&...>;
			static constexpr bool HAS_FINALIZE = std::is_invocable_r_v<void, Finalize&, OmpIteratorRange&, Args&&...>;

			GUTIL_OMP(parallel)
			{
				KernelEval k_eval(*this);
				const OmpIteratorRange range(mesh().element_begin(), mesh().element_end());
				if constexpr (HAS_INIT) {init(range, args...);}
				for (auto it=range.begin; it!=range.end; ++it) {
					k_eval.set_element(*it);
					if constexpr (ACTION_NEEDS_RANGE) {action(k_eval, range, args...);}
					else {action(k_eval, args...);}
				}
				if constexpr (HAS_FINALIZE) {finalize(range, args...);}
			}
		}
	};


	//////////////////////////////////////////////////////////////////
	/// A helper type to evaluate the kernel and manage the per-thread
	/// caches that need to be updated per-element.
	//////////////////////////////////////////////////////////////////
	template<int N, typename T, IsKLinearKernel KernelType, typename KernelWeightType, typename... HandlerTypes>
	struct KLinearForm<N,T,KernelType,KernelWeightType,HandlerTypes...>::KernelEval {
		using ValueArg = typename Kernel_t::template ValueArg<QuadRule_t>; //array of pointers to DofValueCache
		using GradArg = typename Kernel_t::template GradArg<QuadRule_t>;	//array of pointers to DofGradCache
		using WeightArg = typename Kernel_t::template WeightArg<QuadRule_t>; //pointer to this

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


		//////////////////////////////////////////////////////////////
		/// Per-thread constructor (before element loop)
		//////////////////////////////////////////////////////////////
		KernelEval(const KLinearForm& kf) noexcept : 
			kform{kf}, qr{kf.mesh()}, kernel{kform.kernel}, weight{kform.weight} {
				if constexpr (K==0) {qr_depth = 0;}
				else {
					qr_depth = [&]<size_t... Is>(std::index_sequence<Is...>) {
						return std::min((uint8_t)kform.mesh().max_depth, gutil::max(kform.template get_handler<Is>().max_depth_distance()...));
					}(std::make_index_sequence<K>{});
				}
			}


		//////////////////////////////////////////////////////////////
		/// Per-element update methods
		//////////////////////////////////////////////////////////////
		void set_element(MeshElem_t el) noexcept {
			qr.set_element(el, qr_depth);
			if constexpr (Kernel_t::NEEDS_WEIGHT) {
				if constexpr (Weight_t::NEEDS_GEO_POINTS) {qr.build_geometric_coords();}
				if constexpr (Weight_t::NEEDS_SCALAR_VALS) {GUTIL_ABORT("Not supported");} //TODO: I think we can remove this from the weight
				wt_cache = weight.template build_weights<QuadRule_t>(nullptr, qr);
			}
			build_caches();
		}


		//////////////////////////////////////////////////////////////
		/// Helper methods to build the caches and select arguments from
		/// the caches. Note that these methods take advantage of the
		/// symmetry so that other methods can be fairly generic and efficient.
		//////////////////////////////////////////////////////////////
		void build_caches() noexcept {
			[&]<size_t... Is>(std::index_sequence<Is...>) {
				(ElemCache_t<Is>::template Update<GetFormPosition(Is)>(&std::get<Is>(dof_caches), 
															kform.template get_handler<Is>(), qr), ...);
			}(std::make_index_sequence<N_HANDLERS>{});
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
	};


}