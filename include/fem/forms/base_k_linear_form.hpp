#pragma once


#include "gutil.hpp"


namespace GV {
	//////////////////////////////////////////////////////////////////
	/// A base class for K-linear forms. This class stores the shared logic
	/// between these forms, but it may be best to have biliinear form inherit
	/// from KLinearForm<N,T,K, H1, H2> and similar for other specialized types.
	//////////////////////////////////////////////////////////////////
	template<int N, typename T, typename KernelWeightType, typename... HandlerTypes>
	class KLinearForm {
	public:
		static constexpr size_t K = sizeof...(HandlerTypes);
		using HandlerTuple = std::tuple<HandlerTypes...>;

		template<size_t Index>
		using HandlerType = std::tuple_element_t<Index, HandlerTuple>;

		using Weight_t   = KernelWeightType;
		using QuadRule_t = MeshQuadratureRule<N,T>;
		using Scalar_t   = T;
		using Mesh_t     = UnstructuredVoxelMesh<T>;
		using MeshElem_t = typename Mesh_t::Elem_t;

	protected:
		const Mesh_t*              mesh_ptr{nullptr};
		Weight_t                   weight;
		std::array<const void*, K> handlers;

	public:
		//////////////////////////////////////////////////////////////////
		/// Constructors
		//////////////////////////////////////////////////////////////////
		KLinearForm(const Mesh_t& m, KernelWeightType w, const HandlerTypes&... hs) :
			mesh_ptr(&m), weight(std::move(w)), handlers{static_cast<const void*>(&hs)...} {}

		KLinearForm()=default;
		KLinearForm(const KLinearForm&)=default;
		KLinearForm(KLinearForm&&)=default;
		KLinearForm& operator=(const KLinearForm&)=default;
		KLinearForm& operator=(KLinearForm&&)=default;


		//////////////////////////////////////////////////////////////////
		/// Accessors
		//////////////////////////////////////////////////////////////////
		[[nodiscard]] const Mesh_t& mesh() const noexcept {GUTIL_ASSERT(mesh_ptr); return *mesh_ptr;}

		template<size_t Index>
		[[nodiscard]] const HandlerType<Index>& get_form() const noexcept {
			return *reinterpret_cast<const HandlerType<Index>*>(handlers[Index]);
		}

		[[nodiscard]] const auto& form0() const noexcept requires (K>0) {return get_form<0>();}
		[[nodiscard]] const auto& form1() const noexcept requires (K>1) {return get_form<1>();}
		[[nodiscard]] const auto& form2() const noexcept requires (K>2) {return get_form<2>();}
		[[nodiscard]] const auto& form3() const noexcept requires (K>3) {return get_form<3>();}
	};
}