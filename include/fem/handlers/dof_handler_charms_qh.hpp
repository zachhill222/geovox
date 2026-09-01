#pragma once

#include "gutil.hpp"

#include "util/util.hpp"

#include "fem/handlers/dof_handler.hpp"

#include <concepts>
#include <cstdint>
#include <vector>
#include <span>
#include <type_traits>
#include <algorithm>
#include <thread>


namespace GV {


	/////////////////////////////////////////////////////////////////////////////
	/// DOF handler class to work with CHARMS quasi-hierarchical refinement.
	/////////////////////////////////////////////////////////////////////////////
	template<VoxelMeshType MeshType, typename DofType>
	struct CharmsHandlerQH : public DofHandler<MeshType,DofType> {
		using BASE = DofHandler<MeshType,DofType>;

		/////////////////////////////////////////////////////////////////////////
		/// Aliases and constants
		/////////////////////////////////////////////////////////////////////////
		static constexpr uint64_t HFlag = DofHierarchicalVariants::QuasiHierarchical | DofHierarchicalVariants::Conformal;

		using BASE::VERTEX_DOF ;
		using BASE::ELEMENT_DOF;
		using BASE::FACE_DOF   ;
		
		//DOF features may be periodic
		using DOF_t        = DofType;
		using DofVert_t    = Keys::VoxelVertex<DOF_t::PERIOD>;
		using DofElem_t    = Keys::VoxelElement<DOF_t::PERIOD>;
		using DofFeature_t = std::conditional_t<VERTEX_DOF, DofVert_t, std::conditional_t<ELEMENT_DOF, DofElem_t, void>>;
		static_assert(!std::same_as<DofFeature_t,void>);

		//Mesh features are never periodic
		using Mesh_t        = MeshType;
		using MeshElem_t    = typename Mesh_t::Elem_t;
		using MeshVert_t    = typename Mesh_t::Vert_t;
		using MeshFeature_t = std::conditional_t<VERTEX_DOF, MeshVert_t, std::conditional_t<ELEMENT_DOF, MeshElem_t, void>>;
		static_assert(!std::same_as<MeshFeature_t,void>);

		using BASE::IS_DEPTH_SEPARABLE;
		
		//The maximum depth of a given mesh is specified at runtime. However, to avoid accidentally
		//requesting say depth 12 (2^(3*12) elements at depth 12, (2^33 -1)/7 ~ 10^10.3 total elements)
		//we set a maximum depth at compile time.
		using BASE::max_depth;
		using BASE::max_possible_dofs;

		//For quasi-hierarchical refinement, there is generally a difference of at most 2 between an active element
		//and a dof. However, if multiple handlers are interested in the same mesh (e.g., Q1-iso-Q2 elements in a Stokes system),
		//we may specify this bound. It is up to the user to ensure that the bound is satisfied.
		using BASE::max_depth_distance;

		using BASE::ACTIVE_BIT; 	 //0b00000001;	
		using BASE::REFINED_BIT; 	 //0b00000010;
		using BASE::INITIAL_DOF_BIT; //0b00000100;	//these dofs don't check their parents and cannot be unrefined
		using BASE::FREE_BITS;		 //0b11111000;


		/////////////////////////////////////////////////////////////////////////
		/// Storage. Store a vector<uint8_t> for O(1) active queries.
		/// Additionally, store a compressed list of active dofs for tracking
		/// global DOF numbers. It is essential for fast quadrature that we may
		/// look up all active DOFs whos support OVERLAPS a given active element.
		///
		/// Using uint8_t instead of bool guarantees thread safe access of different elements
		/// and allows one bit to be used for an "is active" flag and another bit for
		/// "has been refined" flag, which is useful for hierarchical methods. Additionally,
		/// it gives us 6 more bits that could be used for other purposes.
		/////////////////////////////////////////////////////////////////////////
		protected:
		using BASE::key_mask;
		using BASE::threads;							//max hardware concurency by default
		
		public:
		using BASE::is_key_mask_unstable;
		using BASE::is_key_mask_stable;
		using BASE::begin_key_mask_stable;
		using BASE::end_key_mask_stable;
		using BASE::begin_key_mask_unstable;
		using BASE::end_key_mask_unstable;
		
		using BASE::is_active_keys_unstable;
		using BASE::is_active_keys_stable;
		using BASE::begin_active_keys_stable;
		using BASE::end_active_keys_stable;
		using BASE::begin_active_keys_unstable;
		using BASE::end_active_keys_unstable;

		using BASE::active_dofs;
		using BASE::mesh;								//link to the mesh, we can request refinement through const methods
		
		using BASE::is_current;							//check if the dofs are collected
		using BASE::get_mask;
		using BASE::get_mask_ref;

		/////////////////////////////////////////////////////////////////////////
		/// Constructors. The dofhandler must be linked to the mesh at construction
		/// and the mesh must outlive the dofhandler.
		/////////////////////////////////////////////////////////////////////////
		using BASE::BASE;


		///////////////////////////////////////////////////////////////////////////
		/// Define functions to sort DOFs into bins and to look up global dof numbers
		///////////////////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr int dof_key_bin(uint64_t dof_key) noexcept {
			return DOF_t{dof_key}.depth();
		}
		

		/////////////////////////////////////////////////////////////////////////
		/// Refinement queries that can be called inside either a stable or unstable
		/// region.
		///
		///	From 'Natural hierarchical refinement for finite element methods'
		///	in International J. for Numerical Methods in Engineering (2003, DOI 10.1002/nme.601)
		///
		/// 	There are 3 rules to guide refinement:
		///
		///	1) The refining/unrefining of a dof at depth dd may activate or deactivate that dof
		///			or any of its children at depth dd+1.
		///	2) A dof on level dd+1>0 may be refined only when all its parents on level dd have
		///			been refined. (We track this by the INITIAL_DOF_BIT. here dd=0 is the root element)
		///	3) A dof on level dd may be unrefined only if a) it was previously refined and
		///			b) all its children on level dd+1 are not refined.
		/////////////////////////////////////////////////////////////////////////
		[[nodiscard]] bool can_refine(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			if (dof.depth() >= max_depth) {return false;}		//we can't refine past max depth
			uint8_t byte = BASE::get_mask_no_check(dof.linear_index());
			
			
			if ((byte&ACTIVE_BIT)==0) {return false;}			//in QH only active dofs can be refined
			if (byte&REFINED_BIT) 	  {return false;}			//we can't refine a dof twice
			if (byte&INITIAL_DOF_BIT) {return true;}			//an unrefined initial dof not at max depth can be refined
			
			return BASE::has_all_refined_parents(dof);
		}

		[[nodiscard]] bool can_unrefine(DOF_t dof) const noexcept {
			GUTIL_ASSERT(dof.is_valid());
			if (!BASE::is_refined_no_check(dof)) {return false;}
			if (BASE::has_any_refined_child(dof)) {return false;}
			return true;
		}

		void activate(DOF_t dof) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());

			uint8_t& byte = get_mask_ref(dof);
			if (byte&ACTIVE_BIT) {
				GUTIL_ASSERT(!BASE::is_refined_no_check(dof));
				return;
			}

			byte|=ACTIVE_BIT;
			byte&=~REFINED_BIT;										//in QH, a dof can't be active and refined
			
			const uint8_t depth = dof.depth_u8();					//when refining, it is essential to have the mesh be able to resolve the support
			if (depth==0) {return;}

			for (DofElem_t spt : dof.support()) {			
				if (!spt.exists()) {continue;}
				MeshElem_t el = static_cast<MeshElem_t>(spt);
				if (mesh.read_depth_field(el) < depth) {
					GUTIL_ASSERT(mesh.is_active(el.parent()))				//the mesh should be respecting a 2-1 refinement rule
					mesh.request_refine(el.parent());						//this is a request. pushes the element to a mutable list. it is protected by a mutex.
				}
			}
		}

		////////////////////////////////////////////////////////////////////
		/// Refinement operations that must be called from within an unstable mask region.
		///
		/// Note that we are only requesting the mesh to refine. There is no guarentee
		/// that it will resolve every requested support element (e.g., when a complex geometry is being modeled).
		/// 
		/// Note that the mesh has a depth field so that we do not have to see if the support of a dof
		/// is resolved by finer elements than needed. If that is the case, the depth marker will be larger
		/// than the element depth.
		////////////////////////////////////////////////////////////////////
		void refine(DOF_t dof) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(BASE::is_active_unstable(dof));
			GUTIL_ASSERT(can_refine(dof));

			for (DOF_t c : dof.children()) {
				if (c.exists()) { activate(c); }
			}

			uint8_t& byte = get_mask_ref(dof);
			byte&=~ACTIVE_BIT;
			byte|=REFINED_BIT;
		}

		void unrefine(DOF_t dof) noexcept {
			GV_ASSERT_KEY_MASK_UNSTABLE_STATE
			GUTIL_ASSERT(dof.is_valid());
			GUTIL_ASSERT(can_unrefine(dof));

			activate(dof);											//send mesh refinement request
			for (DOF_t c : dof.children()) {
				if (c.exists() && !BASE::has_any_refined_parent(c)) {
					BASE::set_active(c,false);
				}
			}
		}


		//////////////////////////////////////////////////////////////////////////////////////
		/// Primary refine/unrefine commands for bulk operations. Called from outside this class.
		//////////////////////////////////////////////////////////////////////////////////////
		template<typename Container_t>
		[[maybe_unused]] size_t refine(const Container_t& elems) noexcept {
			return refine(GV::as_span(elems));
		}

		template<std::contiguous_iterator I>
		[[maybe_unused]] size_t refine(I begin, I end) noexcept {
			return refine(std::span<std::iter_value_t<I>>{begin, end});
		}

		template<typename Elem_t> requires (std::same_as<Elem_t,MeshElem_t> || std::same_as<Elem_t,DofElem_t>)
		[[maybe_unused]] size_t refine(std::span<const Elem_t> elems) noexcept {
			GUTIL_ASSERT(mesh.is_current() && mesh.is_depth_field_correct());
			GUTIL_ASSERT(is_current());
			GUTIL_PROFILE("Refining (QH) dofs on ", elems.size(), " elements");
			size_t n_start = active_dofs.size();

			auto pred = [this](DOF_t dof) { return can_refine(dof); };


			BASE::mark_stale();
			{
				GV_BEGIN_MASK_UNSTABLE
				std::span<const DofElem_t> d_elems = gutil::reinterpret_as_span<DofElem_t,Elem_t>(elems);
				std::vector<DOF_t> dofs = BASE::get_dofs_impl(d_elems, max_depth_distance, std::move(pred));
				GUTIL_PROFILE("Processing ", dofs.size(), " dofs for refinement");
				
				gutil::BinSort<DOF_t> dof_depth_sorter(dofs, max_depth+1);
				dof_depth_sorter.dispatch_sort([](DOF_t dof){return (int)dof.depth();}, &threads);
				threads.wait_idle();

				for (int dd=0; dd<dof_depth_sorter.n_bins(); ++dd) {
					auto list = dof_depth_sorter.get_bin(dd);
					if (list.empty()) {continue;}

					gutil::BinSort<DOF_t> pairity_sorter(list, DOF_t::N_CHILDREN);
					pairity_sorter.dispatch_sort([](DOF_t dof) {
						const uint64_t ii = dof.i()%3;
						const uint64_t jj = dof.j()%3;
						const uint64_t kk = dof.k()%3;

						return (int) ii + 3*(jj + 3*kk);
					}, &threads);
					threads.wait_idle();
					GUTIL_PROFILE("Checking ", list.size(), " dofs for refinement at depth ", dd);
					for (int cc=0; cc<pairity_sorter.n_bins(); ++cc) {
						auto par_list = pairity_sorter.get_bin(cc);
						GUTIL_OMP(parallel for)
						for (size_t i=0; i<par_list.size(); ++i) {
							if (can_refine(par_list[i])) {
								refine(par_list[i]);
							}
						}
					}
				}
				GV_END_MASK_UNSTABLE
			}


			BASE::collect_dofs();
			size_t n_end = active_dofs.size();
			GUTIL_ASSERT(n_end>=n_start);
			return n_end - n_start;
		}

		template<typename Container_t>
		[[maybe_unused]] size_t unrefine(const Container_t& elems) noexcept {
			return unrefine(GV::as_span(elems));
		}

		template<std::contiguous_iterator I>
		[[maybe_unused]] size_t unrefine(I begin, I end) noexcept {
			return unrefine(std::span<std::iter_value_t<I>>{begin, end});
		}

		template<typename Elem_t> requires (std::same_as<Elem_t,MeshElem_t> || std::same_as<Elem_t,DofElem_t>)
		[[maybe_unused]] size_t unrefine(std::span<const Elem_t> elems) noexcept {
			GUTIL_ASSERT(mesh.is_current() && mesh.is_depth_field_correct());
			GUTIL_ASSERT(is_current());
			GUTIL_PROFILE("Unrefining (QH) dofs on ", elems.size(), " elements");
			size_t n_start = active_dofs.size();

			auto pred = [this](DOF_t dof) {
				return can_unrefine(dof);
			};

			BASE::mark_stale();
			{
				GV_BEGIN_MASK_UNSTABLE
				std::span<const DofElem_t> d_elems = gutil::reinterpret_as_span<DofElem_t,Elem_t>(elems);
				std::vector<DOF_t> dofs = BASE::get_dofs_impl(d_elems, max_depth_distance, std::move(pred));
				GUTIL_PROFILE("Processing ", dofs.size(), " dofs for unrefinement");
				
				for (DOF_t dof : dofs) {
					if (can_unrefine(dof)) {
						unrefine(dof);
					}
				}
				
				GV_END_MASK_UNSTABLE
			}
			BASE::collect_dofs();
			
			size_t n_end = active_dofs.size();
			GUTIL_ASSERT(n_start>=n_end);
			return n_start-n_end;
		}
	};


	template<VoxelMeshType MeshType, typename DofType>
	std::ostream& operator<<(std::ostream& os, const CharmsHandlerQH<MeshType,DofType>& handler) {
		os << "CharmsHandlerQH: " << DofType::name() + "\n";
		os << handler.summary();
		os << gutil::format(handler.n_dofs(),16) << " active dofs\n";
		return os;
	}
}