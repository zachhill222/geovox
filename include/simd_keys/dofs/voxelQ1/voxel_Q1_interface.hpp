#pragma once

#include "gutil.hpp"
#include "simd_keys/keyed_object.hpp"
#include "simd_keys/mesh/mesh_keys.hpp"

#include "simd_keys/dofs/utility.hpp"
#include "simd_keys/dofs/voxelQ1/voxel_Q1_implementation.hpp"

#include <cstdint>
#include <iostream>
#include <array>

namespace GV {
namespace Keys {
namespace DOFS {

	/////////////////////////////////////////////////////////////////
	/// Extend KeyedObject for mesh elements.
	/// Cartesian encoding is the "cononical" representation.
	/////////////////////////////////////////////////////////////////
	template<uint8_t Period=0> requires(Period<8)
	struct VoxelQ1 : public KeyedObject<Mesh3D::VERTEX_FLAG> {


		/////////////////////////////////////////////////////////////
		/// Constructors and factories
		/////////////////////////////////////////////////////////////
		using BASE = KeyedObject<Mesh3D::VERTEX_FLAG>;
		using BASE::BASE;
		using BASE::ID;
		using BASE::key;

		using DofElem_t  = VoxelElement<Period>;
		using DofVert_t  = VoxelVertex<Period>;
		using MeshElem_t = VoxelElement<0>;
		using MeshVert_t = VoxelVertex<0>;

		static constexpr uint8_t  PERIOD         = Period;
		static constexpr uint64_t N_DOF_PER_ELEM = LagrangeQ1::N_DOF_PER_ELEM;
		static constexpr uint64_t N_SUPPORT_ELEM = LagrangeQ1::N_SUPPORT_ELEM;
		static constexpr uint64_t N_CHILDREN     = LagrangeQ1::N_CHILDREN;
		static constexpr uint64_t N_PARENTS      = LagrangeQ1::N_PARENTS;

		[[nodiscard]] static std::string name() noexcept {return "VoxelQ1<" + std::to_string(Period) + ">";}

		static constexpr VoxelQ1 MakeFromIndex(uint64_t index) noexcept {
			return VoxelQ1<Period>{Mesh3D::VertexFromGlobalIndex_SIMD<Period>(index)};
		}

		static constexpr VoxelQ1 None() noexcept {return VoxelQ1{0};}

		constexpr VoxelQ1(uint64_t depth, uint64_t ii, uint64_t jj, uint64_t kk) noexcept :
			BASE{Mesh3D::MakeVertex<Period>(depth,ii,jj,kk)} {}

		
		/////////////////////////////////////////////////////////////
		/// Convert to/from the mesh vertex type
		/// We always assume that using static cast produces a valid result.
		/// Otherwise use the raw constructor VoxelVertex<P> vtx{dof.key}
		/////////////////////////////////////////////////////////////
		[[nodiscard]] explicit constexpr operator MeshVert_t() const noexcept {
			//note that vertices being periodic affects which indices are allowed
			//for the dof. the mesh feature has a larger space.
			return MeshVert_t{key};
		}
		
		[[nodiscard]] explicit constexpr operator DofVert_t() const noexcept requires (!std::same_as<DofVert_t,MeshVert_t>) {
			//note that vertices being periodic affects which indices are allowed
			//for the dof. the mesh feature has a larger space.
			return DofVert_t{key};
		}

		explicit constexpr VoxelQ1(MeshVert_t vtx) requires(Period!=0)
			: BASE{static_cast<DofVert_t>(vtx).key} {}
		explicit constexpr VoxelQ1(DofVert_t vtx) : BASE{vtx.key} {}


		/////////////////////////////////////////////////////////////
		/// Queries
		/////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr bool exists() const noexcept { return Mesh3D::Exists(key); }
		[[nodiscard]] constexpr bool is_valid() const noexcept { return Mesh3D::IsValid<Period>(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t linear_index_simd() const noexcept { return Mesh3D::GlobalVertexIndex_SIMD(key); }
		[[nodiscard]] constexpr uint64_t linear_index() const noexcept {
			GUTIL_ASSERT(Mesh3D::IsValid<0>(key)); //linear index does not take into account periodicity;
			return Mesh3D::GlobalVertexIndex(key);
		}
		[[nodiscard]] constexpr uint64_t depth_linear_index() const noexcept {
			GUTIL_ASSERT(is_valid());
			return Mesh3D::GlobalVertexIndex_SIMD(key) - Mesh3D::VerticesBelowDepth(Mesh3D::Depth(key));
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t i() const noexcept { return Mesh3D::IndexI_SIMD(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t j() const noexcept { return Mesh3D::IndexJ_SIMD(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t k() const noexcept { return Mesh3D::IndexK_SIMD(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t depth() const noexcept { return Mesh3D::Depth(key); }
		
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint8_t depth_u8() const noexcept { return Mesh3D::Depth_u8(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t total_possible(uint8_t dd) noexcept {
			//note that we always use non-periodic indices, so this is possibly an overcount
			return Mesh3D::VerticesBelowDepth(dd+1);
		}

		/////////////////////////////////////////////////////////////
		/// Support interactions
		/////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] DofElem_t* support_simd(DofElem_t* s) const noexcept {
			GUTIL_ASSERT(s);
			#ifndef NDEBUG
				std::fill(s, s+N_SUPPORT_ELEM, DofElem_t{uint64_t(-1)});
			#endif

			LagrangeQ1::GetDofSupport_SIMD<Period>(s, key);

			#ifndef NDEBUG
				GUTIL_ASSERT(std::find(s, s+N_SUPPORT_ELEM, DofElem_t{uint64_t(-1)})==s+N_SUPPORT_ELEM);
			#endif

			return s;
		}

		[[nodiscard]] constexpr std::array<DofElem_t,N_SUPPORT_ELEM> support() const noexcept {
			GUTIL_ASSERT(is_valid());
			std::array<uint64_t,N_SUPPORT_ELEM> keys;
			LagrangeQ1::GetDofSupport_SIMD<Period>(&keys[0], key);
			std::array<DofElem_t,N_SUPPORT_ELEM> result;
			int i=0; for (uint64_t k : keys) {result[i++] = DofElem_t{k};}
			return result;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t pairity_simd() const noexcept {
			//for general dofs, pairity must partition the dofs into [0,n_dofs_per_elem)
			return Mesh3D::CartesianIndexPairity_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		static constexpr void dofs_on_elem_simd_raw(uint64_t s, uint64_t* dof) noexcept {
			GUTIL_ASSERT(dof);
			GUTIL_ASSERT(DofElem_t{s}.is_valid());
			LagrangeQ1::GetDofsOnElement_SIMD<Period>(s, dof);
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] static VoxelQ1* dofs_on_elem_simd(uint64_t s, VoxelQ1* dof) noexcept {
			GUTIL_ASSERT(dof);
			GUTIL_ASSERT(DofElem_t{s}.is_valid());
			
			#ifndef NDEBUG
				std::fill(dof, dof+N_DOF_PER_ELEM, VoxelQ1{uint64_t(-1)});
			#endif

			LagrangeQ1::GetDofsOnElement_SIMD<Period>(s, reinterpret_cast<uint64_t*>(dof));

			#ifndef NDEBUG
			GUTIL_ASSERT(std::find(dof, dof+N_DOF_PER_ELEM, VoxelQ1{uint64_t(-1)})==dof+N_DOF_PER_ELEM);
			#endif

			return dof;
		}

		[[maybe_unused]] static std::array<VoxelQ1,N_DOF_PER_ELEM> dofs_on_elem(DofElem_t s) noexcept {
			std::array<uint64_t,N_DOF_PER_ELEM> keys{};
			std::array<VoxelQ1,N_DOF_PER_ELEM> dofs{};
			LagrangeQ1::GetDofsOnElement_SIMD<Period>(s.key, &keys[0]);
			for (uint64_t i=0; i<N_DOF_PER_ELEM; ++i) {dofs[i] = VoxelQ1{keys[i]};}
			return dofs;
		}

		template<typename T=double>
		[[maybe_unused]] constexpr DofElem_t project_to_support(MeshElem_t quad, T* X, T* Y, T* Z, uint32_t N) noexcept {
			return ProjectQuadratureElementToSupportElement<T,Period>(depth(), quad, X, Y, Z, N);
		}

		[[nodiscard]] constexpr uint8_t local_dof_number(DofElem_t spt) noexcept {
			return LagrangeQ1::LocalDofNumber_SIMD<Period>(spt.key, key);
		}

		/////////////////////////////////////////////////////////////
		/// Evaluate
		/////////////////////////////////////////////////////////////
		template<typename T=double>
		constexpr void evaluate_simd(uint8_t local, T* val, const T* X, const T* Y, const T* Z, uint32_t N) noexcept {
			LagrangeQ1::GetDofValueByLocalNumber(key, local, val, X, Y, Z, N);
		}

		template<typename T=double>
		constexpr void gradient_simd(uint8_t local, T* val_x, T* val_y, T* val_z, const T* X, const T* Y, const T* Z, uint32_t N) noexcept {
			LagrangeQ1::GetDofGradientByLocalNumber(key, local, val_x, val_y, val_z, X, Y, Z, N);
		}

		template<typename PointContainer>
		[[nodiscard]] constexpr typename PointContainer::value_type evaluate(DofElem_t spt, const PointContainer& pt) noexcept {
			return LagrangeQ1::GetDofValue<Period>(spt.key, key, pt);
		}

		template<typename PointContainer>
		[[nodiscard]] constexpr typename PointContainer::value_type evaluate(uint8_t local, const PointContainer& pt) noexcept {
			typename PointContainer::value_type val;
			evaluate_simd(local, &val, &pt[0], &pt[1], &pt[2], 1);
			return val;
		}

		template<typename PointContainer>
		[[nodiscard]] constexpr PointContainer gradient(DofElem_t spt, const PointContainer& pt) noexcept {
			return LagrangeQ1::GetDofGradient<Period>(spt.key, key, pt);
		}

		template<typename PointContainer>
		[[nodiscard]] constexpr PointContainer gradient(uint8_t local, const PointContainer& pt) noexcept {
			PointContainer val;
			evaluate_simd(local, &val[0], &val[1], &val[2], &pt[0], &pt[1], &pt[2], 1);
			return val;
		}


		/////////////////////////////////////////////////////////////
		/// Hierarchy
		/////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] VoxelQ1* children_simd(VoxelQ1* c) const noexcept {
			GUTIL_ASSERT(c);
			#ifndef NDEBUG
				std::fill(c, c+N_CHILDREN, VoxelQ1{uint64_t(-1)});
			#endif

			LagrangeQ1::GetDofChildren_SIMD<Period>(key, reinterpret_cast<uint64_t*>(c));

			#ifndef NDEBUG
			GUTIL_ASSERT(std::find(c, c+N_CHILDREN, VoxelQ1{uint64_t(-1)})==c+N_CHILDREN);
			#endif

			return c;
		}

		[[nodiscard]] constexpr std::array<VoxelQ1,N_CHILDREN> children() const noexcept {
			GUTIL_ASSERT(is_valid());
			std::array<uint64_t,N_CHILDREN> keys;
			LagrangeQ1::GetDofChildren_SIMD<Period>(key, &keys[0]);
			std::array<VoxelQ1,N_CHILDREN> result;
			int i=0; for (uint64_t k : keys) {result[i++] = VoxelQ1{k};}
			return result;
		}

		GUTIL_DECLARE_SIMD()
		template<typename T=double>
		[[nodiscard]] static constexpr T child_coef(uint8_t c) noexcept {
			return LagrangeQ1::GetChildCoef<T>(c);
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] VoxelQ1* parents_simd(VoxelQ1* p) const noexcept {
			GUTIL_ASSERT(p);
			
			#ifndef NDEBUG
				std::fill(p, p+N_PARENTS, VoxelQ1{uint64_t(-1)});
			#endif

			LagrangeQ1::GetDofParents_SIMD<Period>(key, reinterpret_cast<uint64_t*>(p));

			#ifndef NDEBUG
			GUTIL_ASSERT(std::find(p, p+N_PARENTS, VoxelQ1{uint64_t(-1)})==p+N_PARENTS);
			#endif

			return p;
		}

		[[nodiscard]] constexpr std::array<VoxelQ1,N_PARENTS> parents() const noexcept {
			GUTIL_ASSERT(is_valid());
			std::array<uint64_t,N_PARENTS> keys;
			LagrangeQ1::GetDofParents_SIMD<Period>(key, &keys[0]);
			std::array<VoxelQ1,N_PARENTS> result;
			int i=0; for (uint64_t k : keys) {result[i++] = VoxelQ1{k};}
			return result;
		}

		template<typename T=double>
		[[nodiscard]] constexpr T parent_coef_restrict(VoxelQ1 parent) const noexcept {
			//pass the parent number for a common interface and to help
			//catch bugs via the assert.
			GUTIL_ASSERT(parent.is_valid());
			return LagrangeQ1::GetParentCoef<T,Period>(key, parent.key);
		}

		GUTIL_DECLARE_SIMD()
		template<typename T=double>
		[[nodiscard]] constexpr T parent_coef_weight(uint8_t p) const noexcept {
			//pass the parent number for a common interface and to help
			//catch bugs via the assert.
			GUTIL_ASSERT(p<N_PARENTS);
			return LagrangeQ1::GetParentCoef_Weight<T,Period>(key);
		}
	};


	template<uint8_t Period>
	inline std::string to_string(VoxelQ1<Period> dof) {
		return "VoxelQ1<" + std::to_string(Period) + ">{" 
				+ std::to_string(dof.depth()) + ", " + std::to_string(dof.i())
				+ ", " + std::to_string(dof.j()) + ", " + std::to_string(dof.k()) + "}";
	}

	template<uint8_t Period>
	std::ostream& operator<<(std::ostream& os, VoxelQ1<Period> dof) {
		return os << to_string(dof);
	}


}}}