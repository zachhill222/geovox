#pragma once

#include "gutil.hpp"
#include "simd_keys/keyed_object.hpp"
#include "simd_keys/mesh/mesh_key_implementation.hpp"

#include <cstdint>
#include <iostream>

namespace GV {
namespace Keys{

	/////////////////////////////////////////////////////////////////
	/// Forward declare the other feature classes
	/////////////////////////////////////////////////////////////////
	template<uint8_t Period> requires(Period<8)
	struct VoxelElement;


	/////////////////////////////////////////////////////////////////
	/// Extend KeyedObject for mesh elements.
	/// Cartesian encoding is the "cononical" representation.
	/////////////////////////////////////////////////////////////////
	template<uint8_t Period=0> requires(Period<8)
	struct VoxelVertex : public KeyedObject<Mesh3D::VERTEX_FLAG> {


		/////////////////////////////////////////////////////////////
		/// Constructors and factories
		/////////////////////////////////////////////////////////////
		using BASE = KeyedObject<Mesh3D::VERTEX_FLAG>;
		using BASE::BASE;
		using BASE::ID;
		using BASE::key;

		static constexpr uint8_t PERIOD = Period;

		static constexpr VoxelVertex MakeFromIndex(uint64_t index) noexcept {
			return VoxelVertex<Period>{Mesh3D::VertexFromGlobalIndex_SIMD<Period>(index)};
		}

		static constexpr VoxelVertex None() noexcept {return VoxelVertex{0};}

		constexpr VoxelVertex(uint64_t depth, uint64_t ii, uint64_t jj, uint64_t kk) noexcept :
			BASE{Mesh3D::MakeVertex<Period>(depth,ii,jj,kk)} {GUTIL_ASSERT(is_valid());}
			

		/////////////////////////////////////////////////////////////
		/// Convert between period types
		/////////////////////////////////////////////////////////////
		template<uint8_t P> requires(P<8)
		[[nodiscard]] explicit constexpr operator VoxelVertex<P>() const noexcept {
			//note that vertices being periodic affects which indices are allowed,
			//so we must reconstruct the key
			return VoxelVertex<P>{Mesh3D::Depth(key),
					Mesh3D::IndexI_SIMD(key), Mesh3D::IndexJ_SIMD(key), Mesh3D::IndexK_SIMD(key)};
		}

		
		/////////////////////////////////////////////////////////////
		/// Queries
		/////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr bool exists() const noexcept { return Mesh3D::Exists(key); }
		[[nodiscard]] constexpr bool is_valid() const noexcept { return Mesh3D::IsValid<Period>(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t linear_index_simd() const noexcept { return Mesh3D::GlobalVertexIndex_SIMD(key); }
		[[nodiscard]] constexpr uint64_t linear_index() const noexcept {
			GUTIL_ASSERT(Mesh3D::IsValid<0>(key)); //linear index does not take into account periodicity
			return Mesh3D::GlobalVertexIndex(key);
		}
		[[nodiscard]] constexpr uint64_t depth_linear_index() const noexcept {
			GUTIL_ASSERT(is_valid())
			return Mesh3D::GlobalVertexIndex_SIMD(key) - Mesh3D::VerticesBelowDepth(Mesh3D::Depth(key));
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t i() const noexcept { return Mesh3D::IndexI_SIMD(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t j() const noexcept { return Mesh3D::IndexJ_SIMD(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t k() const noexcept { return Mesh3D::IndexK_SIMD(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr int64_t i_i64() const noexcept { return static_cast<int64_t>(Mesh3D::IndexI_SIMD(key)); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr int64_t j_i64() const noexcept { return static_cast<int64_t>(Mesh3D::IndexJ_SIMD(key)); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr int64_t k_i64() const noexcept { return static_cast<int64_t>(Mesh3D::IndexK_SIMD(key)); }
		
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t depth() const noexcept { return Mesh3D::Depth(key); }
		
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint8_t depth_u8() const noexcept { return Mesh3D::Depth_u8(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t vertices_below_depth(uint64_t dd) noexcept requires(Period==0) {
			return Mesh3D::VerticesBelowDepth(dd);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t kji_pairity_simd() const noexcept {
			return Mesh3D::CartesianIndexPairity_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t total_possible(uint8_t dd) noexcept {
			//note that we always use non-periodic indices, so this is possibly an overcount
			return Mesh3D::VerticesBelowDepth(dd+1);
		}

		/////////////////////////////////////////////////////////////
		/// Hierarchy
		/////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr VoxelVertex child() const noexcept {
			GUTIL_ASSERT(is_valid());
			return VoxelVertex{Mesh3D::VertexChild(key)};
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t child_simd() const noexcept {
			return VoxelVertex{Mesh3D::VertexChild(key)};
		}

		[[nodiscard]] constexpr VoxelVertex parent() const noexcept {
			GUTIL_ASSERT(is_valid());
			return VoxelVertex{Mesh3D::VertexParent(key)};
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t parent_simd() const noexcept {
			return Mesh3D::VertexParent(key);
		}


		/////////////////////////////////////////////////////////////
		/// Adjacency
		/// 
		/// Note that the element query will return elements in cartesian
		/// rather than morton form.
		/////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr std::array<VoxelElement<Period>,8> elements() const noexcept;
		[[nodiscard]] constexpr VoxelElement<Period> element(int i) const noexcept;

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] VoxelElement<Period>* elements_simd(VoxelElement<Period>* ptr) const noexcept;

		[[nodiscard]] constexpr std::array<VoxelVertex,26> neighbors() const noexcept {
			return Mesh3D::GetVertexNeighbors<VoxelVertex>(key);
		}
		[[nodiscard]] constexpr VoxelVertex neighbor(int i) const noexcept {
			GUTIL_ASSERT(0<=i && i<26);
			return neighbors()[i];
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] VoxelVertex* neighbors_simd(VoxelVertex* ptr) const noexcept {
			GUTIL_ASSERT(ptr);

			#ifndef NDEBUG
				std::fill(ptr, ptr+26, VoxelVertex{uint64_t(-1)});
			#endif

			Mesh3D::GetVertexNeighbors_SIMD<Period>(key, reinterpret_cast<uint64_t*>(ptr));

			#ifndef NDEBUG
				GUTIL_ASSERT(std::find(ptr, ptr+26, VoxelVertex{uint64_t(-1)})==ptr+26);
			#endif

			return ptr;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint8_t local_vertex_number_simd(uint64_t el_key) const noexcept {
			GUTIL_ASSERT(Mesh3D::IsValid<Period>(el_key) && Mesh3D::IsElement(el_key));
			GUTIL_ASSERT(depth()==Mesh3D::Depth(el_key));
			return Mesh3D::IsMorton(el_key) ? Mesh3D::GetLocalVertexNumberMorton_SIMD<Period>(el_key, key) : 
								Mesh3D::GetLocalVertexNumberCartesian_SIMD<Period>(el_key, key);
		}

		/////////////////////////////////////////////////////////////
		/// Coordinates
		/////////////////////////////////////////////////////////////
		template<typename T=double>
		[[nodiscard]] constexpr gutil::Point<3,T> normalized_coordinate() const noexcept {
			GUTIL_ASSERT(is_valid());
			return Mesh3D::NormalizedCoordinate<T>(key);
		}

		[[nodiscard]] constexpr VoxelVertex reduced_key() const noexcept {
			GUTIL_ASSERT(is_valid());
			return VoxelVertex{Mesh3D::ReducedVertex(key)};
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t reduced_key_simd() const noexcept {
			return Mesh3D::ReducedVertex_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		constexpr void reduced_key_simd_in_place() noexcept {
			key = Mesh3D::ReducedVertex_SIMD(key);
		}

		[[nodiscard]] constexpr bool is_same_coord(VoxelVertex other) const noexcept {
			GUTIL_ASSERT(is_valid() && other.is_valid());
			return reduced_key() == other.reduced_key();
		}
	};

	template<uint8_t Period>
	inline std::string to_string(VoxelVertex<Period> dof) {
		return "VoxelVertex<" + std::to_string(Period) + ">{" 
				+ std::to_string(dof.depth()) + ", " + std::to_string(dof.i())
				+ ", " + std::to_string(dof.j()) + ", " + std::to_string(dof.k()) + "}";
	}

	template<uint8_t Period>
	std::ostream& operator<<(std::ostream& os, VoxelVertex<Period> dof) {
		return os << to_string(dof);
	}




}}