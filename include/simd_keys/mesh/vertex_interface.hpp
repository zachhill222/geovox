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
struct VoxelElement;


/////////////////////////////////////////////////////////////////
/// Extend KeyedObject for mesh elements.
/// Morton encoding is the "cononical" representation.
/////////////////////////////////////////////////////////////////
struct VoxelVertex : public KeyedObject<Mesh3D::VERTEX_FLAG> {


	/////////////////////////////////////////////////////////////
	/// Constructors and factories
	/////////////////////////////////////////////////////////////
	using BASE = KeyedObject<Mesh3D::VERTEX_FLAG>;
	using BASE::BASE;
	using BASE::key;

	GUTIL_DECLARE_SIMD()
	static constexpr VoxelVertex MakeFromIndex(uint64_t index) noexcept {
		return VoxelVertex{Mesh3D::VertexFromGlobalIndex_SIMD(index)};
	}

	GUTIL_DECLARE_SIMD()
	constexpr VoxelVertex(uint64_t depth, uint64_t ii, uint64_t jj, uint64_t kk) noexcept :
		BASE{Mesh3D::MakeVertex(depth,ii,jj,kk)} {}

	
	/////////////////////////////////////////////////////////////
	/// Queries
	/////////////////////////////////////////////////////////////
	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr bool exists() const noexcept { return Mesh3D::Exists(key); }
	[[nodiscard]] constexpr bool is_valid() const noexcept { return Mesh3D::IsValid(key); }

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr uint64_t linear_index_simd() const noexcept { return Mesh3D::GlobalVertexIndex_SIMD(key); }
	[[nodiscard]] constexpr uint64_t linear_index() const noexcept {
		GUTIL_ASSERT(is_valid());
		return Mesh3D::GlobalVertexIndex(key);
	}
	[[nodiscard]] constexpr uint64_t depth_linear_index() const noexcept {
		GUTIL_ASSERT(is_valid())
		return Mesh3D::GlobalVertexIndex_SIMD(key) - Mesh3D::VerticesBelowDepth(Mesh3D::Depth(key));
	}

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr uint64_t i_simd() const noexcept { return Mesh3D::IndexI_SIMD(key); }
	[[nodiscard]] constexpr uint64_t i() const noexcept { return Mesh3D::IndexI(key); }

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr uint64_t j_simd() const noexcept { return Mesh3D::IndexJ_SIMD(key); }
	[[nodiscard]] constexpr uint64_t j() const noexcept { return Mesh3D::IndexJ(key); }

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr uint64_t k_simd() const noexcept { return Mesh3D::IndexK_SIMD(key); }
	[[nodiscard]] constexpr uint64_t k() const noexcept { return Mesh3D::IndexK(key); }

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr uint64_t depth() const noexcept { return Mesh3D::Depth(key); }
	
	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr uint8_t depth_u8() const noexcept { return Mesh3D::Depth_u8(key); }

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] static constexpr uint64_t vertices_below_depth(uint64_t dd) noexcept {
		return Mesh3D::VerticesBelowDepth(dd);
	}

	/////////////////////////////////////////////////////////////
	/// Hierarchy
	/////////////////////////////////////////////////////////////
	[[nodiscard]] constexpr VoxelVertex child() const noexcept {
		GUTIL_ASSERT(is_valid());
		return VoxelVertex{Mesh3D::VertexChild(key)};
	}

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr VoxelVertex child_simd() const noexcept {
		return VoxelVertex{Mesh3D::VertexChild(key)};
	}

	[[nodiscard]] constexpr VoxelVertex parent() const noexcept {
		GUTIL_ASSERT(is_valid());
		return VoxelVertex{Mesh3D::VertexParent(key)};
	}

	GUTIL_DECLARE_SIMD()
	[[nodiscard]] constexpr VoxelVertex parent_simd() const noexcept {
		return VoxelVertex{Mesh3D::VertexParent(key)};
	}


	/////////////////////////////////////////////////////////////
	/// Adjacency
	/// 
	/// Note that the element query will return elements in cartesian
	/// rather than morton form.
	/////////////////////////////////////////////////////////////
	[[nodiscard]] constexpr std::array<VoxelElement,8> elements() const noexcept;
	[[nodiscard]] constexpr VoxelElement element(int i) const noexcept;

	GUTIL_DECLARE_SIMD()
	[[maybe_unused]] VoxelElement* elements_simd(VoxelElement* ptr) const noexcept;

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
		Mesh3D::GetVertexNeighbors_SIMD(key, reinterpret_cast<uint64_t*>(ptr));
		return ptr;
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
	[[nodiscard]] constexpr VoxelVertex reduced_key_simd() const noexcept {
		return VoxelVertex{Mesh3D::ReducedVertex_SIMD(key)};
	}

	[[nodiscard]] constexpr bool is_same_coord(VoxelVertex other) const noexcept {
		GUTIL_ASSERT(is_valid() && other.is_valid());
		return reduced_key() == other.reduced_key();
	}
};






}}