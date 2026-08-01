#pragma once

#include "gutil.hpp"
#include "simd_keys/mesh/mesh_key_implementation.hpp"
#include "simd_keys/mesh/element_interface.hpp"
#include "simd_keys/mesh/vertex_interface.hpp"


namespace GV {
namespace Keys {


	//////////////////////////////////////////////////////////////
	/// Complete VoxelElement interface
	//////////////////////////////////////////////////////////////
	[[nodiscard]] constexpr std::array<VoxelVertex,8> VoxelElement::vertices() const noexcept {
		GUTIL_ASSERT(is_valid());
		return Mesh3D::GetVerticesOfElement<VoxelVertex>(key);
	}
	[[nodiscard]] constexpr VoxelVertex VoxelElement::vertex(int i) const noexcept {
		GUTIL_ASSERT(is_valid());
		GUTIL_ASSERT(0<=i && i<8);
		return vertices()[i];
	}

	GUTIL_DECLARE_SIMD()
	[[maybe_unused]] VoxelVertex* VoxelElement::vertices_simd(VoxelVertex* ptr) const noexcept {
		GUTIL_ASSERT(ptr);
		Mesh3D::GetVerticesOfElement_SIMD(key, reinterpret_cast<uint64_t*>(ptr));
		return ptr;
	}


	//////////////////////////////////////////////////////////////
	/// Complete VoxelVertex interface
	//////////////////////////////////////////////////////////////
	[[nodiscard]] constexpr std::array<VoxelElement,8> VoxelVertex::elements() const noexcept {
		return Mesh3D::GetElementsOfVertex<VoxelElement>(key);
	}
	[[nodiscard]] constexpr VoxelElement VoxelVertex::element(int i) const noexcept {
		GUTIL_ASSERT(0<=i && i<8);
		return elements()[i];
	}

	GUTIL_DECLARE_SIMD()
	[[maybe_unused]] VoxelElement* VoxelVertex::elements_simd(VoxelElement* ptr) const noexcept {
		GUTIL_ASSERT(ptr);
		Mesh3D::GetElementsOfVertex_SIMD(key, reinterpret_cast<uint64_t*>(ptr));
		return ptr;
	}



}}