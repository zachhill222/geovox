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
	template<uint8_t Period> requires(Period<8)
	[[nodiscard]] constexpr std::array<VoxelVertex<Period>,8> VoxelElement<Period>::vertices() const noexcept {
		GUTIL_ASSERT(is_valid());
		return Mesh3D::GetVerticesOfElement<VoxelVertex<Period>,Period>(key);
	}
	template<uint8_t Period> requires(Period<8)
	[[nodiscard]] constexpr VoxelVertex<Period> VoxelElement<Period>::vertex(int i) const noexcept {
		GUTIL_ASSERT(is_valid());
		GUTIL_ASSERT(0<=i && i<8);
		return vertices()[i];
	}

	GUTIL_DECLARE_SIMD()
	template<uint8_t Period> requires(Period<8)
	[[maybe_unused]] VoxelVertex<Period>* VoxelElement<Period>::vertices_simd(VoxelVertex<Period>* ptr) const noexcept {
		GUTIL_ASSERT(ptr);

		#ifndef NDEBUG
			std::fill(ptr, ptr+8, VoxelVertex<Period>{uint64_t(-1)});
		#endif

		Mesh3D::GetVerticesOfElement_SIMD<Period>(Mesh3D::DecodeElement(key), reinterpret_cast<uint64_t*>(ptr));

		#ifndef NDEBUG
			GUTIL_ASSERT(std::find(ptr, ptr+8, VoxelVertex<Period>{uint64_t(-1)})==ptr+8);
		#endif
		return ptr;
	}


	//////////////////////////////////////////////////////////////
	/// Complete VoxelVertex<Period> interface
	//////////////////////////////////////////////////////////////
	template<uint8_t Period> requires(Period<8)
	[[nodiscard]] constexpr std::array<VoxelElement<Period>,8> VoxelVertex<Period>::elements() const noexcept {
		return Mesh3D::GetElementsOfVertex<VoxelElement<Period>,Period>(key);
	}
	template<uint8_t Period> requires(Period<8)
	[[nodiscard]] constexpr VoxelElement<Period> VoxelVertex<Period>::element(int i) const noexcept {
		GUTIL_ASSERT(0<=i && i<8);
		return elements()[i];
	}

	GUTIL_DECLARE_SIMD()
	template<uint8_t Period> requires(Period<8)
	[[maybe_unused]] VoxelElement<Period>* VoxelVertex<Period>::elements_simd(VoxelElement<Period>* ptr) const noexcept {
		GUTIL_ASSERT(ptr);

		#ifndef NDEBUG
			std::fill(ptr, ptr+8, VoxelElement<Period>{uint64_t(-1)});
		#endif

		Mesh3D::GetElementsOfVertex_SIMD<Period>(key, reinterpret_cast<uint64_t*>(ptr));

		#ifndef NDEBUG
			GUTIL_ASSERT(std::find(ptr, ptr+8, VoxelElement<Period>{uint64_t(-1)})==ptr+8);
		#endif

		return ptr;
	}



}}