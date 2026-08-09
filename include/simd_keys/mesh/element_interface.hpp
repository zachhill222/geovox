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
	struct VoxelVertex;


	/////////////////////////////////////////////////////////////////
	/// Extend KeyedObject for mesh elements.
	/// Morton encoding is the "cononical" representation.
	/////////////////////////////////////////////////////////////////
	template<uint8_t Period=0> requires(Period<8)
	struct VoxelElement : public KeyedObject<Mesh3D::ELEMENT_FLAG> {


		/////////////////////////////////////////////////////////////
		/// Constructors and factories
		/////////////////////////////////////////////////////////////
		using BASE = KeyedObject<Mesh3D::ELEMENT_FLAG>;
		using BASE::BASE;
		using BASE::ID;
		using BASE::key;
		static constexpr uint8_t PERIOD = Period;

		static constexpr VoxelElement MakeFromIndex(uint64_t index) noexcept {
			return VoxelElement{Mesh3D::ElementFromGlobalIndex_SIMD(index)};
		}

		static constexpr VoxelElement None() noexcept {return VoxelElement{0};}

		constexpr VoxelElement(uint64_t depth, uint64_t morton_index) noexcept :
			BASE{Mesh3D::MakeElement(depth, morton_index)} {}

		constexpr VoxelElement(uint64_t depth, uint64_t ii, uint64_t jj, uint64_t kk) noexcept :
			BASE{Mesh3D::MakeElement(depth,ii,jj,kk)} {}
			

		/////////////////////////////////////////////////////////////
		/// Convert between period types
		/////////////////////////////////////////////////////////////
		template<uint8_t OtherPeriod> requires(OtherPeriod<8)
		[[nodiscard]] explicit constexpr operator VoxelElement<OtherPeriod>() const noexcept {
			//note that elements being periodic only affects their relation to vertices
			//and neighbors. It does not affect the encoding of any element.
			return VoxelElement<OtherPeriod>{key};
		}

		
		/////////////////////////////////////////////////////////////
		/// Convert between encodings.
		/////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr VoxelElement encode() noexcept {
			return VoxelElement{ is_encoded() ? key : Mesh3D::EncodeElement_SIMD(key)};
		}
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t encode_simd() noexcept {
			return Mesh3D::EncodeElement_SIMD(key);
		}
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t encode_simd(uint64_t k) noexcept {
			return Mesh3D::EncodeElement_SIMD(k);
		}

		//morton->cartesian
		[[nodiscard]] constexpr VoxelElement decode() const noexcept {
			return VoxelElement{ is_encoded() ? Mesh3D::DecodeElement_SIMD(key) : key };
		}
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t decode_simd() const noexcept {
			return Mesh3D::DecodeElement_SIMD(key);
		}
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t decode_simd(uint64_t k) noexcept {
			return Mesh3D::IsMorton(k) ? Mesh3D::DecodeElement_SIMD(k) : k;
		}

		/// Color54 by pairity - no ring 1 colisions in a 2-1 mesh
		[[nodiscard]] constexpr uint64_t color() const noexcept {
			return is_encoded() ? Mesh3D::CartesianColor54_SIMD(Mesh3D::DecodeElement_SIMD(key)) :
								  Mesh3D::CartesianColor54_SIMD(key);
		}
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t color_simd() const noexcept {
			return Mesh3D::CartesianColor54_SIMD(key);
		}
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t color_simd(uint64_t k) noexcept {
			return Mesh3D::CartesianColor54_SIMD(k);
		}

		/////////////////////////////////////////////////////////////
		/// Queries
		/////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr bool exists() const noexcept { return Mesh3D::Exists(key); }
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr bool is_encoded() const noexcept { return Mesh3D::IsMorton(key); }
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr bool is_cartesian() const noexcept { return Mesh3D::IsCartesian(key); }
		[[nodiscard]] constexpr bool is_valid() const noexcept { return Mesh3D::IsValid(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t linear_index_simd() const noexcept { return Mesh3D::GlobalElementIndex_SIMD(key); }
		[[nodiscard]] constexpr uint64_t linear_index() const noexcept { return Mesh3D::GlobalElementIndex(key); }
		[[nodiscard]] constexpr uint64_t depth_linear_index() const noexcept { return Mesh3D::MortonIndex(key); }

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
		[[nodiscard]] constexpr uint64_t kji_pairity_simd() const noexcept {
			return is_encoded() ? (key&0b111) : ( ((k_simd()&1)<<2)|((j_simd()&1)<<1)|(i_simd()&1) );
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t depth() const noexcept { return Mesh3D::Depth(key); }
		
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint8_t depth_u8() const noexcept { return Mesh3D::Depth_u8(key); }

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t elements_below_depth(uint64_t dd) noexcept {
			return Mesh3D::ElementsBelowDepth(dd);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] static constexpr uint64_t total_possible(uint8_t dd) noexcept {
			return Mesh3D::ElementsBelowDepth(dd+1);
		}

		/////////////////////////////////////////////////////////////
		/// Index arithmetic
		///
		/// Note operator++ is designed to increment through the 8 child elements,
		/// but will work across the entire index region (all depths) so long as
		/// it is morton encoded. When the maximum valid index is exceeded, the check bit
		/// will flip and key.exists() will return false.
		///
		/// We allow the use of el++ and ++el similar to iterators.
		/////////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] constexpr VoxelElement& operator++() {
			GUTIL_ASSERT(is_encoded());
			++key; return *this;
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] constexpr VoxelElement operator++(int) {
			GUTIL_ASSERT(is_encoded());
			uint64_t cpy = key;
			++key; return VoxelElement{cpy};
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] constexpr VoxelElement& operator+=(uint64_t n) {
			GUTIL_ASSERT(is_encoded());
			key+=n; return *this;
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] constexpr VoxelElement operator+(uint64_t n) const {
			GUTIL_ASSERT(is_encoded());
			return VoxelElement{key+n};
		}


		/////////////////////////////////////////////////////////////
		/// Hierarchy
		///
		/// Note that when called at the max depth, m_child_0 = 0 so all
		/// returned children will fail a child.exists() check. In the convenience
		/// method, the returned children will have the same encoding as the parent.
		///
		/// To use hierarchy in simd, we must be encoded. Rather than providing a pointer
		/// to write the children to, we return the first child. parent.children()+4 is the same
		/// as parent.children()[4] from the convenience method (up to encoding of the convenience method)
		/////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr std::array<VoxelElement,8> children() const noexcept {
			GUTIL_ASSERT(is_valid());
			const uint64_t m_child_0 = Mesh3D::ElementChildStart_SIMD(Mesh3D::EncodeElement(key));
			return is_encoded() ? std::array<VoxelElement,8>{ VoxelElement{m_child_0}, VoxelElement{m_child_0+1}, VoxelElement{m_child_0+2},
									VoxelElement{m_child_0+3}, VoxelElement{m_child_0+4}, VoxelElement{m_child_0+5},
									VoxelElement{m_child_0+6}, VoxelElement{m_child_0+7} }
								: std::array<VoxelElement,8>{ VoxelElement{Mesh3D::DecodeElement(m_child_0)}, VoxelElement{Mesh3D::DecodeElement(m_child_0+1)}, VoxelElement{Mesh3D::DecodeElement(m_child_0+2)},
									VoxelElement{Mesh3D::DecodeElement(m_child_0+3)}, VoxelElement{Mesh3D::DecodeElement(m_child_0+4)}, VoxelElement{Mesh3D::DecodeElement(m_child_0+5)},
									VoxelElement{Mesh3D::DecodeElement(m_child_0+6)}, VoxelElement{Mesh3D::DecodeElement(m_child_0+7)} };
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] constexpr VoxelElement children_simd() const noexcept {
			return VoxelElement{Mesh3D::ElementChildStart_SIMD(key)};
		}

		[[nodiscard]] constexpr VoxelElement parent() const noexcept {
			GUTIL_ASSERT(is_valid());
			const uint64_t m_parent = Mesh3D::ElementParent_SIMD(Mesh3D::EncodeElement(key));
			return is_encoded() ? VoxelElement{m_parent} : Mesh3D::Exists(m_parent) 
								? VoxelElement{Mesh3D::DecodeElement(m_parent)} : None();
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] constexpr uint64_t parent_simd() const noexcept {
			return is_encoded() ? Mesh3D::ElementParent_SIMD(key) : Mesh3D::ElementParentCartesian_SIMD(key);
		}


		/////////////////////////////////////////////////////////////
		/// Adjacency
		///
		/// Note that vertices use cartesian encoding only, so decoding
		/// the elements is necessary to use these in simd.
		///
		/// Index arithmetic isn't as fast in cartesion coordinates,
		/// so a pointer to the begining of where the vertices go must
		/// be provided. For convenience, it is returned so el.vertices_simd(ptr)[3] is valid
		///
		/// Defined in joint_interface.hpp when the VoxelVertex type is known
		/////////////////////////////////////////////////////////////
		[[nodiscard]] constexpr std::array<VoxelVertex<Period>,8> vertices() const noexcept;
		[[nodiscard]] constexpr VoxelVertex<Period> vertex(int i) const noexcept;

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] VoxelVertex<Period>* vertices_simd(VoxelVertex<Period>* ptr) const noexcept;

		[[nodiscard]] constexpr std::array<VoxelElement,26> neighbors() const noexcept {
			GUTIL_ASSERT(is_valid());
			return Mesh3D::GetElementNeighbors<VoxelElement,Period>(key);
		}
		[[nodiscard]] constexpr VoxelElement neighbor(int i) const noexcept {
			GUTIL_ASSERT(is_valid());
			GUTIL_ASSERT(0<=i && i<26);
			return neighbors()[i];
		}

		GUTIL_DECLARE_SIMD()
		[[maybe_unused]] VoxelElement* neighbors_simd(VoxelElement* ptr) const noexcept {
			GUTIL_ASSERT(ptr);
			
			#ifndef NDEBUG
				std::fill(ptr, ptr+26, VoxelElement{uint64_t(-1)});
			#endif

			Mesh3D::GetElementNeighbors_SIMD<Period>(key, reinterpret_cast<uint64_t*>(ptr));

			#ifndef NDEBUG
				GUTIL_ASSERT(std::find(ptr, ptr+26, VoxelElement{uint64_t(-1)})==ptr+26);
			#endif
			return ptr;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] constexpr uint64_t siblings_simd() const noexcept {
			return Mesh3D::GetElementSiblingsMorton_SIMD(key);
		}

		[[nodiscard]] constexpr VoxelElement siblings() const noexcept {
			GUTIL_ASSERT(is_valid());
			GUTIL_ASSERT(is_encoded());
			return VoxelElement{Mesh3D::GetElementSiblingsMorton_SIMD(key)};
		}


		/////////////////////////////////////////////////////////////
		/// Coordinates
		/////////////////////////////////////////////////////////////
		template<typename T=double>
		[[nodiscard]] constexpr gutil::Point<3,T> normalized_center() const noexcept {
			GUTIL_ASSERT(is_valid())
			return Mesh3D::NormalizedCenter<T>(key);
		}
	};


	template<uint8_t Period>
	inline std::string to_string(VoxelElement<Period> dof) {
		return "VoxelElement<" + std::to_string(Period) + ">{" 
				+ std::to_string(dof.depth()) + ", " + std::to_string(dof.i())
				+ ", " + std::to_string(dof.j()) + ", " + std::to_string(dof.k()) + "}";
	}

	template<uint8_t Period>
	std::ostream& operator<<(std::ostream& os, VoxelElement<Period> dof) {
		return os << to_string(dof);
	}



}}