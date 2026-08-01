#pragma once

#include "gutil.hpp"
#include "simd_keys/keyed_object.hpp"
#include "simd_keys/mesh_key.hpp"

#include <cstdint>
#include <vector>
#include <span>
#include <concepts>
#include <type_traits>
#include <iterator>


namespace GV {
	namespace Keys {
		

		

		using VoxelVertex  = KeyedObject<Mesh3D::VERTEX_FLAG, typename Mesh3D::Tag>;
		using VoxelElement = KeyedObject<Mesh3D::ELEMENT_FLAG, typename Mesh3D::Tag>;
		using VoxelFace    = KeyedObject<Mesh3D::FACE_FLAG, typename Mesh3D::Tag>;
		using VoxelEdge    = KeyedObject<Mesh3D::EDGE_FLAG, typename Mesh3D::Tag>;

		template<typename T>
		concept MeshFeatureType = std::same_as<T,VoxelVertex> || std::same_as<T,VoxelElement>;
		//TODO: add support for edges and faces


		////////////////////////////////////////////////////////////
		/// A container to hold uint64_t and treat them like a mesh
		/// feature with *some* type safety. This container is designed
		/// to hold compressed 'active only' features.
		////////////////////////////////////////////////////////////
		template<typename KeyType>
		struct CompressedKeyVector {
			
			////////////////////////////////////////////////////////
			/// Track the feature flag. only vertices and elements
			/// are supported for now.
			////////////////////////////////////////////////////////
			using value_type = KeyType;


			////////////////////////////////////////////////////////
			/// Track the data and forward standard interface to the vector
			////////////////////////////////////////////////////////
			std::vector<uint64_t> data;
			[[nodiscard]] constexpr size_t size() const noexcept { return data.size(); }
			void clear() noexcept { data.clear(); }
			void resize(size_t n, uint64_t val=0) noexcept { data.resize(n,val); }
			void shrink_to_fit() noexcept { data.shrink_to_fit(); }
			
			auto begin()  noexcept       { return data.begin();  }
			auto end()    noexcept       { return data.end();    }
			auto cbegin() const noexcept { return data.cbegin(); }
			auto cend()   const noexcept { return data.cend();   }
			auto begin()  const noexcept { return data.cbegin(); }
			auto end()    const noexcept { return data.cend();   }

			[[nodiscard]] constexpr uint64_t operator[](size_t idx) const noexcept {
				GUTIL_ASSERT(idx<size());
				return data[idx];
			}
			[[nodiscard]] constexpr uint64_t& operator[](size_t idx) noexcept {
				GUTIL_ASSERT(idx<size());
				return data[idx];
			}

			////////////////////////////////////////////////////////
			/// Get raw pointers for SIMD
			////////////////////////////////////////////////////////
			[[nodiscard]] uint64_t* raw() noexcept { return data.data(); }
			[[nodiscard]] const uint64_t* raw() const noexcept { return data.data(); }
		}


		////////////////////////////////////////////////////////////
		/// A non-owning container to hold uint64_t and treat them like a mesh
		/// feature with *some* type safety. This container is designed
		/// to be a view (span) into some other container.
		////////////////////////////////////////////////////////////
		template<typename KeyType>
		struct CompressedKeySpan {
			
			////////////////////////////////////////////////////////
			/// Track the feature flag. only vertices and elements
			/// are supported for now.
			////////////////////////////////////////////////////////
			using value_type = FeatureType;
			using pointer_type = std::conditional_t<std::same_as<std::remove_const_t<FeatureType>,FeatureType>, uint64_t*, const uint64_t*>;
			using raw_value_type = std::conditional_t<std::same_as<std::remove_const_t<FeatureType>,FeatureType>, uint64_t, const uint64_t>;

			////////////////////////////////////////////////////////
			/// Construct from iterators or a ponter and size
			////////////////////////////////////////////////////////
			template<std::contiguous_iterator I>
			constexpr CompressedKeySpan(I begin, I end) : 
					ptr{std::to_address(begin)}, 
					N{std::static_cast<size_t>(std::distance(begin,end))} {}

			constexpr CompressedKeySpan(pointer_type ptr, size_t N) : ptr{ptr}, N{N} {
				GUTIL_ASSERT( (ptr==nullptr && N==0) || (ptr!=nullptr && N>0) )
			}

			////////////////////////////////////////////////////////
			/// Track the data and implement a standard interface
			////////////////////////////////////////////////////////
			pointer_type ptr{nullptr};
			size_t N{0};

			[[nodiscard]] size_t size() const noexcept { return N; }
			constexpr void clear() noexcept { ptr=nullptr; N=0; }
			
			constexpr pointer_type begin()     noexcept       { return ptr;   }
			constexpr pointer_type end()       noexcept       { return ptr+N; }
			constexpr const uint64_t* cbegin() const noexcept { return ptr;   }
			constexpr const uint64_t* cend()   const noexcept { return ptr+N; }
			constexpr const uint64_t* begin()  const noexcept { return ptr;   }
			constexpr const uint64_t* end()    const noexcept { return ptr+N; }

			[[nodiscard]] constexpr uint64_t operator[](size_t idx) const noexcept {
				GUTIL_ASSERT(idx<size());
				return ptr[idx];
			}
			[[nodiscard]] constexpr uint64_t& operator[](size_t idx) noexcept {
				GUTIL_ASSERT(idx<size());
				return ptr[idx];
			}

			////////////////////////////////////////////////////////
			/// Get raw pointers for SIMD
			////////////////////////////////////////////////////////
			[[nodiscard]] constexpr pointer_type raw() noexcept { return ptr; }
			[[nodiscard]] constexpr const uint64_t* raw() const noexcept { ptr; }


			////////////////////////////////////////////////////////
			/// Convert to std::span for compatibility
			////////////////////////////////////////////////////////
			[[nodiscard]] constexpr std::span<raw_value_type> as_span() noexcept { return std::span<raw_value_type>{ptr,N}; }
			[[nodiscard]] constexpr std::span<const uint64_t> as_span() const noexcept { return std::span<const uint64_t>{ptr,N}; }
		}



	}






}



