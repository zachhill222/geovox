#pragma once

#include <cstdint>
#include <type_traits>

#include "mesh/keys/voxel_key_base.hpp"
#include "mesh/keys/voxel_key_element.hpp"
#include "mesh/keys/voxel_key_vertex.hpp"
#include "mesh/keys/voxel_key_face.hpp"

namespace GV
{
	//concepts
	template<typename T>
	concept VoxelElementKeyType = requires {T::I_W; T::BC_FLAG; T::MORTON;} &&
		std::same_as<T, VoxelElementKey<T::I_W, T::BC_FLAG, T::MORTON>>;
	
	static_assert(VoxelElementKeyType<VoxelElementKey<>>);
	static_assert(!VoxelElementKeyType<VoxelVertexKey<>>);
	static_assert(!VoxelElementKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelVertexKeyType = requires {T::I_W; T::BC_FLAG; T::MORTON;} &&
		std::same_as<T, VoxelVertexKey<T::I_W, T::BC_FLAG, T::MORTON>>;
	static_assert(!VoxelVertexKeyType<VoxelElementKey<>>);
	static_assert(VoxelVertexKeyType<VoxelVertexKey<>>);
	static_assert(!VoxelVertexKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelFaceKeyType = requires {T::I_W; T::BC_FLAG; T::MORTON;} &&
		std::same_as<T, VoxelFaceKey<T::I_W, T::BC_FLAG, T::MORTON>>;
	static_assert(!VoxelFaceKeyType<VoxelElementKey<>>);
	static_assert(!VoxelFaceKeyType<VoxelVertexKey<>>);
	static_assert(VoxelFaceKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelKeyType = VoxelElementKeyType<T> || VoxelVertexKeyType<T> || VoxelFaceKeyType<T>;
	static_assert(VoxelKeyType<VoxelElementKey<>>);
	static_assert(VoxelKeyType<VoxelVertexKey<>>);
	static_assert(VoxelKeyType<VoxelFaceKey<>>);

	//check if two mesh features are compatable
	//they are compatable if they have the same coloring and index width
	//this allows one to check if features are the same up to boundary conditions
	//this can be helpful as DOFs may have feature keys with boundary conditions
	//while mesh features dont
	template<typename A, typename B>
	concept VoxelFeatureCompatible = (A::I_W==B::I_W) && (A::MORTON==B::MORTON) &&
		VoxelKeyType<A> && VoxelKeyType<B>;

	template<typename A, typename B>
	concept VoxelEquivFeature = VoxelFeatureCompatible<A,B> &&
		( 	(VoxelElementKeyType<A> && VoxelElementKeyType<B>) ||
			(VoxelVertexKeyType<A> && VoxelVertexKeyType<B>) ||
			(VoxelFaceKeyType<A> && VoxelFaceKeyType<B>) );


	//useful standalone functions
	template<VoxelKeyType Key_t>
	constexpr uint64_t total_possible(const uint64_t max_depth) {
		return Key_t::depth_linear_start(max_depth+1);
	}


	/// ELEMENT IMPLEMENTATIONS
	template<uint64_t I_W, uint64_t BC, bool MORTON>
	inline constexpr std::array<VoxelVertexKey<I_W,BC,MORTON>,8> VoxelElementKey<I_W,BC,MORTON>::vertices() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth();
		return {
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj,  kk  },
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj,  kk  },
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj+1,kk  },
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj+1,kk  },
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj,  kk+1},
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj,  kk+1},
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj+1,kk+1},
			VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj+1,kk+1}
		};
	}
	
	template<uint64_t I_W, uint64_t BC, bool MORTON>
	inline constexpr std::array<VoxelFaceKey<I_W,BC,MORTON>,6> VoxelElementKey<I_W,BC,MORTON>::faces() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth();
		return {
			VoxelFaceKey<I_W,BC,MORTON>{0, dd, ii  , jj  , kk  },
			VoxelFaceKey<I_W,BC,MORTON>{1, dd, ii  , jj  , kk  },
			VoxelFaceKey<I_W,BC,MORTON>{2, dd, ii  , jj  , kk  },
			VoxelFaceKey<I_W,BC,MORTON>{0, dd, ii+1, jj  , kk  },
			VoxelFaceKey<I_W,BC,MORTON>{1, dd, ii  , jj+1, kk  },
			VoxelFaceKey<I_W,BC,MORTON>{2, dd, ii  , jj  , kk+1}
		};
	}

	/// FACE IMPLEMENTATIONS
	template<uint64_t I_W, uint64_t BC, bool MORTON>
	constexpr std::array<VoxelElementKey<I_W,BC,MORTON>,2> VoxelFaceKey<I_W,BC,MORTON>::elements() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth(), aa=axis();
		switch (aa) {
		case 0: return {
				VoxelElementKey<I_W,BC,MORTON>{dd,ii,jj,kk},
				VoxelElementKey<I_W,BC,MORTON>{dd,ii+1,jj,kk}
			};
		case 1: return {
				VoxelElementKey<I_W,BC,MORTON>{dd,ii,jj,kk},
				VoxelElementKey<I_W,BC,MORTON>{dd,ii,jj+1,kk}
			};
		case 2: return {
				VoxelElementKey<I_W,BC,MORTON>{dd,ii,jj,kk},
				VoxelElementKey<I_W,BC,MORTON>{dd,ii,jj,kk+1}
			};
		default: return {};
		}
	}

	template<uint64_t I_W, uint64_t BC, bool MORTON>
	constexpr std::array<VoxelVertexKey<I_W,BC,MORTON>,4> VoxelFaceKey<I_W,BC,MORTON>::vertices() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth(), aa=axis();
		switch (aa) {
		case 0: return {
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,jj,  kk  },
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,jj+1,kk  },
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,jj,  kk+1},
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,jj+1,kk+1}
			};
		case 1: return {
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj,kk  },
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj,kk  },
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj,kk+1},
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj,kk+1}
			};
		case 2: return {
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj,  kk},
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj,  kk},
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii,  jj+1,kk},
				VoxelVertexKey<I_W,BC,MORTON>{dd,ii+1,jj+1,kk}
			};
		default: return {};
		}
	}

	
	/// VERTEX IMPLEMENTATIONS
	template<uint64_t I_W, uint64_t BC, bool MORTON>
	constexpr std::array<VoxelElementKey<I_W,BC,MORTON>,8> VoxelVertexKey<I_W,BC,MORTON>::elements() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth();

		//underflow to upper bound is not handled by the periodic constructor
		const uint64_t me = uint64_t{1} << dd; //2^d elements per axis
		
		//note (2^64 - 1) % (2^d) = -1 = 2^d -1 in modulo arithmetic, with the last the representation as an uint64
		const uint64_t im1 = PX ? (ii-1)%me : ii-1;
		const uint64_t jm1 = PY ? (jj-1)%me : jj-1;
		const uint64_t km1 = PZ ? (kk-1)%me : kk-1;
		
		return {
			VoxelElementKey<I_W,BC,MORTON>{dd, im1, jm1, km1},
			VoxelElementKey<I_W,BC,MORTON>{dd, im1, jm1, kk },
			VoxelElementKey<I_W,BC,MORTON>{dd, im1, jj,  km1},
			VoxelElementKey<I_W,BC,MORTON>{dd, im1, jj,  kk },
			VoxelElementKey<I_W,BC,MORTON>{dd, ii,  jm1, km1},
			VoxelElementKey<I_W,BC,MORTON>{dd, ii,  jm1, kk },
			VoxelElementKey<I_W,BC,MORTON>{dd, ii,  jj,  km1},
			VoxelElementKey<I_W,BC,MORTON>{dd, ii,  jj,  kk }
		};
	}
}