#pragma once

#include <cstdint>
#include <type_traits>

#include "mesh/keys/voxel_key_base.hpp"
#include "mesh/keys/voxel_key_element.hpp"
#include "mesh/keys/voxel_key_vertex.hpp"
#include "mesh/keys/voxel_key_face.hpp"

#include "util/concepts.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//concepts
	template<typename T>
	concept VoxelElementKeyType = requires {T::MAX_DEPTH; T::BC_FLAG;} &&
		std::same_as<T, VoxelElementKey<T::MAX_DEPTH, T::BC_FLAG>>;
	
	static_assert(VoxelElementKeyType<VoxelElementKey<>>);
	static_assert(!VoxelElementKeyType<VoxelVertexKey<>>);
	static_assert(!VoxelElementKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelVertexKeyType = requires {T::MAX_DEPTH; T::BC_FLAG;} &&
		std::same_as<T, VoxelVertexKey<T::MAX_DEPTH, T::BC_FLAG>>;
	static_assert(!VoxelVertexKeyType<VoxelElementKey<>>);
	static_assert(VoxelVertexKeyType<VoxelVertexKey<>>);
	static_assert(!VoxelVertexKeyType<VoxelFaceKey<>>);

	template<typename T>
	concept VoxelFaceKeyType = requires {T::MAX_DEPTH; T::BC_FLAG;} &&
		std::same_as<T, VoxelFaceKey<T::MAX_DEPTH, T::BC_FLAG>>;
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
	concept VoxelFeatureCompatible = (A::MAX_DEPTH==B::MAX_DEPTH) && VoxelKeyType<A> && VoxelKeyType<B>;

	template<typename A, typename B>
	concept VoxelEquivFeature = VoxelFeatureCompatible<A,B> &&
		( 	(VoxelElementKeyType<A> && VoxelElementKeyType<B>) ||
			(VoxelVertexKeyType<A> && VoxelVertexKeyType<B>) ||
			(VoxelFaceKeyType<A> && VoxelFaceKeyType<B>) );

	//check if a function is a predicate on a given mesh feature
	//note that std::nullptr_t and void can be used as flags for "always true" in funcions
	template<typename Predicate, typename Feature>
	concept VoxelElementPredicate = VoxelElementKeyType<Feature> && 
		(NULLPTR_T<Predicate> || VOID_T<Predicate> || std::is_invocable_r_v<bool, Predicate, Feature>);

	template<typename Predicate, typename Feature>
	concept VoxelVertexPredicate = VoxelVertexKeyType<Feature> && 
		(NULLPTR_T<Predicate> || VOID_T<Predicate> || std::is_invocable_r_v<bool, Predicate, Feature>);

	template<typename Predicate, typename Feature>
	concept VoxelFacePredicate = VoxelFaceKeyType<Feature> && 
		(NULLPTR_T<Predicate> || VOID_T<Predicate> || std::is_invocable_r_v<bool, Predicate, Feature>);


	//check if a function is an action on a given mesh feature
	//must be callable on a feature passed by value or reference and return a void
	template<typename Action, typename Feature>
	concept VoxelElementAction = VoxelElementKeyType<Feature> && (std::same_as<std::invoke_result_t<Action,Feature>,void> || std::same_as<std::invoke_result_t<Action,Feature&>,void>);

	template<typename Action, typename Feature>
	concept VoxelVertexAction = VoxelVertexKeyType<Feature> && (std::same_as<std::invoke_result_t<Action,Feature>,void> || std::same_as<std::invoke_result_t<Action,Feature&>,void>);

	template<typename Action, typename Feature>
	concept VoxelFaceAction = VoxelFaceKeyType<Feature> && (std::same_as<std::invoke_result_t<Action,Feature>,void> || std::same_as<std::invoke_result_t<Action,Feature&>,void>);

	//check if a function is a lookup on a given mesh feature
	//must be callable on a feature passed by value and return some data
	template<typename Lookup, typename Feature>
	concept VoxelElementLookup = VoxelElementKeyType<Feature> && !std::same_as<std::invoke_result_t<Lookup,Feature>,void>;

	template<typename Lookup, typename Feature>
	concept VoxelVertexLookup = VoxelVertexKeyType<Feature> && !std::same_as<std::invoke_result_t<Lookup,Feature>,void>;

	template<typename Lookup, typename Feature>
	concept VoxelFaceLookup = VoxelFaceKeyType<Feature> && !std::same_as<std::invoke_result_t<Lookup,Feature>,void>;


	//useful standalone functions
	template<VoxelKeyType Key_t>
	constexpr uint64_t total_possible(const uint64_t max_depth) {
		return Key_t::depth_linear_start(max_depth+1);
	}


	/// ELEMENT IMPLEMENTATIONS
	template<uint64_t MAX_DEPTH, uint64_t BC>
	inline constexpr std::array<VoxelVertexKey<MAX_DEPTH,BC>,8> VoxelElementKey<MAX_DEPTH,BC>::vertices() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth();
		return {
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj,  kk  },
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj,  kk  },
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj+1,kk  },
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj+1,kk  },
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj,  kk+1},
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj,  kk+1},
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj+1,kk+1},
			VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj+1,kk+1}
		};
	}
	
	template<uint64_t MAX_DEPTH, uint64_t BC>
	inline constexpr std::array<VoxelFaceKey<MAX_DEPTH,BC>,6> VoxelElementKey<MAX_DEPTH,BC>::faces() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth();
		return {
			VoxelFaceKey<MAX_DEPTH,BC>{0, dd, ii  , jj  , kk  },
			VoxelFaceKey<MAX_DEPTH,BC>{1, dd, ii  , jj  , kk  },
			VoxelFaceKey<MAX_DEPTH,BC>{2, dd, ii  , jj  , kk  },
			VoxelFaceKey<MAX_DEPTH,BC>{0, dd, ii+1, jj  , kk  },
			VoxelFaceKey<MAX_DEPTH,BC>{1, dd, ii  , jj+1, kk  },
			VoxelFaceKey<MAX_DEPTH,BC>{2, dd, ii  , jj  , kk+1}
		};
	}

	/// FACE IMPLEMENTATIONS
	template<uint64_t MAX_DEPTH, uint64_t BC>
	constexpr std::array<VoxelElementKey<MAX_DEPTH,BC>,2> VoxelFaceKey<MAX_DEPTH,BC>::elements() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth(), aa=axis();
		switch (aa) {
		case 0: return {
				VoxelElementKey<MAX_DEPTH,BC>{dd,ii,jj,kk},
				VoxelElementKey<MAX_DEPTH,BC>{dd,ii+1,jj,kk}
			};
		case 1: return {
				VoxelElementKey<MAX_DEPTH,BC>{dd,ii,jj,kk},
				VoxelElementKey<MAX_DEPTH,BC>{dd,ii,jj+1,kk}
			};
		case 2: return {
				VoxelElementKey<MAX_DEPTH,BC>{dd,ii,jj,kk},
				VoxelElementKey<MAX_DEPTH,BC>{dd,ii,jj,kk+1}
			};
		default: return {};
		}
	}

	template<uint64_t MAX_DEPTH, uint64_t BC>
	constexpr std::array<VoxelVertexKey<MAX_DEPTH,BC>,4> VoxelFaceKey<MAX_DEPTH,BC>::vertices() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth(), aa=axis();
		switch (aa) {
		case 0: return {
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,jj,  kk  },
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,jj+1,kk  },
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,jj,  kk+1},
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,jj+1,kk+1}
			};
		case 1: return {
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj,kk  },
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj,kk  },
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj,kk+1},
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj,kk+1}
			};
		case 2: return {
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj,  kk},
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj,  kk},
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii,  jj+1,kk},
				VoxelVertexKey<MAX_DEPTH,BC>{dd,ii+1,jj+1,kk}
			};
		default: return {};
		}
	}

	
	/// VERTEX IMPLEMENTATIONS
	template<uint64_t MAX_DEPTH, uint64_t BC>
	constexpr std::array<VoxelElementKey<MAX_DEPTH,BC>,8> VoxelVertexKey<MAX_DEPTH,BC>::elements() const
	{
		const uint64_t ii=i(), jj=j(), kk=k(), dd=depth();

		//underflow to upper bound is not handled by the periodic constructor
		const uint64_t me = uint64_t{1} << dd; //2^d elements per axis
		
		//note (2^64 - 1) % (2^d) = -1 = 2^d -1 in modulo arithmetic, with the last the representation as an uint64
		const uint64_t im1 = PX ? (ii-1)%me : ii-1;
		const uint64_t jm1 = PY ? (jj-1)%me : jj-1;
		const uint64_t km1 = PZ ? (kk-1)%me : kk-1;

		return {
			VoxelElementKey<MAX_DEPTH,BC>{dd, im1, jm1, km1},
			VoxelElementKey<MAX_DEPTH,BC>{dd, im1, jm1, kk },
			VoxelElementKey<MAX_DEPTH,BC>{dd, im1, jj,  km1},
			VoxelElementKey<MAX_DEPTH,BC>{dd, im1, jj,  kk },
			VoxelElementKey<MAX_DEPTH,BC>{dd, ii,  jm1, km1},
			VoxelElementKey<MAX_DEPTH,BC>{dd, ii,  jm1, kk },
			VoxelElementKey<MAX_DEPTH,BC>{dd, ii,  jj,  km1},
			VoxelElementKey<MAX_DEPTH,BC>{dd, ii,  jj,  kk }
		};
	}
}