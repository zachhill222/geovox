#pragma once


#include "util/macros.hpp"
#include "mesh/keys/voxel_key_base.hpp"

#include <cstdint>
#include <cassert>
#include <array>

namespace GV {
	//define the element key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelVertexKey;

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelFaceKey;

	template<uint64_t MAX_DEPTH_=16, uint64_t BC=0>
	struct VoxelElementKey : public VoxelKey<3,0,MAX_DEPTH_+1> {
		static_assert(BC<8, "VoxelElementKey: invalid boundary condition. BC must be from 0 to 7.");

		//inherit constructors
		using BASE = VoxelKey<3,0,MAX_DEPTH_+1>;
		using BASE::BASE;

		//inherit the primary accessors
		using BASE::depth;
		using BASE::i;
		using BASE::j;
		using BASE::k;
		using BASE::set_depth;
		using BASE::set_i;
		using BASE::set_j;
		using BASE::set_k;
		using BASE::_data_;
		using BASE::Hash;

		//usually the fee bits mean a manual color is set
		GEOVOX_DECLARE_SIMD()
		inline constexpr uint64_t color() const {return BASE::free();}
		GEOVOX_DECLARE_SIMD()
		inline constexpr void set_color(const uint64_t clr) {BASE::set_free(clr);}
		static constexpr uint64_t NO_COLOR = BASE::F_M >> BASE::F_S;


		//define useful constants
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;
		static_assert(MAX_DEPTH_==MAX_DEPTH);

		//periodic conditions. the BC bits are stored on the other_nocompare field
		static constexpr uint64_t BC_FLAG = BC;
		static constexpr bool PX = BC&1; //periodic in i/x
		static constexpr bool PY = BC&2; //periodic in j/y
		static constexpr bool PZ = BC&4; //periodic in k/z

		//explicit conversion to the non-periodic type
		using NonPeriodicVariant = VoxelElementKey<MAX_DEPTH_,0>;
		explicit operator NonPeriodicVariant() const {return NonPeriodicVariant{_data_&~BASE::ON_M};}

		template<uint64_t OTHER_BC>
		using PeriodicVariant = VoxelElementKey<MAX_DEPTH_,OTHER_BC>;
		
		
		template<uint64_t OTHER_BC> requires (OTHER_BC<8)
		inline explicit operator PeriodicVariant<OTHER_BC>() const {
			return PeriodicVariant<OTHER_BC>{depth(), i(), j(), k()};
		}

		//define element specific constructors
		GEOVOX_DECLARE_SIMD()
		constexpr VoxelElementKey(const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE( 	ii, jj, kk,
					0, dd, BC, NO_COLOR) {
			if (dd>MAX_DEPTH) {_data_ = DOES_NOT_EXIST; return;}
			if constexpr (PX||PY||PZ) {
				const uint64_t me = uint64_t{1} << dd; //2^d elements per axis
				if constexpr (PX) {if (ii>=me) {set_i(0);}}
				if constexpr (PY) {if (jj>=me) {set_j(0);}}
				if constexpr (PZ) {if (kk>=me) {set_k(0);}}
			}
		}

		GEOVOX_DECLARE_SIMD()
		constexpr VoxelElementKey(const uint64_t dd, uint64_t li) {
			// L = i + j*N + k*N^2

			assert(dd<=MAX_DEPTH);
			assert(li < (uint64_t{1} << (3*dd)));

			const uint64_t N = (uint64_t{1} << dd); //number of elements per side
			const uint64_t ii = li % N; li>>=dd; 	//note li>>=dd is the same as li/=N
			const uint64_t jj = li % N; li>>=dd;
			const uint64_t kk = li;					//note the remainder li is less than N now

			//assemble data
			_data_ = BASE{ii,jj,kk,0,dd,BC,0}._data_;
		}

		
		//check if a voxel is valid
		GEOVOX_DECLARE_SIMD()
		constexpr bool is_valid() const {
			const uint64_t mei = (uint64_t{1} << depth()) - 1; //max element index
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > mei) 			 {return false;}
			if (j() > mei) 			 {return false;}
			if (k() > mei) 			 {return false;}
			return true;
		}

		//get the linear index of the element at the current depth
		// L = i + j*N + k*N^2
		GEOVOX_DECLARE_SIMD()
		constexpr uint64_t depth_linear_index() const {
			assert(is_valid());
			const uint64_t N = (uint64_t{1} << depth()); //number of elements per side
			return i() + N*(j() + N*k());
		}

		GEOVOX_DECLARE_SIMD()
		static constexpr uint64_t depth_linear_start(const uint64_t dd) {
			//sum from d=0 to dd-1 of 8^d
			return ((uint64_t{1} << (3*dd)) - 1)/7;
		}

		GEOVOX_DECLARE_SIMD()
		constexpr uint64_t linear_index() const {
			return depth_linear_start(depth()) + depth_linear_index();
		}

		//hierarchy logic
		constexpr VoxelElementKey parent() const {
			assert(this->exists());
			if (depth()==0) {return VoxelElementKey{DOES_NOT_EXIST};}
			return VoxelElementKey{depth()-1, i()/2, j()/2, k()/2}; //integer division is intended
		}

		inline constexpr auto child(const int i) const {return children()[i];}

		constexpr std::array<VoxelElementKey,8> children() const {
			const uint64_t ii=2*i(), jj=2*j(), kk=2*k(), dd=depth()+1;
			if (dd>MAX_DEPTH) {
				return {
					VoxelElementKey{DOES_NOT_EXIST},
					VoxelElementKey{DOES_NOT_EXIST},
					VoxelElementKey{DOES_NOT_EXIST},
					VoxelElementKey{DOES_NOT_EXIST},
					VoxelElementKey{DOES_NOT_EXIST},
					VoxelElementKey{DOES_NOT_EXIST},
					VoxelElementKey{DOES_NOT_EXIST},
					VoxelElementKey{DOES_NOT_EXIST}
				};
			}

			return {
				VoxelElementKey{dd, ii  ,jj  ,kk  },
				VoxelElementKey{dd, ii+1,jj  ,kk  },
				VoxelElementKey{dd, ii  ,jj+1,kk  },
				VoxelElementKey{dd, ii+1,jj+1,kk  },
				VoxelElementKey{dd, ii  ,jj  ,kk+1},
				VoxelElementKey{dd, ii+1,jj  ,kk+1},
				VoxelElementKey{dd, ii  ,jj+1,kk+1},
				VoxelElementKey{dd, ii+1,jj+1,kk+1}
			};
		}

		//adjacency logic
		inline constexpr auto vertex(int i) const {return vertices()[i];}
		inline constexpr std::array<VoxelVertexKey<MAX_DEPTH_,BC>,8> vertices() const;
		inline constexpr auto face(int i) const {return faces()[i];}
		inline constexpr std::array<VoxelFaceKey<MAX_DEPTH_,BC>,6> faces() const;

		constexpr std::array<VoxelElementKey,26> neighbors() const {
			const uint64_t ii=i(), jj=j(), kk=k(), dd=depth();
			const uint64_t me = (uint64_t{1}<<dd)-1; //maximum valid i,j,k index
			uint64_t im1, jm1, km1, ip1, jp1, kp1;

			if constexpr (PX) {
				im1 = ii==0   ? me : ii-1;
				ip1 = ii==me ? 0   : ii+1;
			} else {
				im1 = ii-1;
				ip1 = ii+1;
			}

			if constexpr (PY) {
				jm1 = jj==0   ? me : jj-1;
				jp1 = jj==me ? 0   : jj+1;
			} else {
				jm1 = jj-1;
				jp1 = jj+1;
			}

			if constexpr (PZ) {
				km1 = kk==0   ? me : kk-1;
				kp1 = kk==me ? 0   : kk+1;
			} else {
				km1 = kk-1;
				kp1 = kk+1;
			}

			return {
				//bottom slice
				VoxelElementKey{dd, im1, jm1, km1},
				VoxelElementKey{dd, ii , jm1, km1},
				VoxelElementKey{dd, ip1, jm1, km1},
				VoxelElementKey{dd, im1, jj , km1},
				VoxelElementKey{dd, ii , jj , km1},
				VoxelElementKey{dd, ip1, jj , km1},
				VoxelElementKey{dd, im1, jp1, km1},
				VoxelElementKey{dd, ii , jp1, km1},
				VoxelElementKey{dd, ip1, jp1, km1},

				//middle slice (remove center)
				VoxelElementKey{dd, im1, jm1, kk},
				VoxelElementKey{dd, ii , jm1, kk},
				VoxelElementKey{dd, ip1, jm1, kk},
				VoxelElementKey{dd, im1, jj , kk},
				// VoxelElementKey{dd, ii , jj , kk},
				VoxelElementKey{dd, ip1, jj , kk},
				VoxelElementKey{dd, im1, jp1, kk},
				VoxelElementKey{dd, ii , jp1, kk},
				VoxelElementKey{dd, ip1, jp1, kk},

				//top slice
				VoxelElementKey{dd, im1, jm1, kp1},
				VoxelElementKey{dd, ii , jm1, kp1},
				VoxelElementKey{dd, ip1, jm1, kp1},
				VoxelElementKey{dd, im1, jj , kp1},
				VoxelElementKey{dd, ii , jj , kp1},
				VoxelElementKey{dd, ip1, jj , kp1},
				VoxelElementKey{dd, im1, jp1, kp1},
				VoxelElementKey{dd, ii , jp1, kp1},
				VoxelElementKey{dd, ip1, jp1, kp1}
			};
		}

		//iterator logic
		GEOVOX_DECLARE_SIMD()
		VoxelElementKey& operator++() {
			assert(is_valid());
			const uint64_t dd  = depth();
			const uint64_t Nm1 = (uint64_t{1} << dd) -1; //max index for i,j,k

			const uint64_t ii = i();
			if (ii<Nm1) {set_i(ii+1); return *this;}
			else {set_i(0);}

			//i must roll over to 0 and increment j
			const uint64_t jj = j();
			if (jj<Nm1) {set_j(jj+1); return *this;}
			else {set_j(0);}

			//j must roll over to 0 and incmement k
			const uint64_t kk = k();
			if (kk<Nm1) {set_k(kk+1); return *this;}
			else {set_k(0);} //return here if we want wrap back to the end

			_data_ = DOES_NOT_EXIST;
			return *this;
		}
	};
}