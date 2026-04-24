#pragma once

#include "mesh/keys/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>
#include <array>

namespace GV
{
	//define the element key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelVertexKey;

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelFaceKey;

	template<uint64_t MAX_DEPTH_=16, uint64_t BC=0>
	struct VoxelElementKey : public VoxelKey<3,0,MAX_DEPTH_+1>
	{
		static_assert(BC<8, "VoxelElementKey: invalid boundary condition. BC must be from 0 to 7.");

		//inherit constructors
		using BASE = VoxelKey<3,0,MAX_DEPTH_+1>;
		using BASE::BASE;

		//inherit the primary accessors
		using BASE::color;
		using BASE::depth;
		using BASE::i;
		using BASE::j;
		using BASE::k;
		using BASE::set_depth;
		using BASE::set_i;
		using BASE::set_j;
		using BASE::set_k;
		using BASE::_data_;

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
		explicit operator NonPeriodicVariant() const {return NonPeriodicVariant{_data_};}

		template<uint64_t OTHER_BC>
		using PeriodicVariant = VoxelElementKey<MAX_DEPTH_,OTHER_BC>;
		
		template<uint64_t OTHER_BC> requires (OTHER_BC<8)
		explicit operator PeriodicVariant<OTHER_BC>() const {return PeriodicVariant<OTHER_BC>{_data_};}

		//define element specific constructors
		constexpr VoxelElementKey(const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE( 	(ii&1)|((jj&1)<<1)|((kk&1)<<2),
					ii>>1, jj>>1, kk>>1,
					0, dd, BC, 0) {
				if (dd>MAX_DEPTH) {_data_ = DOES_NOT_EXIST; return;}
				if constexpr (PX||PY||PZ) {
					const uint64_t me = uint64_t{1} << dd; //2^d elements per axis
					if constexpr (PX) {if (ii>=me) {set_i(0);}}
					if constexpr (PY) {if (jj>=me) {set_j(0);}}
					if constexpr (PZ) {if (kk>=me) {set_k(0);}}
				}
			}

		constexpr VoxelElementKey(const uint64_t dd, uint64_t li) {
			assert(dd<=MAX_DEPTH);
			assert(li < (uint64_t{1} << (3*dd)));

			if (dd==0) {*this = VoxelElementKey{0,0,0,0}; return;}

			const uint64_t r_wd   = dd-1;						//width on non-color index bits
			const uint64_t r_mask = (uint64_t{1}<<r_wd) - 1;	//mask for the non-color index bits

			//split color, i, j, k fields
			const uint64_t cc 	= li&7;
			const uint64_t r_ii = (li>>3) 		 	& r_mask;
			const uint64_t r_jj = (li>>(3+r_wd)) 	& r_mask;
			const uint64_t r_kk = (li>>(3+2*r_wd)) 	& r_mask;

			//assemble data
			_data_ = BASE(cc,r_ii,r_jj,r_kk,0,dd,BC,0)._data_;
		}

		
		//check if a voxel is valid
		constexpr bool is_valid() const {
			const uint64_t mei = (uint64_t{1} << depth()) - 1; //max element index
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > mei) 			 {return false;}
			if (j() > mei) 			 {return false;}
			if (k() > mei) 			 {return false;}
			return true;
		}

		//get the linear index of the element at the current depth
		constexpr uint64_t depth_linear_index() const {
			assert(is_valid());
			//compress the index fields and color
			const uint64_t dd   	= depth();
			if (dd==0) {return 0;}

			const uint64_t r_wd 	= dd-1;
			const uint64_t cc 		= _data_&7;
			const uint64_t r_ii		= (_data_&BASE::I_M) >> BASE::I_S;
			const uint64_t r_jj		= (_data_&BASE::J_M) >> BASE::J_S;
			const uint64_t r_kk 	= (_data_&BASE::K_M) >> BASE::K_S;

			return cc | (r_ii<<3) | (r_jj<<(3+r_wd)) | (r_kk<<(3+2*r_wd));
		}

		static constexpr uint64_t depth_linear_start(const uint64_t dd) {
			//sum from d=0 to dd-1 of 8^d
			return ((uint64_t{1} << (3*dd)) - 1)/7;
		}

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

		//iterator logic
		VoxelElementKey& operator++() {
			assert(is_valid());

			const uint64_t dd = depth();
			if (dd==0) {_data_ = DOES_NOT_EXIST; return *this;}

			const uint64_t cc = _data_&7;

			const uint64_t cc_n = (cc + 1)&7; //next color
			_data_ = (_data_&~BASE::C_M) | cc_n;
			if (cc_n!=0) {return *this;}
			
			//need to carry
			const uint64_t r_wd 	= dd-1;
			const uint64_t r_mask	= (uint64_t{1}<<r_wd) - 1; //also the maximum non-color index
			
			const uint64_t r_ii		= (_data_&BASE::I_M) >> BASE::I_S;
			const uint64_t r_ii_n	= (r_ii+1) & r_mask;
			_data_ = (_data_&~BASE::I_M) | (r_ii_n << BASE::I_S);
			if (r_ii_n!=0) {return *this;}

			const uint64_t r_jj		= (_data_&BASE::J_M) >> BASE::J_S;
			const uint64_t r_jj_n	= (r_jj+1) & r_mask;
			_data_ = (_data_&~BASE::J_M) | (r_jj_n << BASE::J_S);
			if (r_jj_n!=0) {return *this;}
			
			const uint64_t r_kk		= (_data_&BASE::K_M) >> BASE::K_S;
			const uint64_t r_kk_n	= (r_kk+1) & r_mask;
			_data_ = (_data_&~BASE::K_M) | (r_kk_n << BASE::K_S);
			if (r_kk_n!=0) {return *this;}

			_data_ = DOES_NOT_EXIST;
			return *this;
		}
	};
}