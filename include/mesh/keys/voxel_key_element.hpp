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

	template<uint64_t I_W, uint64_t BC, bool MORTON>
	struct VoxelVertexKey;

	template<uint64_t I_W, uint64_t BC, bool MORTON>
	struct VoxelFaceKey;

	template<uint64_t I_W=16, uint64_t BC=0, bool MORTON_=false>
	struct VoxelElementKey : public VoxelKey<0,3,0,I_W>
	{
		static_assert(BC<8, "VoxelElementKey: invalid boundary condition. BC must be from 0 to 7.");

		//inherit constructors
		using BASE = VoxelKey<0,3,0,I_W>;
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

		//define useful constants
		static constexpr bool MORTON = MORTON_;
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;

		//periodic conditions. the BC bits are stored on the other_nocompare field
		static constexpr uint64_t BC_FLAG = BC;
		static constexpr bool PX = BC&1; //periodic in i/x
		static constexpr bool PY = BC&2; //periodic in j/y
		static constexpr bool PZ = BC&4; //periodic in k/z

		//explicit conversion to the non-periodic type
		using NonPeriodicType = VoxelElementKey<I_W,0,MORTON_>;
		explicit operator NonPeriodicType() const {return NonPeriodicType{_data_};}

		template<uint64_t OTHER_BC>
		using OtherPeriodicType = VoxelElementKey<I_W,OTHER_BC,MORTON_>;
		
		template<uint64_t OTHER_BC> requires (OTHER_BC<8)
		explicit operator OtherPeriodicType<OTHER_BC>() const {return OtherPeriodicType<OTHER_BC>{_data_};}

		//define element specific constructors
		constexpr VoxelElementKey(const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE(0,BC,0,dd,ii,jj,kk) {
				if (dd>MAX_DEPTH) {_data_ = DOES_NOT_EXIST; return;}
				if constexpr (PX||PY||PZ) {
					const uint64_t me = uint64_t{1} << dd; //2^d elements per axis
					if constexpr (PX) {if (ii>=me) {set_i(0);}}
					if constexpr (PY) {if (jj>=me) {set_j(0);}}
					if constexpr (PZ) {if (kk>=me) {set_k(0);}}
				}
			}

		constexpr VoxelElementKey(const uint64_t dd, uint64_t li) requires (!MORTON) {
			assert(dd<=MAX_DEPTH);
			assert(li < (uint64_t{1} << (3*dd)));

			const uint64_t mask = (uint64_t{1} << dd) - 1;
			const uint64_t ii   = li & mask; li >>= dd;
			const uint64_t jj   = li & mask; li >>= dd;
			const uint64_t kk   = li & mask;

			_data_ = (dd << BASE::D_S) | (ii << BASE::I_S) | (jj << BASE::J_S) | (kk << BASE::K_S) | (BC << BASE::ON_S);
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

		VoxelElementKey& operator++() requires (!MORTON) {
			uint64_t dd = depth();
			const uint64_t mi = (uint64_t{1} << dd)-1; //2^dd elements per axis (max index)
			uint64_t ii=i(), jj=j(), kk=k();
			
			//increment with carry, but we must respect the entire
			//field width of I_W, J_W, and K_W
			++ii;
			if (ii>mi) {ii=0; ++jj;}
			if (jj>mi) {jj=0; ++kk;}
			if (kk>mi) {kk=0; ++dd;}

			//reset d-i-j-k bits
			_data_ &= ~BASE::DEPTH_INDEX_MASK;
			_data_ |=  BASE::DEPTH_INDEX_MASK & ((ii<<BASE::I_S) | (jj<<BASE::J_S) | (kk<<BASE::K_S) | (dd<<BASE::D_S));
			return *this;
		}

		//get the linear index of the element at the current depth
		constexpr uint64_t depth_linear_index() const {
			if (!is_valid()) {
				std::cout << "INVALID ELEMENT\n" << *this << std::endl;
			}

			assert(is_valid());
			const uint64_t dd   = depth();
			const uint64_t mask = ((uint64_t{1} << dd) - 1); //maximum element index at this depth
			const uint64_t ii   = (_data_ >> BASE::I_S) & mask;
			const uint64_t jj   = (_data_ >> BASE::J_S) & mask;
			const uint64_t kk   = (_data_ >> BASE::K_S) & mask;
			
			if constexpr (MORTON) {
				//morton indexing (two features of the same color on the same depth are 8 indices from eachother)
				//better for looping over a single color
				const uint64_t clr = (ii&1) | ((jj&1)<<1) | ((kk&1)<<2); //color in the least significant bits
				const uint64_t rem = ((ii>>1)) | ((jj>>1) << (dd-1)) | ((kk>>1) << (2*dd-1));
				return (rem<<3) | clr;
			}
			else {
				//standard ordering (contiguous i, loop k->j->i (outer to inner))
				return ii | (jj << dd) | (kk << (2*dd));
			}
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
		inline constexpr std::array<VoxelVertexKey<I_W,BC,MORTON_>,8> vertices() const;
		
		inline constexpr auto face(int i) const {return faces()[i];}
		inline constexpr std::array<VoxelFaceKey<I_W,BC,MORTON_>,6> faces() const;

		//get the color of the feature by index modding. There are 8 unique colors.
		//coloring means different things for differnet mesh elements, but for example,
		//two elements with the same color at the same depth share no vertices or faces
		inline constexpr uint64_t color() const {
			//even/even/even -> 0
			//odd/even/even  -> 1
			//even/odd/even  -> 2
			//etc.
			return (i()&1) | ((j()&1) << 1) | ((k()&1) << 2);
		}
	};
}