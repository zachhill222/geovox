#pragma once

#include "mesh/keys/voxel_key_base.hpp"
#include <cstdint>
#include <cassert>


namespace GV
{
	//define the face key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelElementKey;

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelVertexKey;

	template<uint64_t MAX_DEPTH_=16, uint64_t BC=0>
	struct VoxelFaceKey : public VoxelKey<3,2,MAX_DEPTH_+1>
	{
		//inherit constructors
		using BASE = VoxelKey<3,2,MAX_DEPTH_+1>;
		using BASE::BASE;

		//inherit the primary accessors
		using BASE::depth;
		using BASE::color;
		using BASE::i;
		using BASE::j;
		using BASE::k;
		using BASE::set_depth;
		using BASE::set_i;
		using BASE::set_j;
		using BASE::set_k;
		using BASE::_data_;

		//re-name other_c() to axis() for readability
		//get the normal axis to the face
		inline constexpr uint64_t axis() const {return this->other_c();}

		//define useful constants
		static constexpr uint64_t MAX_FACE_INDEX_NAX = BASE::MAX_INDEX -1;
		static constexpr uint64_t MAX_FACE_INDEX_AX  = BASE::MAX_INDEX;
		static constexpr uint64_t A_S                = BASE::OC_S;
		static constexpr uint64_t A_W				 = BASE::OC_W;
		static constexpr uint64_t A_M				 = BASE::OC_M;
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;
		static_assert(MAX_DEPTH_==MAX_DEPTH);

		//periodic conditions. the BC bits are stored on the other_nocompare field
		static constexpr uint64_t BC_FLAG = BC;
		static constexpr bool PX = BC&1; //periodic in i/x
		static constexpr bool PY = BC&2; //periodic in j/y
		static constexpr bool PZ = BC&4; //periodic in k/z

		//explicit conversion to the non-periodic type and to a periodic type
		using NonPeriodicVariant = VoxelFaceKey<MAX_DEPTH_,0>;
		explicit operator NonPeriodicVariant() const {return NonPeriodicVariant{_data_};}

		template<uint64_t OTHER_BC>
		using PeriodicVariant = VoxelFaceKey<MAX_DEPTH_,OTHER_BC>;
		
		template<uint64_t OTHER_BC> requires (OTHER_BC<8)
		explicit operator PeriodicVariant<OTHER_BC>() const {return PeriodicVariant<OTHER_BC>{_data_};}


		//define face specific constructors
		VoxelFaceKey(const uint64_t aa, const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE( (ii&1)|((jj&1)<<1)|((kk&1)<<2),
					ii>>1, jj>>1, kk>>1,
					aa, dd, BC, 0
				) {
				if (dd>MAX_DEPTH) {_data_ = DOES_NOT_EXIST; return;}
				if constexpr (PX||PY||PZ) {
					const uint64_t mn = (uint64_t{1} << dd); //2^d elements per axis, number of faces in non-axis directions
					const uint64_t ma = mn+1; //number of faces in the axis direction

					if constexpr (PX) {if (ii>= (aa==0 ? ma : mn) ) {set_i(0);}}
					if constexpr (PY) {if (jj>= (aa==1 ? ma : mn) ) {set_j(0);}}
					if constexpr (PZ) {if (kk>= (aa==2 ? ma : mn) ) {set_k(0);}}
				}
			}

		constexpr VoxelFaceKey(const uint64_t dd, uint64_t li) {
			//define by linear index
			assert(dd <= MAX_DEPTH);
			assert(li < 3*(uint64_t{1} << (3*dd+1)));

			//partition the indices into: axis0 normal | axis1 normal | axis2 normal
			//the numbering within and across each partition is continuous as there are N = 2^d * 2^d * 2^(d+1) = 2^(3d+1) faces with a given normal axis
			//thus axis0 uses indices [0,N), axis1 uses [N,2N), and axis2 uses [2N,3N).
			
			if (dd==0) {
				switch (li) {
				case 0: *this = VoxelFaceKey{0,0,0,0,0}; return;
				case 1: *this = VoxelFaceKey{0,0,1,0,0}; return;
				case 2: *this = VoxelFaceKey{1,0,0,0,0}; return;
				case 3: *this = VoxelFaceKey{1,0,0,1,0}; return;
				case 4: *this = VoxelFaceKey{2,0,0,0,0}; return;
				case 5: *this = VoxelFaceKey{2,0,0,0,1}; return;
				default: _data_ = DOES_NOT_EXIST;  		 return;
				}
			}

			const uint64_t aa 	= li >> (3*dd+1); //note li/N = li * 2^-(3d+1)

			const uint64_t cc 	= li&7;
			const uint64_t mna 	= (uint64_t{1}<<(dd-1)) - 1; //mask for non-color bits for non-axis indices
			const uint64_t ma 	= (uint64_t{1}<<(dd))   - 1; //mask for non-color bits for axis index

			uint64_t r_ii, r_jj, r_kk;
			switch (aa) {
			case 0:
				r_ii = (li>>3)			& ma;
				r_jj = (li>>(3+dd))		& mna;
				r_kk = (li>>(2+2*dd))	& mna;
				break;
			
			case 1:
				r_ii = (li>>3)			& mna;
				r_jj = (li>>(2+dd))		& ma;
				r_kk = (li>>(2+2*dd))	& mna;
				break;
			
			case 2:
				r_ii = (li>>3)			& mna;
				r_jj = (li>>(2+dd))		& mna;
				r_kk = (li>>(1+2*dd))	& ma;
				break;
			default:
				_data_ = DOES_NOT_EXIST;
				return;
			}

			_data_ = BASE(cc,r_ii,r_jj,r_kk,aa,dd,BC,0)._data_;
		}

		//check if a face is valid
		constexpr bool is_valid() const {
			const uint64_t mfiax  = uint64_t{1} << depth(); //max face index in the normal axis direction
			const uint64_t mfinax = mfiax-1;
			const uint64_t aa     = axis();
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > ((aa==0) ? mfiax : mfinax)) {return false;}
			if (j() > ((aa==1) ? mfiax : mfinax)) {return false;}
			if (k() > ((aa==2) ? mfiax : mfinax)) {return false;}
			return true;
		}

		//partition the indices into: axis0 normal | axis1 normal | axis2 normal
		//the numbering within and across each partition is continuous as there are N = 2^d * 2^d * 2^(d+1) = 2^(3d+1) faces with a given normal axis
		//thus axis0 uses indices [0,N), axis1 uses [N,2N), and axis2 uses [2N,3N).

		constexpr uint64_t depth_linear_index() const {
			assert(is_valid());

			const uint64_t dd = depth();
			const uint64_t aa = axis();
			const uint64_t ps = depth_axis_start(dd,aa); //start of this partition

			const uint64_t cc 	= _data_&7;
			const uint64_t r_ii	= (_data_ & BASE::I_M) >> BASE::I_S;
			const uint64_t r_jj	= (_data_ & BASE::J_M) >> BASE::J_S;
			const uint64_t r_kk	= (_data_ & BASE::K_M) >> BASE::K_S;

			//assemble index cc | r_i | r_j | r_k
			//r_* has width dd-1 for non-axis and width dd for axis indices (one bit was moved to the color)
			switch (aa) {
			case 0:
				return ps + ( cc | (r_ii<<3) | (r_jj<<(3+dd)) | (r_kk<<(2+2*dd)) );
			case 1:
				return ps + ( cc | (r_ii<<3) | (r_jj<<(2+dd)) | (r_kk<<(2+2*dd)) );
			case 2:
				return ps + ( cc | (r_ii<<3) | (r_jj<<(2+dd)) | (r_kk<<(1+2*dd)) );
			default: return DOES_NOT_EXIST;
			}
		}

		static constexpr uint64_t depth_axis_start(const uint64_t dd, const uint64_t aa) {
			const uint64_t N = uint64_t{1} << (3*dd+1);
			return aa*N;
		}

		static constexpr uint64_t depth_linear_start(const uint64_t dd) {
			// 3*2^(3d+1) faces per axis at depth d, summed from 0 to dd-1
			return 6 * ( ((uint64_t{1} << (3*dd)) - 1)/7 );
		}

		constexpr uint64_t linear_index() const {
			return depth_linear_start(depth()) + depth_linear_index();
		}

		//geometry logic
		constexpr bool on_bbox_boundary() const {
			assert(this->exists());
			const uint64_t mfiax = uint64_t{1} << depth();
			const uint64_t idx   = this->index(axis());
			return idx==0 || idx==mfiax;
		}

		//hierarchy logic
		inline constexpr auto child(const int i) const {return children()[i];}

		constexpr std::array<VoxelFaceKey,4> children() const {
			assert(this->exists());
			const uint64_t ci=2*i(), cj=2*j(), ck=2*k(), cd=depth()+1, aa=axis();
			if (cd>MAX_DEPTH) {
				return {
					VoxelFaceKey{DOES_NOT_EXIST},
					VoxelFaceKey{DOES_NOT_EXIST},
					VoxelFaceKey{DOES_NOT_EXIST},
					VoxelFaceKey{DOES_NOT_EXIST}
				};
			}

			switch(aa) {
			case 0:
				return {
					VoxelFaceKey{aa,cd, ci, cj,   ck  },
					VoxelFaceKey{aa,cd, ci, cj+1, ck  },
					VoxelFaceKey{aa,cd, ci, cj,   ck+1},
					VoxelFaceKey{aa,cd, ci, cj+1, ck+1}
				};
			case 1:
				return {
					VoxelFaceKey{aa,cd, ci,   cj, ck  },
					VoxelFaceKey{aa,cd, ci+1, cj, ck  },
					VoxelFaceKey{aa,cd, ci,   cj, ck+1},
					VoxelFaceKey{aa,cd, ci+1, cj, ck+1}
				};
			case 2:
				return {
					VoxelFaceKey{aa,cd, ci,   cj,   ck},
					VoxelFaceKey{aa,cd, ci+1, cj,   ck},
					VoxelFaceKey{aa,cd, ci,   cj+1, ck},
					VoxelFaceKey{aa,cd, ci+1, cj+1, ck}
				};
			default:
				return {
					VoxelFaceKey{DOES_NOT_EXIST},
					VoxelFaceKey{DOES_NOT_EXIST},
					VoxelFaceKey{DOES_NOT_EXIST},
					VoxelFaceKey{DOES_NOT_EXIST}
				};
			}
		}

		constexpr VoxelFaceKey parent() const {
			assert(this->exists());
			//not all faces have parents
			const uint64_t aa = axis();
			const uint64_t dd = depth();
			const uint64_t ii = i();
			const uint64_t jj = j();
			const uint64_t kk = k();

			if (dd==0) {return VoxelFaceKey{DOES_NOT_EXIST};}
			switch (aa) {
			case 0: return (jj&1 || kk&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{aa,dd-1, ii>>1, jj>>1, kk>>1};
			case 1: return (kk&1 || ii&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{aa,dd-1, ii>>1, jj>>1, kk>>1};
			case 2: return (ii&1 || jj&1) ? VoxelFaceKey{DOES_NOT_EXIST} : VoxelFaceKey{aa,dd-1, ii>>1, jj>>1, kk>>1};
			default: return VoxelFaceKey{DOES_NOT_EXIST};
			}
		}

		//adjacency operations
		inline constexpr auto element(const int i) const {return elements()[i];}
		constexpr std::array<VoxelElementKey<MAX_DEPTH_,BC>,2> elements() const;

		inline constexpr auto vertex(const int i) const {return vertices()[i];}
		constexpr std::array<VoxelVertexKey<MAX_DEPTH_,BC>,4> vertices() const;

		//iterator logic
		VoxelFaceKey& operator++() {
			assert(is_valid());

			//this is tricky, move to the constructor
			const uint64_t dd  	= depth();
			const uint64_t idx 	= depth_linear_index()+1;
			if (idx >= 3*(uint64_t{1}<<(3*dd+1))) {
				_data_ = DOES_NOT_EXIST;
				return *this;
			}
			else {
				*this = VoxelFaceKey{dd, idx};
			}
			return *this;
		}
	};
}