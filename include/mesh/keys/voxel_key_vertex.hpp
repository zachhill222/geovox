#pragma once

#include "gutil.hpp"

#include "mesh/keys/voxel_key_base.hpp"

#include <cstdint>
#include <cassert>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace GV
{
	//define the vertex key and implement most methods.
	//adjacency methods must be implemented in a separate file after
	//all mesh feature keys are defined

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelElementKey;

	template<uint64_t MAX_DEPTH_, uint64_t BC>
	struct VoxelFaceKey;

	template<uint64_t MAX_DEPTH_=16, uint64_t BC=0>
	struct VoxelVertexKey : public VoxelKey<3,0,MAX_DEPTH_+1>
	{
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
		using Hash = typename BASE::Hash;

		//usually the fee bits mean a manual color is set
		#pragma omp declare simd
		inline constexpr uint64_t color() const {return BASE::free();}
		#pragma omp declare simd
		inline constexpr void set_color(const uint64_t clr) {BASE::set_free(clr);}
		static constexpr uint64_t NO_COLOR = BASE::F_M >> BASE::F_S;

		//define useful constants
		static constexpr uint64_t MAX_VERTEX_INDEX = BASE::MAX_INDEX;
		using BASE::MAX_DEPTH;
		using BASE::DOES_NOT_EXIST;
		static_assert(MAX_DEPTH_==MAX_DEPTH);

		[[nodiscard]] static constexpr VoxelVertexKey None() { return {DOES_NOT_EXIST}; }

		//periodic conditions. the BC bits are stored on the other_nocompare field
		static constexpr uint64_t BC_FLAG = BC;
		static constexpr bool PX = BC&1; //periodic in i/x
		static constexpr bool PY = BC&2; //periodic in j/y
		static constexpr bool PZ = BC&4; //periodic in k/z

		//explicit conversion to the non-periodic type
		using NonPeriodicVariant = VoxelVertexKey<MAX_DEPTH_,0>;
		explicit operator NonPeriodicVariant() const {return NonPeriodicVariant{_data_&~BASE::ON_M};}

		template<uint64_t OTHER_BC>
		using PeriodicVariant = VoxelVertexKey<MAX_DEPTH_,OTHER_BC>;
		
		template<uint64_t OTHER_BC> requires (OTHER_BC<8)
		inline explicit operator PeriodicVariant<OTHER_BC>() const {
			return PeriodicVariant<OTHER_BC>{depth(), i(), j(), k()};
		}

		//define vertex specific constructors
		constexpr VoxelVertexKey(const uint64_t dd, const uint64_t ii, const uint64_t jj, const uint64_t kk) :
			BASE(	ii, jj, kk,
					0, dd, BC, NO_COLOR) {
				if (dd>MAX_DEPTH) {_data_ = DOES_NOT_EXIST; return;}
				if constexpr (PX||PY||PZ) {
					//2^d elements per axis, one extra vertex
					//note that the vertices need to wrap at the index of the largest element
					//so that the upper index of the largest element is the lower vertex of the smallest element
					const uint64_t mv = (uint64_t{1} << dd);
					if constexpr (PX) {if (ii>=mv) {set_i(0);}}
					if constexpr (PY) {if (jj>=mv) {set_j(0);}}
					if constexpr (PZ) {if (kk>=mv) {set_k(0);}}
				}
			}

		constexpr VoxelVertexKey(const uint64_t dd, uint64_t li) {
			assert(dd<=MAX_DEPTH);
			const uint64_t nv   = (uint64_t{1} << dd) + 1; //2^d + 1 vertices per axis
			const uint64_t ii   = li % nv; li /= nv;
			const uint64_t jj   = li % nv; li /= nv;
			const uint64_t kk   = li;

			_data_ = BASE{ii,jj,kk,0,dd,BC,0}._data_;
		}

		//check if a voxel is valid
		constexpr bool is_valid() const {
			const uint64_t mvi = uint64_t{1} << depth(); //max vertex index
			if (depth() > MAX_DEPTH) {return false;}
			if (i() > mvi) 			 {return false;}
			if (j() > mvi) 			 {return false;}
			if (k() > mvi) 			 {return false;}
			return true;
		}

		//get the linear index of the element at the current depth
		constexpr uint64_t depth_linear_index() const {
			assert(is_valid());
			const uint64_t nv   = (uint64_t{1} << depth()) + 1; //number of vertices per side
			return i() + nv*(j() + nv*k());
		}

		static constexpr uint64_t depth_linear_start(const uint64_t dd) {
			//(2^d + 1)^3 vertices per depth. expand and sum from 0 to dd-1
			return ((uint64_t{1} << (3*dd)) - 1)/7 
				 + (uint64_t{1}<<(2*dd)) 
				 + 3*(uint64_t{1}<<dd)
				 + dd - 4;
		}

		constexpr uint64_t linear_index() const {
			return depth_linear_start(depth()) + depth_linear_index();
		}

		//geometry logic
		constexpr bool on_bbox_boundary() const {
			const uint64_t mvi = uint64_t{1} << depth();
			const uint64_t ii=i(), jj=j(), kk=k();
			return ii==0 || ii==mvi || jj==0 || jj==mvi || kk==0 || kk==mvi;
		}

		constexpr bool on_bbox_boundary_x() const {
			const uint64_t mvi = uint64_t{1} << depth();
			const uint64_t idx = i();
			return idx==0 || idx==mvi;
		}
		constexpr bool on_bbox_boundary_y() const {
			const uint64_t mvi = uint64_t{1} << depth();
			const uint64_t idx = j();
			return idx==0 || idx==mvi;
		}
		constexpr bool on_bbox_boundary_z() const {
			const uint64_t mvi = uint64_t{1} << depth();
			const uint64_t idx = k();
			return idx==0 || idx==mvi;
		}

		inline constexpr double x() const {return std::ldexp(static_cast<double>(i()), -static_cast<int>(depth()));}
		inline constexpr double y() const {return std::ldexp(static_cast<double>(j()), -static_cast<int>(depth()));}
		inline constexpr double z() const {return std::ldexp(static_cast<double>(k()), -static_cast<int>(depth()));}

		constexpr VoxelVertexKey reduced_key() const {
			//return the vertex key at the lowest depth
			//that is at the same gemetric location as this vertex
			uint64_t dd = depth();
			uint64_t ii=i(), jj=j(), kk=k();
			while (dd>0 && !(ii&1) && !(jj&1) && !(kk&1)) {
				//shift right until the least significant bit of i,j,k is used or the depth is 0
				dd  -= 1;
				ii >>= 1;
				jj >>= 1;
				kk >>= 1;
			}
			return VoxelVertexKey{dd, ii, jj, kk};
		}

		constexpr bool is_same_coord(const VoxelVertexKey other) const {
			return reduced_key() == other.reduced_key();
		}

		constexpr gutil::Point<3,double> normalized_coordinate() const {
			const int exponent = -static_cast<int>(depth());
			return gutil::Point<3,double>{
				std::ldexp(static_cast<double>(i()), exponent),
				std::ldexp(static_cast<double>(j()), exponent),
				std::ldexp(static_cast<double>(k()), exponent)
			};
		}

		//hierarchy logic
		constexpr VoxelVertexKey parent() const {
			assert(this->exists());
			const uint64_t ii = i();
			const uint64_t jj = j();
			const uint64_t kk = k();
			const uint64_t dd = depth();
			if (ii&1 || jj&1 || kk&1 || dd==0) {return VoxelVertexKey{DOES_NOT_EXIST};} //vertices with odd indices don't have parents
			return VoxelVertexKey{dd-1, ii/2, jj/2, kk/2};
		}

		inline constexpr VoxelVertexKey child() const {
			assert(this->exists());
			if (depth()>=MAX_DEPTH) {return VoxelVertexKey{DOES_NOT_EXIST};}
			return VoxelVertexKey{depth()+1, 2*i(), 2*j(), 2*k()};
		}

		//adjacency logic
		inline constexpr auto element(int i) const {return elements()[i];}
		constexpr std::array<VoxelElementKey<MAX_DEPTH_,BC>,8> elements() const;

		//the reference coordinate of this vertex in each of the 8 elements it belongs to
		static constexpr auto ref_coord(const int i) {return ref_coords()[i];}
		static constexpr std::array<gutil::Point<3,double>,8> ref_coords() {
			return {
				gutil::Point<3,double>{ 1.0,  1.0,  1.0},
				gutil::Point<3,double>{ 1.0,  1.0, -1.0},
				gutil::Point<3,double>{ 1.0, -1.0,  1.0},
				gutil::Point<3,double>{ 1.0, -1.0, -1.0},
				gutil::Point<3,double>{-1.0,  1.0,  1.0},
				gutil::Point<3,double>{-1.0,  1.0, -1.0},
				gutil::Point<3,double>{-1.0, -1.0,  1.0},
				gutil::Point<3,double>{-1.0, -1.0, -1.0}
			};
		}

		//iterator logic
		VoxelVertexKey& operator++() {
			assert(is_valid());

			uint64_t dd = depth();
			const uint64_t mi = uint64_t{1} << dd; //2^dd +1 vertices per axis (max index)
			uint64_t ii=i(), jj=j(), kk=k();
			
			//increment with carry, but we must respect the entire
			//field width of I_W, J_W, and K_W
			++ii;
			if (ii>mi) {ii=0; ++jj;}
			if (jj>mi) {jj=0; ++kk;}
			if (kk>mi) {kk=0; ++dd;}
			
			*this = VoxelVertexKey{dd,ii,jj,kk};
			return *this;
		}

	};


	//print debug
	template<uint64_t MAX_DEPTH_, uint64_t BC>
	std::ostream& operator<<(std::ostream& os, const VoxelVertexKey<MAX_DEPTH_, BC> k) {
		os << static_cast<const typename VoxelVertexKey<MAX_DEPTH_,BC>::BASE&>(k);
		os << "linear_index: " << k.linear_index() << "\n";
		os << "coord (x,y,z): (" << k.x() << ", " << k.y() << ", " << k.z() << ")\n";
		return os;
	}
}

