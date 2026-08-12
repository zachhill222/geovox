#pragma once

#include "gutil.hpp"

#include <cstdint>
#include <iostream>


namespace GV {


	namespace Keys{

	///////////////////////////////////////////////////////////////
	/// A namespace for indexing voxel mesh features.
	/// Operate on raw uint64_t types for simd compatibility
	///
	/// Note that some operations allow for a Period template.
	/// 
	/// The Period is a number 0bzyx, where the bit x,y,z sets
	/// the periodic nature of that axis. For example,
	/// Period = 6 = 0b110 is periodic in z and y but not x
	///
	/// Note that specifying a Period does not change the index of any
	/// feature, only which indices are feasible. This is to allow
	/// maximum compatibility between meshes using non-periodic features
	/// and DOFs using periodic features.
	///////////////////////////////////////////////////////////////
	namespace Mesh3D {

		
		///////////////////////////////////////////////////////////
		/// Define constants and bit masks
		///////////////////////////////////////////////////////////
		inline constexpr uint64_t MAX_DEPTH      = 15;							//4 bits reserved for depth
		inline constexpr uint64_t TOTAL_IDX_MASK = (uint64_t{1} << 3*16) - 1;	//bits reserved for the primary feature index
		inline constexpr uint64_t NONE           = 0;							//all valid features have a reserved check bit set

		inline constexpr uint64_t IDX_MASK     = (uint64_t{1} << 16) - 1;	//16 bits for each index
		inline constexpr uint64_t DEPTH_MASK   = (uint64_t{1} << 4 ) - 1;	//4 bits for the depth
		inline constexpr uint64_t FEAT_MASK    = (uint64_t{1} << 2 ) - 1;	//2 bits to encode the feature
		inline constexpr uint64_t SUB_IDX_MASK = (uint64_t{1} << 2)  - 1;	//2 bits to encode a sub-index (e.g. Q2 dofs or DG FEM)
		inline constexpr uint64_t AXIS_MASK    = (uint64_t{1} << 2)  - 1;   //2 bits to encode the axis (for edges and faces)

		inline constexpr uint64_t I_SHIFT     	= 0;
		inline constexpr uint64_t J_SHIFT     	= 16;
		inline constexpr uint64_t K_SHIFT     	= 32;
		inline constexpr uint64_t DEPTH_SHIFT 	= 48;
		inline constexpr uint64_t FEAT_SHIFT  	= 53;
		inline constexpr uint64_t SUB_I_SHIFT	= 55;
		inline constexpr uint64_t SUB_J_SHIFT   = 57;
		inline constexpr uint64_t SUB_K_SHIFT	= 59;
		inline constexpr uint64_t AXIS_SHIFT    = 61;
		
		inline constexpr uint64_t MORTON_BIT    = uint64_t{1} << 63;	//when set to 1, the 48-LSB are a morton index
		inline constexpr uint64_t CHECK_BIT     = uint64_t{1} << 52;	//always set to 1, set to 0 for 'does not exist'

		inline constexpr uint64_t VERTEX_FLAG  = 0;
		inline constexpr uint64_t EDGE_FLAG    = 1;
		inline constexpr uint64_t FACE_FLAG    = 2;
		inline constexpr uint64_t ELEMENT_FLAG = 3;


		// note that the check bit will roll over when incrementing the index out of bounds
		//                                                         |     OR MORTON     |
		//                                                         |-------------------|
		//Layout: M-1 | A-2 | SK-2 | SJ-2 | SI-2 | F-2 | C-1 | D-4 | K-16 | J-16 | I-16|
		//		:   63|   61|   59|    57|    55|    53|   52|   48|    32|    16|    0|
		static_assert(
				(CHECK_BIT ^ MORTON_BIT ^ (AXIS_MASK<<AXIS_SHIFT) ^
				(SUB_IDX_MASK<<SUB_K_SHIFT) ^ (SUB_IDX_MASK<<SUB_J_SHIFT) ^ (SUB_IDX_MASK<<SUB_I_SHIFT) ^
				(FEAT_MASK<<FEAT_SHIFT) ^ (DEPTH_MASK<<DEPTH_SHIFT) ^
				(IDX_MASK<<K_SHIFT) ^ (IDX_MASK<<J_SHIFT) ^ (IDX_MASK<<I_SHIFT) ) == uint64_t(-1),
				"GV::MeshKey - bit packing error");
		

		///////////////////////////////////////////////////////////
		/// Encodng independent information
		///////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr bool Exists(uint64_t key) noexcept {
			return key&CHECK_BIT;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr bool IsMorton(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return key&MORTON_BIT;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr bool IsCartesian(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return !static_cast<bool>(key&MORTON_BIT);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr bool IsVertex(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return ((key>>FEAT_SHIFT)&FEAT_MASK) == VERTEX_FLAG;}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr bool IsEdge(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key))
			return ((key>>FEAT_SHIFT)&FEAT_MASK) == EDGE_FLAG;}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr bool IsFace(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key))
			return ((key>>FEAT_SHIFT)&FEAT_MASK) == FACE_FLAG;}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr bool IsElement(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key))
			return ((key>>FEAT_SHIFT)&FEAT_MASK) == ELEMENT_FLAG;}
		
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t Depth(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return (key>>DEPTH_SHIFT)&DEPTH_MASK;
		}

		GUTIL_DECLARE_SIMD()	//just in case _SIMD is added next to Index*_SIMD
		[[nodiscard]] inline constexpr uint64_t Depth_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return (key>>DEPTH_SHIFT)&DEPTH_MASK;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t Axis(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key))
			return (key>>AXIS_SHIFT)&AXIS_MASK;
		}

		///////////////////////////////////////////////////////////
		/// Encode/decode linear index (morton at each depth)
		/// morton codes are (i,j,k) bits from high to low.
		/// Morton encoding is only supported for elements.
		///
		/// The non-simd versions can take either encoding (fast return),
		/// but to use simd, we need the instructions to be predicable.
		///////////////////////////////////////////////////////////
		[[nodiscard]] inline constexpr uint64_t EncodeElement(uint64_t key) noexcept {
			GUTIL_ASSERT(IsElement(key));
			if (IsMorton(key)) {return key;}

			//no padding ever needed for elements
			const uint64_t ii = (key>>I_SHIFT)&IDX_MASK;
			const uint64_t jj = (key>>J_SHIFT)&IDX_MASK;
			const uint64_t kk = (key>>K_SHIFT)&IDX_MASK;

			uint64_t morton_idx = 0;
			for (int8_t dd = static_cast<int8_t>(Depth(key))-1; dd>=0; --dd) {
				morton_idx <<= 3;
				morton_idx |= (((kk>>dd)&1) << 2) | (((jj>>dd)&1) << 1) | ((ii>>dd)&1) ;
			}

			key |= MORTON_BIT;		//mark that the element is morton encoded
			key &= ~TOTAL_IDX_MASK;	//clear i,j,k
			key |= morton_idx;		//overwrite i,j,k with the morton index
			return key;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t EncodeElement_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(IsElement(key));
			GUTIL_ASSERT(IsCartesian(key));

			//no padding ever needed for elements
			const uint64_t ii = (key>>I_SHIFT)&IDX_MASK;
			const uint64_t jj = (key>>J_SHIFT)&IDX_MASK;
			const uint64_t kk = (key>>K_SHIFT)&IDX_MASK;

			uint64_t morton_idx = 0;
			for (int8_t dd = static_cast<int8_t>(MAX_DEPTH-1); dd>=0; --dd) {
				morton_idx <<= 3;
				morton_idx |= (((kk>>dd)&1) << 2) | (((jj>>dd)&1) << 1) | ((ii>>dd)&1) ;
			}

			key |= MORTON_BIT;		//mark that the element is morton encoded
			key &= ~TOTAL_IDX_MASK;	//clear i,j,k
			key |= morton_idx;		//overwrite i,j,k with the morton index
			return key;
		}

		[[nodiscard]] inline constexpr uint64_t DecodeElement(uint64_t key) noexcept {
			GUTIL_ASSERT(IsElement(key));
			if (IsCartesian(key)) {return key;}

			const uint64_t morton_idx = key & TOTAL_IDX_MASK;
			
			uint64_t ii{0}, jj{0}, kk{0};
			for (int8_t dd = static_cast<int8_t>(Depth(key))-1; dd>=0; --dd) {
				kk <<= 1; kk |= ((morton_idx >> (3*dd+2)) & 1);
				jj <<= 1; jj |= ((morton_idx >> (3*dd+1)) & 1);
				ii <<= 1; ii |= ((morton_idx >> (3*dd  )) & 1);
			}

			key &= ~MORTON_BIT;		//mark that the element is not encoded
			key &= ~TOTAL_IDX_MASK;	//clear i,j,k
			key |= (ii<<I_SHIFT) | (jj<<J_SHIFT) | (kk<<K_SHIFT);
			return key;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t DecodeElement_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(IsElement(key));
			GUTIL_ASSERT(IsMorton(key));

			const uint64_t morton_idx = key & TOTAL_IDX_MASK;
			
			uint64_t ii{0}, jj{0}, kk{0};
			for (int8_t dd = static_cast<int8_t>(MAX_DEPTH-1); dd>=0; --dd) {
				kk <<= 1; kk |= ((morton_idx >> (3*dd+2)) & 1);
				jj <<= 1; jj |= ((morton_idx >> (3*dd+1)) & 1);
				ii <<= 1; ii |= ((morton_idx >> (3*dd  )) & 1);
			}

			key &= ~MORTON_BIT;		//mark that the element is not encoded
			key &= ~TOTAL_IDX_MASK;	//clear i,j,k
			key |= (ii<<I_SHIFT) | (jj<<J_SHIFT) | (kk<<K_SHIFT);
			return key;
		}
		

		///////////////////////////////////////////////////////////
		/// Query functions. If the encoding of the key is known,
		/// it may be preferable to use the simd versions.
		///////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t IndexI_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsCartesian(key));
			return (key>>I_SHIFT)&IDX_MASK;
		}

		[[nodiscard]] inline constexpr uint64_t IndexI(uint64_t key) noexcept {
			return IsElement(key) ? IndexI_SIMD(DecodeElement(key)) : IndexI_SIMD(key);
		}
		
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t IndexJ_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsCartesian(key));
			return (key>>J_SHIFT)&IDX_MASK;
		}

		[[nodiscard]] inline constexpr uint64_t IndexJ(uint64_t key) noexcept {
			return IsElement(key) ? IndexJ_SIMD(DecodeElement(key)) : IndexJ_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t IndexK_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsCartesian(key));
			return (key>>K_SHIFT)&IDX_MASK;
		}

		[[nodiscard]] inline constexpr uint64_t IndexK(uint64_t key) noexcept {
			return IsElement(key) ? IndexK_SIMD(DecodeElement(key)) : IndexK_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t MortonIndex_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsMorton(key));
			GUTIL_ASSERT(IsElement(key));
			return key&TOTAL_IDX_MASK;
		}

		[[nodiscard]] inline constexpr uint64_t MortonIndex(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			return MortonIndex_SIMD(EncodeElement(key));
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t SubIndexI_SIMD(uint64_t key) noexcept{
			GUTIL_ASSERT(Exists(key));
			return (key>>SUB_I_SHIFT)&SUB_IDX_MASK;
		}

		[[nodiscard]] inline constexpr uint64_t SubIndexI(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return IsElement(key) ? SubIndexI_SIMD(DecodeElement(key)) : SubIndexI_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t SubIndexJ_SIMD(uint64_t key) noexcept{
			GUTIL_ASSERT(Exists(key));
			return (key>>SUB_J_SHIFT)&SUB_IDX_MASK;
		}

		[[nodiscard]] inline constexpr uint64_t SubIndexJ(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return IsElement(key) ? SubIndexJ_SIMD(DecodeElement(key)) : SubIndexJ_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t SubIndexK_SIMD(uint64_t key) noexcept{
			GUTIL_ASSERT(Exists(key));
			return (key>>SUB_K_SHIFT)&SUB_IDX_MASK;
		}

		[[nodiscard]] inline constexpr uint64_t SubIndexK(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			return IsElement(key) ? SubIndexK_SIMD(DecodeElement(key)) : SubIndexK_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t MortonIndexPairity_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(IsMorton(key));
			return (key&0b111);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t CartesianIndexPairity_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(IsCartesian(key));
			return ((IndexK_SIMD(key)&1) << 2) | ((IndexJ_SIMD(key)&1) << 1) | ((IndexI_SIMD(key)&1));
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t IndexPairity_SIMD(uint64_t key) noexcept {
			return IsMorton(key) ? MortonIndexPairity_SIMD(key) : CartesianIndexPairity_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t CartesianColor54_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(IsElement(key) && IsCartesian(key));
			const uint64_t d_par = Depth(key)&1;			//low color bit
			const uint64_t i_par = IndexI_SIMD(key)%3;		//
			const uint64_t j_par = IndexJ_SIMD(key)%3;		//	i_par + 3*j_par + 9*k_par are the high color bits
			const uint64_t k_par = IndexK_SIMD(key)%3;		//
			return  ((i_par + 3*(j_par + 3*k_par)) *2) + d_par;
		}



		///////////////////////////////////////////////////////////
		/// Query functions to return a smaller integer width
		///////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint8_t Depth_u8(uint64_t key) noexcept {
			return static_cast<uint8_t>(Depth(key));
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr int8_t Depth_i8(uint64_t key) noexcept {
			return static_cast<int8_t>(Depth(key));
		}
		
		[[nodiscard]] inline constexpr uint16_t IndexI_u16(uint64_t key) noexcept {
			return static_cast<uint16_t>(IndexI(key));
		}

		[[nodiscard]] inline constexpr int16_t IndexI_i16(uint64_t key) noexcept {
			return static_cast<int16_t>(IndexI(key));
		}
		
		[[nodiscard]] inline constexpr uint16_t IndexJ_u16(uint64_t key) noexcept {
			return static_cast<uint16_t>(IndexJ(key));
		}

		[[nodiscard]] inline constexpr int16_t IndexJ_i16(uint64_t key) noexcept {
			return static_cast<int16_t>(IndexJ(key));
		}

		[[nodiscard]] inline constexpr uint16_t IndexK_u16(uint64_t key) noexcept {
			return static_cast<uint16_t>(IndexK(key));
		}

		[[nodiscard]] inline constexpr int16_t IndexK_i16(uint64_t key) noexcept {
			return static_cast<int16_t>(IndexK(key));
		}

		[[nodiscard]] inline constexpr uint8_t SubIndexI_u8(uint64_t key) noexcept{
			return static_cast<uint8_t>(SubIndexI(key));
		}

		[[nodiscard]] inline constexpr int8_t SubIndexI_i8(uint64_t key) noexcept{
			return static_cast<int8_t>(SubIndexI(key));
		}

		[[nodiscard]] inline constexpr uint8_t SubIndexJ_u8(uint64_t key) noexcept{
			return static_cast<uint8_t>(SubIndexJ(key));
		}

		[[nodiscard]] inline constexpr int8_t SubIndexJ_i8(uint64_t key) noexcept{
			return static_cast<int8_t>(SubIndexJ(key));
		}

		[[nodiscard]] inline constexpr uint8_t SubIndexK_u8(uint64_t key) noexcept{
			return static_cast<uint8_t>(SubIndexK(key));
		}

		[[nodiscard]] inline constexpr int8_t SubIndexK_i8(uint64_t key) noexcept{
			return static_cast<int8_t>(SubIndexK(key));
		}


		///////////////////////////////////////////////////////////
		/// Factory methods
		///////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t MakeElement(uint64_t dd, uint64_t ii, uint64_t jj, uint64_t kk) {
			const uint64_t n_el = uint64_t{1} << dd;
			return (dd<=MAX_DEPTH && ii<n_el && jj<n_el && kk<n_el) ?
					(CHECK_BIT | (ELEMENT_FLAG<<FEAT_SHIFT) | (dd<<DEPTH_SHIFT) |
					 	(kk<<K_SHIFT) | (jj<<J_SHIFT) | (ii<<I_SHIFT)) : 0;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t MakeElement(uint64_t dd, uint64_t morton_idx) {
			const uint64_t total_els = uint64_t{1} << 3*dd;	//total number of elements at this depth
			return (dd<=MAX_DEPTH && morton_idx < total_els) ? 
					(CHECK_BIT | MORTON_BIT | (ELEMENT_FLAG<<FEAT_SHIFT) | (dd<<DEPTH_SHIFT) | morton_idx) : 0;
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)
		[[nodiscard]] inline constexpr uint64_t MakeVertex(uint64_t dd, uint64_t ii, uint64_t jj, uint64_t kk) {
			const uint64_t n_el = uint64_t{1} << dd; //largest non-periodic index

			//handle periodic axes. note the lower index is the cononical representation
			if constexpr (Period&0b001) { ii%=n_el; }
			if constexpr (Period&0b010) { jj%=n_el; }
			if constexpr (Period&0b100) { kk%=n_el; }

			return (dd<=MAX_DEPTH && ii<=n_el && jj<=n_el && kk<=n_el) ?
					(CHECK_BIT | (VERTEX_FLAG<<FEAT_SHIFT) | (dd<<DEPTH_SHIFT) |
					 	(kk<<K_SHIFT) | (jj<<J_SHIFT) | (ii<<I_SHIFT)) : 0;
		}

		static_assert(EncodeElement(MakeElement(9,123,231,312)) == EncodeElement_SIMD(MakeElement(9,123,231,312)));
		static_assert(DecodeElement(MakeElement(7,123456)) == DecodeElement_SIMD(MakeElement(7,123456)));
		static_assert(DecodeElement(EncodeElement(MakeElement(9,123,231,312))) == MakeElement(9,123,231,312));
		static_assert(EncodeElement(DecodeElement(MakeElement(7,123456))) == MakeElement(7,123456));


		///////////////////////////////////////////////////////////
		/// Encode/decode elements and vertices into an index that is
		/// contiguous across depths.
		/// use ~CHECK_BIT (half of max uint64_t) as an 'invalid input'
		/// return so that if it is accidentally treated as a key, then
		/// an assert should trigger.
		///////////////////////////////////////////////////////////

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t ElementsBelowDepth(uint64_t depth) noexcept {
			return ((uint64_t{1} << (3*depth)) - 1)/uint64_t{7}; //geometric sum up to depth-1
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t GlobalElementIndex_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsMorton(key));
			GUTIL_ASSERT(IsElement(key));
			const uint64_t dd = Depth(key);
			const uint64_t offset = ElementsBelowDepth(dd);
			
			const uint64_t morton_idx = MortonIndex_SIMD(key);
			const uint64_t depth_els = uint64_t{1} << (3*dd);
			return (morton_idx < depth_els) ? offset+morton_idx : ~CHECK_BIT;
		}

		[[nodiscard]] inline constexpr uint64_t GlobalElementIndex(uint64_t key) {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			return GlobalElementIndex_SIMD(EncodeElement(key));
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t ElementFromGlobalIndex_SIMD(uint64_t index) noexcept {
			uint64_t dd = 0;
			for (uint64_t d=0; d<MAX_DEPTH; ++d) {
				const uint64_t layer_els = uint64_t{1} << (3*dd);
				dd    += (index>=layer_els) ? 1 : 0;
				index -= (index>=layer_els) ? layer_els : 0;
			}
			return MakeElement(dd, index);	//index too large checked here
		}

		static_assert(ElementFromGlobalIndex_SIMD(0) == MakeElement(0,0));
		static_assert(ElementFromGlobalIndex_SIMD(0) == EncodeElement(MakeElement(0,0,0,0)));
		static_assert(GlobalElementIndex_SIMD(MakeElement(0,0)) == 0);
		static_assert(GlobalElementIndex_SIMD(MakeElement(0,0,0,0)) == 0);
		static_assert(ElementFromGlobalIndex_SIMD(10) == MakeElement(2,1));
		static_assert(ElementFromGlobalIndex_SIMD(10) == EncodeElement(MakeElement(2,1,0,0)));
		static_assert(GlobalElementIndex_SIMD(MakeElement(2,1)) == 10);
		static_assert(GlobalElementIndex_SIMD(EncodeElement(MakeElement(2,1,0,0))) == 10);
		static_assert(ElementFromGlobalIndex_SIMD(GlobalElementIndex_SIMD(EncodeElement(MakeElement(9,123,231,312)))) ==
							EncodeElement(MakeElement(9,123,231,312)));



		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t VerticesBelowDepth(uint64_t dd) noexcept {
			//(2^d + 1)^3 vertices per depth. expand and sum from 0 to dd-1
			return  ((uint64_t{1} << (3*dd)) - 1)/7 
				 +   (uint64_t{1} << (2*dd)) 
				 + 3*(uint64_t{1} << dd)
				 + dd - 4;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t GlobalVertexIndex_SIMD(uint64_t key) {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			const uint64_t dd = Depth(key);
			const uint64_t offset = VerticesBelowDepth(dd);
			
			const uint64_t N = (uint64_t{1} << dd) + 1; //encode depth linear index as a base N number
			const uint64_t idx = IndexI_SIMD(key) + N*(IndexJ_SIMD(key) + N*IndexK_SIMD(key));

			return (idx < N*N*N) ? offset+idx : ~CHECK_BIT;
		}

		[[nodiscard]] inline constexpr uint64_t GlobalVertexIndex(uint64_t key) {
			//just a naming convention for readability
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			return GlobalVertexIndex_SIMD(key);
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)
		[[nodiscard]] inline constexpr uint64_t VertexFromGlobalIndex_SIMD(uint64_t index) {
			uint64_t dd = 0;
			for (uint64_t d=0; d<MAX_DEPTH; ++d) {
				const uint64_t next_threshold = VerticesBelowDepth(dd+1);
				dd += (index>=next_threshold) ? 1 : 0;
			}
			const uint64_t offset = VerticesBelowDepth(dd);
			const uint64_t N = (uint64_t{1} << dd) + 1;
			index -= offset;
			const uint64_t ii = index%N; index/=N;
			const uint64_t jj = index%N; index/=N;
			// const uint64_t kk = index;
			return MakeVertex<Period>(dd, ii, jj, index); //index too large checked here (k will overflow)
		}

		static_assert(VertexFromGlobalIndex_SIMD(0) == MakeVertex(0,0,0,0));
		static_assert(GlobalVertexIndex_SIMD(MakeVertex(0,0,0,0)) == 0);
		static_assert(VertexFromGlobalIndex_SIMD(10) == MakeVertex(1,2,0,0));
		static_assert(GlobalVertexIndex_SIMD(MakeVertex(1,2,0,0)) == 10);
		static_assert(VertexFromGlobalIndex_SIMD(GlobalVertexIndex_SIMD(MakeVertex(9,123,231,312))) ==
							MakeVertex(9,123,231,312));


		///////////////////////////////////////////////////////////
		/// Use the morton index to define parent/child relations
		/// Note that the morton key for the first child is 8 times
		/// the morton key for the parent and that the children are
		/// contiguous.
		///////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t ElementChildStart_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			GUTIL_ASSERT(IsMorton(key));	//the elements are only contiguous in morton index
			const uint64_t morton_idx = MortonIndex(key) << 3;
			key &= ~TOTAL_IDX_MASK;
			key |= morton_idx;
			
			const uint64_t dd = Depth(key) + 1;
			key &= ~(DEPTH_MASK<<DEPTH_SHIFT);
			return (dd <= MAX_DEPTH) ? ( key | (dd<<DEPTH_SHIFT)) : 0;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t ElementParent_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			GUTIL_ASSERT(IsMorton(key));	//keep all element hierarcy relations in morton codes by default
			const uint64_t morton_idx = MortonIndex(key) >> 3;
			key &= ~TOTAL_IDX_MASK;
			key |= morton_idx;

			const uint64_t dd = Depth(key)-1; //overflow if the depth was 0 (no parent)
			key &= ~(DEPTH_MASK<<DEPTH_SHIFT);
			return (dd <= MAX_DEPTH) ? ( key | (dd<<DEPTH_SHIFT)) : 0;
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t ElementParentCartesian_SIMD(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			GUTIL_ASSERT(IsCartesian(key));
			#ifndef NDEBUG
				uint64_t p = ElementParent_SIMD(EncodeElement_SIMD(key));
			#endif

			//truncate each index and move back to the original position (divided by 2)
			//                                  get the index bits      /2    shift into place
			const uint64_t ii_in_place = (((key >> I_SHIFT) & IDX_MASK) >> 1) << I_SHIFT;
			const uint64_t jj_in_place = (((key >> J_SHIFT) & IDX_MASK) >> 1) << J_SHIFT;
			const uint64_t kk_in_place = (((key >> K_SHIFT) & IDX_MASK) >> 1) << K_SHIFT;
			const uint64_t dd          = Depth(key)-1; //overflow if the depth was 0 (no parent)
			
			constexpr uint64_t DEPTH_INDEX_MASK = (DEPTH_MASK << DEPTH_SHIFT) | TOTAL_IDX_MASK;
			key &= ~DEPTH_INDEX_MASK;
			key |= (dd<<DEPTH_SHIFT) | ii_in_place | jj_in_place | kk_in_place;

			GUTIL_ASSERT( (!Exists(p) && dd>MAX_DEPTH) || (DecodeElement_SIMD(p) == key));
			return (dd <= MAX_DEPTH) ? key : 0;
		}


		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t VertexParent(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			const uint64_t dd = Depth(key);  if (dd==0){return 0;}
			const uint64_t ii = IndexI(key); if (ii&1) {return 0;}
			const uint64_t jj = IndexJ(key); if (jj&1) {return 0;}
			const uint64_t kk = IndexK(key); if (kk&1) {return 0;}
			return MakeVertex(dd-1, ii>>1, jj>>1, kk>>1);	
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t VertexChild(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			return MakeVertex(Depth(key)+1, IndexI(key)<<1, IndexJ(key)<<1, IndexK(key)<<1);
		}

		GUTIL_DECLARE_SIMD()
		[[nodiscard]] inline constexpr uint64_t ReducedVertex_SIMD(uint64_t key) noexcept {
			//it is useful to have a method to call on vertices that don't exist
			//for uniform simd operations
			for (uint64_t dd=0; dd<MAX_DEPTH; ++dd) {
				uint64_t parent = VertexParent(key);
				key = Exists(parent) ? parent : key;
			}
			return key;
		}

		[[nodiscard]] inline constexpr uint64_t ReducedVertex(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			for (uint64_t dd=0; dd<MAX_DEPTH; ++dd) {
				uint64_t parent = VertexParent(key);
				if (!Exists(parent)) { return key; }
				else {key = parent;}
			}
			return key;
		}

		///////////////////////////////////////////////////////////
		/// Debug methods
		///////////////////////////////////////////////////////////
		template<uint8_t Period=0> requires(Period<8)
		[[nodiscard]] inline constexpr bool IsValid(uint64_t key) noexcept {
			if (!Exists(key)) { return false; }
			if (IsElement(key)) {
				key = DecodeElement(key);
				const uint64_t n_el = uint64_t{1}<<Depth(key);
				return (IndexI(key)<n_el) && (IndexJ(key)<n_el) && (IndexK(key)<n_el);
			}
			else if (IsVertex(key)) {
				if (!IsCartesian(key)) {GUTIL_ABORT("vertices must be in cartesian coordinates"); return false;}
				const uint64_t n_el = uint64_t{1}<<Depth(key);
				const uint64_t n_x = (Period&0b001) ? n_el-1 : n_el;
				const uint64_t n_y = (Period&0b010) ? n_el-1 : n_el;
				const uint64_t n_z = (Period&0b100) ? n_el-1 : n_el;
				return (IndexI_SIMD(key)<=n_x) && (IndexJ_SIMD(key)<=n_y) && (IndexK_SIMD(key)<=n_z);
			}
			GUTIL_ABORT("Only elements and vertices are supported for now");
			return false;
		}

		[[nodiscard]] inline std::string ToString(uint64_t key) noexcept {
			if (!Exists(key)) {return "None";}
			if (IsElement(key)) { return IsMorton(key) ? 
						"Element(" + std::to_string(Depth(key)) + ", " + std::to_string(MortonIndex(key)) + ")" :
						"Element(" + std::to_string(Depth(key)) + ", " + std::to_string(IndexI(key)) + ", " +
								std::to_string(IndexJ(key)) + ", " + std::to_string(IndexK(key)) + ")";}
			else if (IsVertex(key)) { return IsMorton(key) ?
						"Vertex(" + std::to_string(Depth(key)) + ", " + std::to_string(MortonIndex(key)) + ")" :
						"Vertex(" + std::to_string(Depth(key)) + ", " + std::to_string(IndexI(key)) + ", " +
								std::to_string(IndexJ(key)) + ", " + std::to_string(IndexK(key)) + ")";}
			else { return "Not Implemented"; }
		}

		template<typename T>
		[[nodiscard]] inline constexpr gutil::Point<3,T> NormalizedCoordinate(uint64_t key) noexcept {
			//get the coordinate of the vertex in the [0,1]^3 reference box
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			const T scale = gutil::ldexp(T{1}, -Depth_i8(key));
			return {scale * static_cast<T>(IndexI(key)),
					scale * static_cast<T>(IndexJ(key)),
					scale * static_cast<T>(IndexK(key))};
		}

		template<typename T>
		[[nodiscard]] inline constexpr gutil::Point<3,T> NormalizedCenter(uint64_t key) noexcept {
			//get the coordinate of the center in the [0,1]^3 reference box
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			key = DecodeElement(key);
			const T scale = gutil::ldexp(T{1}, -Depth_i8(key));
			return {T{0.5}*scale + scale*static_cast<T>(IndexI(key)),
					T{0.5}*scale + scale*static_cast<T>(IndexJ(key)),
					T{0.5}*scale + scale*static_cast<T>(IndexK(key))};
		}


		///////////////////////////////////////////////////////////
		/// Element-Element adjacency
		///
		/// Adapt the im/p1 values for periodicity as needed.
		/// Invalid neighbors will overflow and return 0.
		///////////////////////////////////////////////////////////
		GUTIL_DECLARE_SIMD()
		constexpr uint64_t GetElementSiblingsMorton_SIMD(uint64_t key) {
			//just clear the last three bits, then the result and the next 7 are siblings
			constexpr uint64_t mask = ~uint64_t(0b111);
			return key&mask;
		}


		template<typename T=uint64_t, uint8_t Period=0> requires (Period<8)
		[[nodiscard]] constexpr std::array<T,26> GetElementNeighbors(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			key = DecodeElement(key);

			const uint64_t dd = Depth(key);
			const uint64_t ii = IndexI_SIMD(key); uint64_t im1=ii-1; uint64_t ip1=ii+1;
			const uint64_t jj = IndexJ_SIMD(key); uint64_t jm1=jj-1; uint64_t jp1=jj+1;
			const uint64_t kk = IndexK_SIMD(key); uint64_t km1=kk-1; uint64_t kp1=kk+1;

			[[maybe_unused]] const uint64_t N = uint64_t{1} << dd;
			if constexpr (Period&0b001) {im1%=N; ip1%=N;}
			if constexpr (Period&0b010) {jm1%=N; jp1%=N;}
			if constexpr (Period&0b100) {km1%=N; kp1%=N;}

			return {
				//bottom slice
				T{MakeElement(dd, im1, jm1, km1)},
				T{MakeElement(dd, ii , jm1, km1)},
				T{MakeElement(dd, ip1, jm1, km1)},
				T{MakeElement(dd, im1, jj , km1)},
				T{MakeElement(dd, ii , jj , km1)},
				T{MakeElement(dd, ip1, jj , km1)},
				T{MakeElement(dd, im1, jp1, km1)},
				T{MakeElement(dd, ii , jp1, km1)},
				T{MakeElement(dd, ip1, jp1, km1)},

				//middle slice (remove center)
				T{MakeElement(dd, im1, jm1, kk)},
				T{MakeElement(dd, ii , jm1, kk)},
				T{MakeElement(dd, ip1, jm1, kk)},
				T{MakeElement(dd, im1, jj , kk)},
				// T{MakeElement(dd, ii , jj , kk)},
				T{MakeElement(dd, ip1, jj , kk)},
				T{MakeElement(dd, im1, jp1, kk)},
				T{MakeElement(dd, ii , jp1, kk)},
				T{MakeElement(dd, ip1, jp1, kk)},

				//top slice
				T{MakeElement(dd, im1, jm1, kp1)},
				T{MakeElement(dd, ii , jm1, kp1)},
				T{MakeElement(dd, ip1, jm1, kp1)},
				T{MakeElement(dd, im1, jj , kp1)},
				T{MakeElement(dd, ii , jj , kp1)},
				T{MakeElement(dd, ip1, jj , kp1)},
				T{MakeElement(dd, im1, jp1, kp1)},
				T{MakeElement(dd, ii , jp1, kp1)},
				T{MakeElement(dd, ip1, jp1, kp1)}
			};
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)
		constexpr void GetElementNeighbors_SIMD(uint64_t key, uint64_t* ptr) noexcept {
			GUTIL_ASSERT(ptr!=nullptr);
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			GUTIL_ASSERT(IsCartesian(key));

			const uint64_t dd = Depth(key);
			const uint64_t ii = IndexI_SIMD(key); uint64_t im1=ii-1; uint64_t ip1=ii+1;
			const uint64_t jj = IndexJ_SIMD(key); uint64_t jm1=jj-1; uint64_t jp1=jj+1;
			const uint64_t kk = IndexK_SIMD(key); uint64_t km1=kk-1; uint64_t kp1=kk+1;

			[[maybe_unused]] const uint64_t N = uint64_t{1} << dd;
			if constexpr (Period&0b001) {im1%=N; ip1%=N;}
			if constexpr (Period&0b010) {jm1%=N; jp1%=N;}
			if constexpr (Period&0b100) {km1%=N; kp1%=N;}

			//bottom slice
			ptr[0]  = MakeElement(dd, im1, jm1, km1);
			ptr[1]  = MakeElement(dd, ii , jm1, km1);
			ptr[2]  = MakeElement(dd, ip1, jm1, km1);
			ptr[3]  = MakeElement(dd, im1, jj , km1);
			ptr[4]  = MakeElement(dd, ii , jj , km1);
			ptr[5]  = MakeElement(dd, ip1, jj , km1);
			ptr[6]  = MakeElement(dd, im1, jp1, km1);
			ptr[7]  = MakeElement(dd, ii , jp1, km1);
			ptr[8]  = MakeElement(dd, ip1, jp1, km1);

			//middle slice (remove center)
			ptr[9]  = MakeElement(dd, im1, jm1, kk);
			ptr[10] = MakeElement(dd, ii , jm1, kk);
			ptr[11] = MakeElement(dd, ip1, jm1, kk);
			ptr[12] = MakeElement(dd, im1, jj , kk);
			//ptr[] = MakeElement(dd, ii , jj , kk);
			ptr[13] = MakeElement(dd, ip1, jj , kk);
			ptr[14] = MakeElement(dd, im1, jp1, kk);
			ptr[15] = MakeElement(dd, ii , jp1, kk);
			ptr[16] = MakeElement(dd, ip1, jp1, kk);

			//top slice
			ptr[17] = MakeElement(dd, im1, jm1, kp1);
			ptr[18] = MakeElement(dd, ii , jm1, kp1);
			ptr[19] = MakeElement(dd, ip1, jm1, kp1);
			ptr[20] = MakeElement(dd, im1, jj , kp1);
			ptr[21] = MakeElement(dd, ii , jj , kp1);
			ptr[22] = MakeElement(dd, ip1, jj , kp1);
			ptr[23] = MakeElement(dd, im1, jp1, kp1);
			ptr[24] = MakeElement(dd, ii , jp1, kp1);
			ptr[25] = MakeElement(dd, ip1, jp1, kp1);
		}


		///////////////////////////////////////////////////////////
		/// Vertex-Vertex adjacency
		///
		/// Adapt the im/p1 values for periodicity as needed.
		/// Invalid neighbors will overflow and return 0.
		///////////////////////////////////////////////////////////
		template<typename T=uint64_t, uint8_t Period=0> requires(Period<8)
		[[nodiscard]] constexpr std::array<T,26> GetVertexNeighbors(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			const uint64_t dd = Depth(key);
			const uint64_t ii = IndexI_SIMD(key); uint64_t im1=ii-1; uint64_t ip1=ii+1;
			const uint64_t jj = IndexJ_SIMD(key); uint64_t jm1=jj-1; uint64_t jp1=jj+1;
			const uint64_t kk = IndexK_SIMD(key); uint64_t km1=kk-1; uint64_t kp1=kk+1;

			[[maybe_unused]] const uint64_t N = (uint64_t{1} << dd) + 1;
			if constexpr (Period&0b001) {im1%=N; ip1%=N;}
			if constexpr (Period&0b010) {jm1%=N; jp1%=N;}
			if constexpr (Period&0b100) {km1%=N; kp1%=N;}

			return {
				//bottom slice
				T{MakeVertex<Period>(dd, im1, jm1, km1)},
				T{MakeVertex<Period>(dd, ii , jm1, km1)},
				T{MakeVertex<Period>(dd, ip1, jm1, km1)},
				T{MakeVertex<Period>(dd, im1, jj , km1)},
				T{MakeVertex<Period>(dd, ii , jj , km1)},
				T{MakeVertex<Period>(dd, ip1, jj , km1)},
				T{MakeVertex<Period>(dd, im1, jp1, km1)},
				T{MakeVertex<Period>(dd, ii , jp1, km1)},
				T{MakeVertex<Period>(dd, ip1, jp1, km1)},

				//middle slice (remove center)
				T{MakeVertex<Period>(dd, im1, jm1, kk)},
				T{MakeVertex<Period>(dd, ii , jm1, kk)},
				T{MakeVertex<Period>(dd, ip1, jm1, kk)},
				T{MakeVertex<Period>(dd, im1, jj , kk)},
				// T{MakeVertex<Period>(dd, ii , jj , kk)},
				T{MakeVertex<Period>(dd, ip1, jj , kk)},
				T{MakeVertex<Period>(dd, im1, jp1, kk)},
				T{MakeVertex<Period>(dd, ii , jp1, kk)},
				T{MakeVertex<Period>(dd, ip1, jp1, kk)},

				//top slice
				T{MakeVertex<Period>(dd, im1, jm1, kp1)},
				T{MakeVertex<Period>(dd, ii , jm1, kp1)},
				T{MakeVertex<Period>(dd, ip1, jm1, kp1)},
				T{MakeVertex<Period>(dd, im1, jj , kp1)},
				T{MakeVertex<Period>(dd, ii , jj , kp1)},
				T{MakeVertex<Period>(dd, ip1, jj , kp1)},
				T{MakeVertex<Period>(dd, im1, jp1, kp1)},
				T{MakeVertex<Period>(dd, ii , jp1, kp1)},
				T{MakeVertex<Period>(dd, ip1, jp1, kp1)}
			};
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)
		constexpr void GetVertexNeighbors_SIMD(uint64_t key, uint64_t* ptr) noexcept {
			GUTIL_ASSERT(ptr!=nullptr);
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			const uint64_t dd = Depth(key);
			const uint64_t ii = IndexI_SIMD(key); uint64_t im1=ii-1; uint64_t ip1=ii+1;
			const uint64_t jj = IndexJ_SIMD(key); uint64_t jm1=jj-1; uint64_t jp1=jj+1;
			const uint64_t kk = IndexK_SIMD(key); uint64_t km1=kk-1; uint64_t kp1=kk+1;

			[[maybe_unused]] const uint64_t N = (uint64_t{1} << dd) + 1;
			if constexpr (Period&0b001) {im1%=N; ip1%=N;}
			if constexpr (Period&0b010) {jm1%=N; jp1%=N;}
			if constexpr (Period&0b100) {km1%=N; kp1%=N;}

			//bottom slice
			ptr[0]  = MakeVertex<Period>(dd, im1, jm1, km1);
			ptr[1]  = MakeVertex<Period>(dd, ii , jm1, km1);
			ptr[2]  = MakeVertex<Period>(dd, ip1, jm1, km1);
			ptr[3]  = MakeVertex<Period>(dd, im1, jj , km1);
			ptr[4]  = MakeVertex<Period>(dd, ii , jj , km1);
			ptr[5]  = MakeVertex<Period>(dd, ip1, jj , km1);
			ptr[6]  = MakeVertex<Period>(dd, im1, jp1, km1);
			ptr[7]  = MakeVertex<Period>(dd, ii , jp1, km1);
			ptr[8]  = MakeVertex<Period>(dd, ip1, jp1, km1);

			//middle slice (remove center)
			ptr[9]  = MakeVertex<Period>(dd, im1, jm1, kk);
			ptr[10] = MakeVertex<Period>(dd, ii , jm1, kk);
			ptr[11] = MakeVertex<Period>(dd, ip1, jm1, kk);
			ptr[12] = MakeVertex<Period>(dd, im1, jj , kk);
			//ptr[] = MakeVertex<Period>(dd, ii , jj , kk);
			ptr[13] = MakeVertex<Period>(dd, ip1, jj , kk);
			ptr[14] = MakeVertex<Period>(dd, im1, jp1, kk);
			ptr[15] = MakeVertex<Period>(dd, ii , jp1, kk);
			ptr[16] = MakeVertex<Period>(dd, ip1, jp1, kk);

			//top slice
			ptr[17] = MakeVertex<Period>(dd, im1, jm1, kp1);
			ptr[18] = MakeVertex<Period>(dd, ii , jm1, kp1);
			ptr[19] = MakeVertex<Period>(dd, ip1, jm1, kp1);
			ptr[20] = MakeVertex<Period>(dd, im1, jj , kp1);
			ptr[21] = MakeVertex<Period>(dd, ii , jj , kp1);
			ptr[22] = MakeVertex<Period>(dd, ip1, jj , kp1);
			ptr[23] = MakeVertex<Period>(dd, im1, jp1, kp1);
			ptr[24] = MakeVertex<Period>(dd, ii , jp1, kp1);
			ptr[25] = MakeVertex<Period>(dd, ip1, jp1, kp1);
		}



		///////////////////////////////////////////////////////////
		/// Vertex-Element adjacency
		///////////////////////////////////////////////////////////
		template<typename T=uint64_t, uint8_t Period=0> requires(Period<8)
		[[nodiscard]] inline constexpr std::array<T,8> GetElementsOfVertex(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));
			const uint64_t dd = Depth(key);
			
			const uint64_t ii = IndexI(key); uint64_t im1 = ii-1;
			const uint64_t jj = IndexJ(key); uint64_t jm1 = jj-1;
			const uint64_t kk = IndexK(key); uint64_t km1 = kk-1;

			[[maybe_unused]] const uint64_t N = uint64_t{1} << dd;
			if constexpr (Period&0b001) {im1%=N;}
			if constexpr (Period&0b010) {jm1%=N;}
			if constexpr (Period&0b100) {km1%=N;}

			//MakeElement will set any invalid elements to 0
			return {
				T{MakeElement(dd, im1, jm1, km1)},
				T{MakeElement(dd, ii,  jm1, km1)},
				T{MakeElement(dd, im1, jj,  km1)},
				T{MakeElement(dd, ii,  jj,  km1)},
				T{MakeElement(dd, im1, jm1, kk )},
				T{MakeElement(dd, ii,  jm1, kk )},
				T{MakeElement(dd, im1, jj,  kk )},
				T{MakeElement(dd, ii,  jj,  kk )}
			};
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)
		inline constexpr void GetElementsOfVertex_SIMD(uint64_t key, uint64_t* ptr) {
			//ptr must be a pointer to the start of 8 allocated uint64_t values
			GUTIL_ASSERT(ptr!=nullptr);
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsVertex(key));
			GUTIL_ASSERT(IsCartesian(key));

			const uint64_t dd = Depth(key);
			const uint64_t ii = IndexI_SIMD(key); uint64_t im1 = ii-1;
			const uint64_t jj = IndexJ_SIMD(key); uint64_t jm1 = jj-1;
			const uint64_t kk = IndexK_SIMD(key); uint64_t km1 = kk-1;

			[[maybe_unused]] const uint64_t N = uint64_t{1} << dd;
			if constexpr (Period&0b001) {im1%=N;}
			if constexpr (Period&0b010) {jm1%=N;}
			if constexpr (Period&0b100) {km1%=N;}

			//MakeElement will set any invalid elements to 0
			*(ptr+0) = MakeElement(dd, im1, jm1, km1);
			*(ptr+1) = MakeElement(dd, ii , jm1, km1);
			*(ptr+2) = MakeElement(dd, im1, jj , km1);
			*(ptr+3) = MakeElement(dd, ii , jj , km1);
			*(ptr+4) = MakeElement(dd, im1, jm1, kk );
			*(ptr+5) = MakeElement(dd, ii , jm1, kk );
			*(ptr+6) = MakeElement(dd, im1, jj , kk );
			*(ptr+7) = MakeElement(dd, ii , jj , kk );
		}

		template<typename T=uint64_t, uint8_t Period=0> requires(Period<8)
		[[nodiscard]] inline constexpr std::array<T,8> GetVerticesOfElement(uint64_t key) noexcept {
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			key = DecodeElement(key);

			const uint64_t dd = Depth(key);
			const uint64_t ii = IndexI(key); uint64_t ip1=ii+1;
			const uint64_t jj = IndexJ(key); uint64_t jp1=jj+1;
			const uint64_t kk = IndexK(key); uint64_t kp1=kk+1;

			[[maybe_unused]] const uint64_t N = (uint64_t{1} << dd) + 1;
			if constexpr (Period&0b001) {ip1%=N;}
			if constexpr (Period&0b010) {jp1%=N;}
			if constexpr (Period&0b100) {kp1%=N;}

			//MakeVertex will set any invalid elements to 0
			return {
				T{MakeVertex<Period>(dd, ii , jj , kk )},
				T{MakeVertex<Period>(dd, ip1, jj , kk )},
				T{MakeVertex<Period>(dd, ii , jp1, kk )},
				T{MakeVertex<Period>(dd, ip1, jp1, kk )},
				T{MakeVertex<Period>(dd, ii , jj , kp1)},
				T{MakeVertex<Period>(dd, ip1, jj , kp1)},
				T{MakeVertex<Period>(dd, ii , jp1, kp1)},
				T{MakeVertex<Period>(dd, ip1, jp1, kp1)}
			};
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)
		inline constexpr void GetVerticesOfElement_SIMD(uint64_t key, uint64_t* ptr) noexcept {
			//ptr must be a pointer to the start of 8 allocated uint64_t values
			GUTIL_ASSERT(ptr!=nullptr);
			GUTIL_ASSERT(Exists(key));
			GUTIL_ASSERT(IsElement(key));
			GUTIL_ASSERT(IsCartesian(key));

			const uint64_t dd = Depth(key);
			const uint64_t ii = IndexI_SIMD(key); uint64_t ip1 = ii+1;
			const uint64_t jj = IndexJ_SIMD(key); uint64_t jp1 = jj+1;
			const uint64_t kk = IndexK_SIMD(key); uint64_t kp1 = kk+1;

			//MakeVertex will set any invalid elements to 0
			*(ptr+0) = MakeVertex<Period>(dd, ii , jj , kk );	//local index 0b000
			*(ptr+1) = MakeVertex<Period>(dd, ip1, jj , kk );	//local index 0b100
			*(ptr+2) = MakeVertex<Period>(dd, ii , jp1, kk );	//local index 0b010
			*(ptr+3) = MakeVertex<Period>(dd, ip1, jp1, kk );	//local index 0b110
			*(ptr+4) = MakeVertex<Period>(dd, ii , jj , kp1);	//local index 0b001
			*(ptr+5) = MakeVertex<Period>(dd, ip1, jj , kp1);	//local index 0b101
			*(ptr+6) = MakeVertex<Period>(dd, ii , jp1, kp1);	//local index 0b011
			*(ptr+7) = MakeVertex<Period>(dd, ip1, jp1, kp1);	//local index 0b111
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)		
		[[nodiscard]] inline constexpr uint8_t GetLocalVertexNumberCartesian_SIMD(uint64_t el, uint64_t vtx) noexcept {
			GUTIL_ASSERT(Exists(el) && IsElement(el) && IsCartesian(el));
			GUTIL_ASSERT(Exists(vtx) && IsVertex(vtx) && IsCartesian(vtx));
			GUTIL_ASSERT(Depth(el) == Depth(vtx));

			//use index pairity to determine the vertex number
			//suppose vtx has last bits of its index V=0bkji, and the last bits of each index of el are E=0bKJI
			//and the local index I=0bzyx.
			//Looking at GetVerticesOfElement_SIMD, we see that z=0 when k==K and z=1 when k!=K, so we set z = k xor K
			//the same follows for the other bits.

			//Note the logic is the same for periodic vertices, just the meaning of WHERE the local vertex is changes
			//the template is included for debugging only.

			const uint8_t V = ((IndexK_SIMD(vtx)&1) << 2) | ((IndexJ_SIMD(vtx)&1) << 1) | (IndexI_SIMD(vtx)&1);
			const uint8_t E = ((IndexK_SIMD(el)&1)  << 2) | ((IndexJ_SIMD(el)&1)  << 1) | (IndexI_SIMD(el)&1);

			#ifndef NDEBUG
				uint64_t check = GetVerticesOfElement<uint64_t,Period>(el)[V^E];
				if (check!=vtx) {
					std::cout << "el:    " << print_bytes(el) << "\n";
					std::cout << "vtx:   " << print_bytes(vtx) << "\n";
					std::cout << "local: " << (int) (V^E) << "\n";
					std::cout << "check: " << print_bytes(check) << "\n";
				}
				GUTIL_ASSERT(check==vtx);
			#endif
			return V^E;
		}

		GUTIL_DECLARE_SIMD()
		template<uint8_t Period=0> requires(Period<8)
		[[nodiscard]] inline constexpr uint8_t GetLocalVertexNumberMorton_SIMD(uint64_t el, uint64_t vtx) noexcept {
			GUTIL_ASSERT(Exists(el) && IsElement(el) && IsMorton(el));
			GUTIL_ASSERT(Exists(vtx) && IsVertex(vtx) && IsCartesian(vtx));
			GUTIL_ASSERT(Depth(el) == Depth(vtx));

			//use index pairity to determine the vertex number
			//suppose vtx has last bits of its index V=0bkji, and the last bits of each index of el are E=0bKJI
			//and the local index I=0bzyx.
			//Looking at GetVerticesOfElement_SIMD, we see that z=0 when k==K and z=1 when k!=K, so we set z = k xor K
			//the same follows for the other bits.

			//Note the logic is the same for periodic vertices, just the meaning of WHERE the local vertex is changes
			//the template is included for debugging only.

			const uint8_t V = ((IndexK_SIMD(vtx)&1) << 2) | ((IndexJ_SIMD(vtx)&1) << 1) | (IndexI_SIMD(vtx)&1);
			const uint8_t E = el & 0b111; //note the last index bits are immediately accessible

			#ifndef NDEBUG
				uint64_t check = GetVerticesOfElement<uint64_t,Period>(el)[V^E];
				GUTIL_ASSERT(check==vtx);
			#endif
			return V^E;
		}
		
	}//Mesh3D
}//Keys
}//GV