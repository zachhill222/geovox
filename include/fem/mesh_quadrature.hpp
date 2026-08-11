#pragma once

#include "gutil.hpp"

#include "util/util.hpp"
#include "mesh/mesh.hpp"



namespace GV {


	//////////////////////////////////////////////////////////////////////
	/// A helper class to handle essential operations that interact with the mesh.
	/// This class will be used per-thread when integrating over mesh elements of
	/// some color.
	///
	/// N is the number of quadrature points per-axis
	/// T is the arithmetic type (e.g., float or double)
	//////////////////////////////////////////////////////////////////////
	template<int N, typename T>
	struct MeshQuadratureRule {
		

		//////////////////////////////////////////////////////////////////
		/// Aliases and mesh data
		//////////////////////////////////////////////////////////////////
		using Scalar_t   = T;
		using Mesh_t     = UnstructuredVoxelMesh<T>;
		using MeshElem_t = typename Mesh_t::Elem_t;
		
		const Mesh_t& mesh;
		MeshQuadratureRule(const Mesh_t& mesh) : 
			mesh(mesh), 
			support_element(mesh.max_depth+1), 
			projected_quad_points(mesh.max_depth+1),
			jacobian_diag(mesh.max_depth+1),
			jacobian_diag_inv(mesh.max_depth+1),
			jacobian_det(mesh.max_depth+1),
			jacobian_det_inv(mesh.max_depth+1) {
				//pre-compute jacobians for each depth
				T scale{1};
				for (uint8_t dd=0; dd<=mesh.max_depth; ++dd) {
					jacobian_diag[dd][0] = scale*mesh.diag[0];
					jacobian_diag[dd][1] = scale*mesh.diag[1];
					jacobian_diag[dd][2] = scale*mesh.diag[2];

					jacobian_diag_inv[dd][0] = T{1}/jacobian_diag[dd][0];
					jacobian_diag_inv[dd][1] = T{1}/jacobian_diag[dd][1];
					jacobian_diag_inv[dd][2] = T{1}/jacobian_diag[dd][2];

					jacobian_det[dd] 	 = jacobian_diag[dd][0] * jacobian_diag[dd][1] * jacobian_diag[dd][2];
					jacobian_det_inv[dd] = jacobian_diag_inv[dd][0] * jacobian_diag_inv[dd][1] * jacobian_diag_inv[dd][2];

					scale *= T{0.5};
				}
			}


		//////////////////////////////////////////////////////////////////
		/// Quadrature rule data
		//////////////////////////////////////////////////////////////////
		static constexpr int TOTAL_QUAD_POINTS = N*N*N;
		static constexpr int index(int i, int j, int k) noexcept { return i + N*(j + N*k); };

		//see util/quadrature_rules.hpp
		static constexpr std::array<T,N> axis_quad_points  = GV::gauss_legendre_x<N,T>();
		static constexpr std::array<T,N> axis_quad_weights = GV::gauss_legendre_w<N,T>();

		//see util/quadrature_rules.hpp
		static constexpr std::array<T,TOTAL_QUAD_POINTS> total_quad_x = GV::gauss_legendre_cartesian_coord_component<N,3,0,T>();
		static constexpr std::array<T,TOTAL_QUAD_POINTS> total_quad_y = GV::gauss_legendre_cartesian_coord_component<N,3,1,T>();;
		static constexpr std::array<T,TOTAL_QUAD_POINTS> total_quad_z = GV::gauss_legendre_cartesian_coord_component<N,3,2,T>();;
		static constexpr std::array<T,TOTAL_QUAD_POINTS> total_quad_w = GV::gauss_legendre_cartesian_weight<N,3,T>();


		//////////////////////////////////////////////////////////////////
		/// Current quadrature data
		//////////////////////////////////////////////////////////////////
		MeshElem_t 										q_el{0};
		std::vector<MeshElem_t> 						support_element{};
		std::vector<std::array<3*TOTAL_QUAD_POINTS>> 	projected_quad_points{};//all x-values, y-values, z-values per-depth
		std::vector<std::array<T,3>>					jacobian_diag{};		//jacobian info at each depth, set at contruction time
		std::vector<std::array<T,3>>					jacobian_diag_inv{};
		std::vector<T>									jacobian_det{};
		std::vector<T>									jacobian_det_inv{};

		std::span<const T,TOTAL_QUAD_POINTS> quad_x(uint8_t depth) const {
			return {&projected_quad_points[depth][0],TOTAL_QUAD_POINTS};
		}
		std::span<const T,TOTAL_QUAD_POINTS> quad_y(uint8_t depth) const {
			return {&projected_quad_points[depth][TOTAL_QUAD_POINTS],TOTAL_QUAD_POINTS};
		}
		std::span<const T,TOTAL_QUAD_POINTS> quad_z(uint8_t depth) const {
			return {&projected_quad_points[depth][2*TOTAL_QUAD_POINTS],TOTAL_QUAD_POINTS};
		}

		static constexpr std::span<const T, TOTAL_QUAD_POINTS> quad_w() {
			return {&total_quad_w[0], TOTAL_QUAD_POINTS};
		}
		
		//////////////////////////////////////////////////////////////////
		/// Collect quadrature information on a mesh element
		//////////////////////////////////////////////////////////////////
		void set_element(MeshElem_t el, uint8_t depth_range) {
			//ensure we are in cartesian form
			el   = el.decode();
			q_el = el;

			//initialzie the quadrature points at the quadrature depth
			uint8_t depth = el.depth_u8();
			support_element[depth] = el;
			std::copy(total_quad_x.begin(), total_quad_x.end(), quad_x(depth).begin());
			std::copy(total_quad_y.begin(), total_quad_y.end(), quad_y(depth).begin());
			std::copy(total_quad_z.begin(), total_quad_z.end(), quad_z(depth).begin());

			//basis elements may have support at el.depth() or coarser
			//project the quadrature reference points up to the coarser elements
			for (uint8_t dd=0; dd<depth_range && depth>0; ++dd, --depth) {
				//determine which octant to project up as
				const T dx = static_cast<T>(el.i_simd()&1) - T{0.5};
				const T dy = static_cast<T>(el.j_simd()&1) - T{0.5};
				const T dz = static_cast<T>(el.k_simd()&1) - T{0.5};

				GUTIL_SIMD()
				for (uint32_t i=0; i<TOTAL_QUAD_POINTS; ++i) {
					quad_x(depth-1)[i] = T{0.5}*quad_x(depth)[i] + dx;
					quad_y(depth-1)[i] = T{0.5}*quad_y(depth)[i] + dy;
					quad_z(depth-1)[i] = T{0.5}*quad_z(depth)[i] + dz;
				}

				el = el.parent();
				support_element[depth-1] = el;
			}

			//for some depth in [min(0,q_el.depth()-depth_range), q_el.depth()]
			//quad_x(depth) gives all TOTAL_QUAD_POINT quadrature points
			//and support_element[depth] is the support element for the basis functions
		}
	};
}