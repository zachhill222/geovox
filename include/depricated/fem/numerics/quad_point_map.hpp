#pragma once

#include "gutil.hpp"

#include "mesh/keys/voxel_key.hpp"
#include "util/quadrature_rules.hpp"

#include <span>
#include <cassert>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace GV
{
	//A class to handle mapping of the quadrature points to each element
	template<VoxelElementKeyType QuadElem_t, uint64_t N_QUAD_POINTS=4>
	struct QuadPointMap
	{
		//standard axis values
		static constexpr uint64_t NQ_A = N_QUAD_POINTS;  //number of quadrature points along each axis
		static constexpr uint64_t NQ   = NQ_A*NQ_A*NQ_A; //total number of quadrature points
		static constexpr uint64_t MAX_DEPTH = QuadElem_t::MAX_DEPTH; //maximum depth that we may need to project points to

		static constexpr std::array<double, NQ_A> gq_x = gauss_legendre_x<NQ_A>(); //x,y,z coordinates on the standard [-1,1] interval
		
		static constexpr std::array<double, NQ> p_qw = gauss_legendre_cartesian_weight<NQ_A, 3, double>(); //quadrature weight at each point (on [-1,1]^3 refrence element)
		//during a kernel call, we will need to convert the quadrature points from the quadrature element
		//to the reference coordinates in the support element for each basis function. when doing this
		//we can recover the support element for the basis function. this process only depends on the 
		//depth of the basis function support elments and not on the basis function itself. we can "sweep"
		//the computation from the depth of the quadrature element up to depth 0 one time rather than for each element.
		//note that the quadrature points are a cartesian grid on each support element, so we only need to compute the unique x,y,z values once
		std::array<std::array<double, NQ_A>, MAX_DEPTH> p_qx, p_qy, p_qz; //projected quadrature points (unique only)
		std::array<std::array<double, NQ>, MAX_DEPTH> p_qxa, p_qya, p_qza; //projected quadrature points (all assembled, use these to pass to eval)
		std::array<QuadElem_t, MAX_DEPTH> s_el; //support element for each depth

		constexpr void project_to_support(const QuadElem_t q_elem) {
			const uint64_t md = q_elem.depth(); //max starting depth, work to depth 0
			//the unique x,y,z coordinates are all the standard gauss-legendre points at the starting depth
			p_qx[md] = gq_x;
			p_qy[md] = gq_x;
			p_qz[md] = gq_x;
			s_el[md] = q_elem;

			for (uint64_t dd = md; dd>0; --dd) {
				//determine if the element is an even/odd child in each coordinate
				//and compute the increment to shift each coordinate
				const double dx = static_cast<double>(s_el[dd].i() & 1) - 0.5;
				const double dy = static_cast<double>(s_el[dd].j() & 1) - 0.5;
				const double dz = static_cast<double>(s_el[dd].k() & 1) - 0.5;

				#pragma omp simd
				for (uint64_t q=0; q<NQ_A; ++q) {
					p_qx[dd-1][q] = 0.5*p_qx[dd][q] + dx;
					p_qy[dd-1][q] = 0.5*p_qy[dd][q] + dy;
					p_qz[dd-1][q] = 0.5*p_qz[dd][q] + dz;
				}

				s_el[dd-1] = s_el[dd].parent();
			}
		}

		//before passing to the dofs, we need to assemble the x/y/z coordinates together
		//the quadrature points are always a NQ_A x NQ_A x NQ_A cartesian grid with the grid locations
		//at each depth stored in p_q* for each x,y,z axis. This routine takes their axial values
		//and produces all NQ_A^3 points, split into x,y,z components.
		//the linear index of the (i,j,k) point is
		//
		//		l = i + NQ_A*(j + NQ_A*k) = i + j*NQ_A + k*NQ_A^2
		//
		constexpr void collect_quad_points(const uint64_t md)
		{
			for (uint64_t dd=0; dd<=md; ++dd) {
				#pragma omp simd collapse(3)
				for (uint64_t k=0; k<NQ_A; ++k) {
					for (uint64_t j=0; j<NQ_A; ++j) {
						for (uint64_t i=0; i<NQ_A; ++i) {
							const uint64_t l = i + NQ_A*(j + NQ_A*k);
							p_qxa[dd][l] = p_qx[dd][i];
							p_qya[dd][l] = p_qy[dd][j];
							p_qza[dd][l] = p_qz[dd][k];
						}
					}
				}
			}
		}

		//set the quadrature element. this handles the above logic.
		//the support elements and quadrature points at depths elem.depth() and above (coarser, smaller depth)
		//will be set and can be used after this.
		QuadElem_t last_elem;
		constexpr void set_quad_element(const QuadElem_t elem) {
			assert(elem.is_valid());
			last_elem = elem;
			project_to_support(elem);
			collect_quad_points(elem.depth());
			set_jacobian();
		}

		constexpr QuadElem_t get_support_element(const uint64_t depth) const {
			assert(depth<=last_elem.depth());
			return s_el[depth];
		}

		constexpr std::span<const double, NQ> get_quad_points_x(const uint64_t depth) const {
			assert(depth<=last_elem.depth());
			return {p_qxa[depth]};
		}

		constexpr std::span<const double, NQ> get_quad_points_y(const uint64_t depth) const {
			assert(depth<=last_elem.depth());
			return {p_qya[depth]};
		}

		constexpr std::span<const double, NQ> get_quad_points_z(const uint64_t depth) const {
			assert(depth<=last_elem.depth());
			return {p_qza[depth]};
		}

		static constexpr std::span<const double, NQ> get_quad_weights() {
			return {p_qw};
		}

		//store mesh corners to allow convering from reference to geometric points if necessary
		//similarly, comput the jacobian for the quadrature element
		template<typename Coord_t>
		void set_bounds(const Coord_t& low, const Coord_t& high) {
			for (int i=0; i<3; ++i) {
				mesh_low[i]  = low[i];
				mesh_high[i] = high[i];
				mesh_diag[i] = high[i] - low[i];
			}
		}

		void set_jacobian() {
			const double scale = std::ldexp(0.5, -static_cast<int>(last_elem.depth()));
			Jac[0] = mesh_diag[0] * scale;
			Jac[1] = mesh_diag[1] * scale;
			Jac[2] = mesh_diag[2] * scale;
		}

		gutil::Point<3,double> mesh_low  {-1,-1,-1};
		gutil::Point<3,double> mesh_high { 1, 1, 1};
		gutil::Point<3,double> mesh_diag { 2, 2, 2};
		gutil::Point<3,double> Jac       { 1, 1, 1};
		
		constexpr void ref2geo(std::span<double> x, std::span<double> y, std::span<double> z, QuadElem_t el) const {
			//vertex is in the normalized [0,1] interval
			const gutil::Point<3,double> mid  = mesh_low + 0.5*mesh_diag*(el.vertex(7).normalized_coordinate()+el.vertex(0).normalized_coordinate());
			const gutil::Point<3,double> del  = 0.5*mesh_diag*(el.vertex(7).normalized_coordinate()-el.vertex(0).normalized_coordinate());

			#pragma omp simd
			for (uint64_t i=0; i<x.size(); ++i) {x[i] = mid[0] + x[i]*del[0];}

			#pragma omp simd
			for (uint64_t i=0; i<y.size(); ++i) {y[i] = mid[1] + y[i]*del[1];}

			#pragma omp simd
			for (uint64_t i=0; i<z.size(); ++i) {z[i] = mid[2] + z[i]*del[2];}
		}
	};
}