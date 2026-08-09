#pragma once

#include "mesh/keys/voxel_key.hpp"
#include "fem/numerics/quad_point_map.hpp"
#include "util/compatibility.hpp"

#include <vector>
#include <span>
#include <cstdint>
#include <cassert>

namespace GV
{
	//a class responsible for storing and computing local values of the dofs
	//TODO: can we make anything more efficient with a prepare(..) method?
	template<typename TestDOF_t, typename EvalPolicy>
	struct LinearFormComputeLocalVec
	{
		//handle local matrix
		std::vector<double> loc_vec;
		uint64_t n_test=0;
		inline double& value(uint64_t i) {
			assert(i<n_test);
			return loc_vec[i];
		}

		inline double value(uint64_t i) const {
			assert(i<n_test);
			return loc_vec[i];
		}

		inline std::span<const double> data() const {return {loc_vec};}

		EvalPolicy eval{};
		void set_eval_policy(EvalPolicy ep) {eval = ep;}
		LinearFormComputeLocalVec(EvalPolicy eval) : eval(eval) {}

		//compute local matrix
		template<VoxelElementKeyType QuadElem_t, uint64_t N_QUAD_POINTS>
		void compute(	std::span<const TestDOF_t> test,
						const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map) {
			n_test  = test.size();
			loc_vec.assign(n_test, 0.0); //the initialize is safe but unnecessary

			constexpr uint64_t NQ = QuadPointMap<QuadElem_t,N_QUAD_POINTS>::NQ;
			std::span<const double> weights = q_map.get_quad_weights();
			std::array<double,NQ> vals; //value of the integrand at each quadrature point

			//get the jacobian from the quadruatre class
			const double Jxx=q_map.Jac[0];
			const double Jyy=q_map.Jac[1];
			const double Jzz=q_map.Jac[2];

			
			for (uint64_t i=0; i<n_test; ++i) {	
				const TestDOF_t psi_i  = test[i];
				const uint64_t depth_i = psi_i.depth();
				const QuadElem_t spt_i = q_map.get_support_element(depth_i);
				std::span<const double,NQ> X_i = q_map.get_quad_points_x(depth_i);
				std::span<const double,NQ> Y_i = q_map.get_quad_points_y(depth_i);
				std::span<const double,NQ> Z_i = q_map.get_quad_points_z(depth_i);

				if constexpr (EvalPolicy::NEEDS_GEOMETRY_POINTS) {
					eval.x.assign(X_i.begin(), X_i.end());
					eval.y.assign(Y_i.begin(), Y_i.end());
					eval.z.assign(Z_i.begin(), Z_i.end());
					q_map.ref2geo(eval.x, eval.y, eval.z, spt_i);
				}

				eval.template evaluate<NQ>(vals, Jxx, Jyy, Jzz,
					psi_i, spt_i, X_i, Y_i, Z_i);

				double v_i = 0.0;
				#pragma omp simd reduction(+:v_i)
				for (uint64_t l=0; l<NQ; ++l) {
					v_i += vals[l] * weights[l];
				}

				value(i) = v_i;
			}
		}
	};
}