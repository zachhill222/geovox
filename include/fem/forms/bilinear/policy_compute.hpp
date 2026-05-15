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
	//a class responsible for storing and computing local stiffness matrices
	//TODO: can we make anything more efficient with a prepare(..) method?
	template<typename TestDOF_t, typename TrialDOF_t, typename EvalPolicy>
	struct BilinearFormComputeLocalMat
	{
		static constexpr bool IS_SYMMETRIC = EvalPolicy::IS_SYMMETRIC;

		//handle local matrix
		std::vector<double> loc_mat;
		uint64_t n_test=0, m_trial=0;
		inline double& value(uint64_t i, uint64_t j) {
			assert(i<n_test);
			assert(j<m_trial);
			return loc_mat[j + i*m_trial]; //row-major is better for BC setting
		}

		inline double value(uint64_t i, uint64_t j) const {
			assert(i<n_test);
			assert(j<m_trial);
			return loc_mat[j + i*m_trial]; //row-major is better for BC setting
		}

		inline std::span<const double> data() const {return {loc_mat};}
		std::vector<double> diag() const {
			assert(n_test==m_trial);
			std::vector<double> result(n_test);
			#pragma omp simd
			for (uint64_t i=0; i<n_test; ++i) {
				result[i] = value(i,i);
			}
			return result;
		}

		EvalPolicy eval{};
		void set_eval_policy(EvalPolicy ep) {eval = ep;}

		//compute local matrix
		template<VoxelElementKeyType QuadElem_t, uint64_t N_QUAD_POINTS>
		void compute(	std::span<const TestDOF_t> test, 
						std::span<const TrialDOF_t> trial, 
						const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map) {
			n_test  = test.size();
			m_trial = trial.size();
			loc_mat.assign(n_test*m_trial, 0.0); //the initialize is safe but unnecessary

			constexpr uint64_t NQ = QuadPointMap<QuadElem_t,N_QUAD_POINTS>::NQ;
			std::span<const double> weights = q_map.get_quad_weights();
			std::array<double,NQ> vals; //value of the integrand at each quadrature point

			//get the jacobian from the quadruatre class
			const double Jxx=q_map.Jac[0];
			const double Jyy=q_map.Jac[1];
			const double Jzz=q_map.Jac[2];

			for (uint64_t j=0; j<m_trial; ++j) {
				const TrialDOF_t phi_j = trial[j];
				const uint64_t depth_j = phi_j.depth();
				const QuadElem_t spt_j = q_map.get_support_element(depth_j);
				std::span<const double,NQ> X_j = q_map.get_quad_points_x(depth_j);
				std::span<const double,NQ> Y_j = q_map.get_quad_points_y(depth_j);
				std::span<const double,NQ> Z_j = q_map.get_quad_points_z(depth_j);

				for (uint64_t i=0; i<n_test; ++i) {
					if constexpr (IS_SYMMETRIC) {if (i<j) {continue;}}

					const TestDOF_t psi_i  = test[i];
					const uint64_t depth_i = psi_i.depth();
					const QuadElem_t spt_i = q_map.get_support_element(depth_i);
					std::span<const double,NQ> X_i = q_map.get_quad_points_x(depth_i);
					std::span<const double,NQ> Y_i = q_map.get_quad_points_y(depth_i);
					std::span<const double,NQ> Z_i = q_map.get_quad_points_z(depth_i);

					eval.template evaluate<NQ>(vals, Jxx, Jyy, Jzz,
						psi_i, spt_i, X_i, Y_i, Z_i,
						phi_j, spt_j, X_j, Y_j, Z_j);

					double v_ij = 0.0;
					#pragma omp simd reduction(+:v_ij)
					for (uint64_t l=0; l<NQ; ++l) {
						v_ij += vals[l] * weights[l];
					}

					value(i,j) = v_ij;
					if constexpr (IS_SYMMETRIC) {value(j,i) = v_ij;}
				}
			}
		}
	};


	//a class responsible for storing and computing only the diagonal of the local stiffness matrix
	template<typename TestDOF_t, typename TrialDOF_t, typename EvalPolicy>
	struct BilinearFormComputeLocalMatDiag
	{
		//handle local vector
		std::vector<double> loc_diag;
		inline double  local(const uint64_t i) const {assert(i<loc_diag.size()); return loc_diag[i];}
		inline double& local(const uint64_t i) {assert(i<loc_diag.size()); return loc_diag[i];}

		//set evaluation method/policy
		EvalPolicy eval{};
		void set_eval_policy(EvalPolicy ep) {eval = ep;}

		//compute local vector
		template<VoxelElementKeyType QuadElem_t, uint64_t N_QUAD_POINTS>
		void compute(	std::span<const TestDOF_t> test, 
						std::span<const TrialDOF_t> trial, 
						const QuadPointMap<QuadElem_t,N_QUAD_POINTS>& q_map) {
			assert(test.size()==trial.size());
			loc_diag.assign(test.size(), 0.0); //the initialize is safe but unnecessary

			constexpr uint64_t NQ = QuadPointMap<QuadElem_t,N_QUAD_POINTS>::NQ;
			std::span<const double> weights = q_map.get_quad_weights();
			std::array<double,NQ> vals; //value of the integrand at each quadrature point

			//get the jacobian from the quadruatre class
			const double Jxx=q_map.Jac[0];
			const double Jyy=q_map.Jac[1];
			const double Jzz=q_map.Jac[2];

			for (uint64_t j=0; j<trial.size(); ++j) {
				const TrialDOF_t phi_j = trial[j];
				const uint64_t depth_j = phi_j.depth();
				const QuadElem_t spt_j = q_map.get_support_element(depth_j);
				std::span<const double,NQ> X_j = q_map.get_quad_points_x(depth_j);
				std::span<const double,NQ> Y_j = q_map.get_quad_points_y(depth_j);
				std::span<const double,NQ> Z_j = q_map.get_quad_points_z(depth_j);

				//use "subscript" i for convenience. The test and trial spaces may use different dof handlers, so the supports may be different.
				const TestDOF_t psi_i  = test[j];
				const uint64_t depth_i = psi_i.depth();
				const QuadElem_t spt_i = q_map.get_support_element(depth_i);
				std::span<const double,NQ> X_i = q_map.get_quad_points_x(depth_i);
				std::span<const double,NQ> Y_i = q_map.get_quad_points_y(depth_i);
				std::span<const double,NQ> Z_i = q_map.get_quad_points_z(depth_i);

				eval.template evaluate<NQ>(vals, Jxx, Jyy, Jzz,
					psi_i, spt_i, X_i, Y_i, Z_i,
					phi_j, spt_j, X_j, Y_j, Z_j);

				double diag = 0.0;
				#pragma omp simd reduction(+:diag)
				for (uint64_t l=0; l<NQ; ++l) {
					diag += vals[l] * weights[l];
				}

				loc_diag[j] = diag;
			}
		}
	};
}