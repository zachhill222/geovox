#pragma once

#include "fem/numerics/csr_storage.hpp"
#include "fem/forms/form_actions.hpp"

#include "util/log_time.hpp"
#include "util/compatibility.hpp"

#include <vector>
#include <span>
#include <unordered_map>

namespace GV
{
	//A generic linear form type for assembling FEM right hand sides.
	//This class will primarily be passed to a Kernel class which will coordinate everything.
	//This class will construct a single vector for the rhs in the global dofs. It can be sampled to get
	//the rhs for currently active dofs in an arbitrary order.
	//The evaluation method must take the signature
	//
	//		constexpr void eval(val,Jxx,Jyy,Jzz, psi_,spt,X,Y,Z) const;
	//
	//Where val is a reference to an (un-initialized) array<double,NQ> to store the values at each quadratrure point,
	//Jxx,Jyy,Jzz are the diagonal entries of the jacobian, psi is the test dof, X, Y, Z are const references to an array<double,NQ>
	//of quadrature points in the support element spt.
	//The DOF types should provide vectorized methods for eval or grad as needed with a similar signature.
	//
	//For a linear form integral_D b(psi) dx, the corresponding entry in the global matrix is 
	//		B_I = integral_D b(psi) = sum_E integral_E b(psi)
	//where D is the problem domain, b(*) is the linear form, and E is an element of the mesh of D.
	//The contrubution for element E is then
	//      b_i = integral_E b(psi_I) ~ sum_q b(psi_i)(q) * w_q
	//where i is the local dof number corresponding to the global numbers I, q is a quadrature point
	//with w(q) its corresponding quadrature weight.


	//information required for each evaluation method.
	//bilinear forms will be constructed and passed to the kernel. the kernel will handle dispatching evaluations
	//to accumulate the local matrix for each bilinear form. The storage of the local matrix and logic for accessing values
	//is stored in the bilinear form, as this changes if the form is symmetric or not.
	//additionally, based on boundary conditions, the bilinear form may be responsible for applying boundary condions to the
	//local matrix after it is assembled by the kernel (with 'natural' BC).
	//for better convenience when applying BC as a post processing step, the full local matrix is stored, even in the symmetric case.
	template<typename TestHandler_type, typename Action_type=ScatterAction>	 
	struct LinearForm {
		using TestDOF_t    = typename TestHandler_type::DOF_t;
		using Mesh_t       = typename TestHandler_type::Mesh_t;
		using QuadElem_t   = typename TestDOF_t::QuadElem_t::NonPeriodicVariant;
		using VecStorage_t = std::unordered_map<TestDOF_t, double, typename TestDOF_t::Hash>;
		
		explicit LinearForm(const TestHandler_type& handler) : handler(handler) {}

		const TestHandler_type& 	handler;    //link to the dof handler
		std::vector<double>      	loc_b_v; 	//local vector values (n_test)
		std::span<const TestDOF_t>  test_dofs;	//local test basis functions (row dofs) (note a span is non-owning)

		
		//store contributions of global dofs
		VecStorage_t* global_vec = nullptr;
		inline void set_storage(VecStorage_t& vec) requires ScatterActionType<Action_type> {global_vec = &vec;}
		
		//compute the action and store the result in a vector
		std::span<double> vec;
		template<typename Container_t>
		inline void set_vec(Container_t& v) requires DotActionType<Action_type> {vec = as_span(v);}

		uint64_t n_test;
		template<typename Container_t> requires std::same_as<typename Container_t::value_type, TestDOF_t>
		void set_basis(const Container_t& dofs) {
			test_dofs = as_span(dofs);
			n_test    = dofs.size();
			loc_b_v.assign(n_test, 0.0);
		}

		void scatter() requires ScatterActionType<Action_type> {
			if (global_vec==nullptr) {throw std::runtime_error("LinearForm::scatter - called with no global vector (nullptr)");}
			for (size_t i=0; i<loc_b_v.size(); ++i) {
				(*global_vec)[test_dofs[i]] += loc_b_v[i];
			}
		}

		void scatter() requires DotActionType<Action_type> {
			assert(!vec.empty());

			for (size_t i=0; i<n_test; ++i) {
				const auto N = handler.compressed_index(test_dofs[i]);
				assert(N>=0);
				vec[N] += loc_b_v[i];
			}
		}

		inline double loc_val(const uint64_t i) const {assert(i<loc_b_v.size()); return loc_b_v[i];}
		inline double& loc_val(const uint64_t i) {assert(i<loc_b_v.size()); return loc_b_v[i];}

		Eigen::VectorXd to_eigen_Xd(const std::vector<TestDOF_t>& dofs) const requires ScatterActionType<Action_type> {
			Eigen::VectorXd result(dofs.size());
			for (size_t i=0; i<dofs.size(); ++i) {
				const TestDOF_t dof = dofs[i];
				auto it = global_vec->find(dof);
				result[i] = (it!=global_vec->end()) ? it->second : 0.0;
			}
			return result;
		}

		//for weighted forms, it is convenient to evaluate in the mesh coordinates
		template<uint64_t N>
		void ref2geo(
			std::array<double,N>& x,
			std::array<double,N>& y,
			std::array<double,N>& z,
			const QuadElem_t spt,
			const std::array<double,N> X,
			const std::array<double,N> Y,
			const std::array<double,N> Z) const 
		{
			const Mesh_t& mesh = handler.mesh;

			const auto el   = static_cast<typename Mesh_t::VoxelElement>(spt);
			const auto low  = mesh.ref2geo(el.vertex(0));
			const auto high = mesh.ref2geo(el.vertex(7));
			const auto mid  = 0.5*(low+high);
			const auto del  = 0.5*(high-low);

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {x[i] = mid[0] + X[i]*del[0];}

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {y[i] = mid[1] + Y[i]*del[1];}

			#pragma omp simd
			for (uint64_t i=0; i<N; ++i) {z[i] = mid[2] + Z[i]*del[2];}
		}
	};
}
