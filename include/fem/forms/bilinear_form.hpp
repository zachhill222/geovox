#pragma once

#include "fem/numerics/csr_storage.hpp"
#include "util/log_time.hpp"

#include<vector>
#include<span>

namespace GV
{
	//A generic bilinear form type for assembling FEM matrices.
	//This class will primarily be passed to a Kernel class which will coordinate everything.
	//This class will construct one or more local matricies (e.g., mass/stiffness)
	//over a single quadrature element of m trial basis functions (solution dofs) against
	//n test basis functions (also dofs, but can be of a different type than the trial).
	//the result is an n-by-m local matrix (stored as a vector).
	//This is designed for use in the CHARMS method, so m and n are unknown at compile time.
	//The H in CHARMS stands for Hierarchical. For each quadrature point in the quadrature element,
	//we must `project' it to the reference coordinate in the support element for each basis function.
	//This only needs to be done once for each unique depth of the basis functions (in the mesh/function space hierarchy)
	//To avoid doing this computation (and to avoid unnecessary looping through the mesh elements) more than necessary,
	//multiple evaluation methods can be passed to the kernel. One local matrix is produced per evaluation method.
	//The evaluation method must take the signature
	//
	//		constexpr void eval(val,Jxx,Jyy,Jzz, psi_i,spt_i,X_i,Y_i,Z_i, phi_j,spt_j,X_j,Y_j,Z_j) const;
	//
	//Where val is a reference to an (un-initialized) array<double,NQ> to store the values, J** are the diagonals of the jacobian,
	//psi_i is the test dof, phi_j is the trial dof, X_*, Y_*, Z_* are const references to an array<double,NQ>
	//that record the reference coordinate of basis psi_i or phi_j on its support element spt_*.
	//The DOF types should provide vectorized methods for eval or grad as needed with a similar signature.
	//
	//For a bilinear form integral_D a(psi, phi) dx, the corresponding entry in the global matrix is 
	//		A_IJ = integral_D a(psi_I,phi_J) = sum_E integral_E a(psi_I,phi_J)
	//where D is the problem domain, a(*,*) is the bilinear form, and E is an element of the mesh of D.
	//The local matrix for element E is then
	//      a_ij = integral_E a(psi_I, phi_J) ~ sum_q a(psi_i, phi_j, q) * w_q
	//where i and j are local dof numbers corresponding to the global numbers I and J, q is a quadrature point
	//with w(q) its corresponding quadrature weight. For hierarchical methods, it is more feasible to convert
	//the quadrature points to the correct reference points for each function psi_i and phi_j ahead of time.
	//the supplied eval(*) method described above is responsible for evaluating a(psi_i, phi_j, q) at each of the
	//supplied quadrature points. The values are then reduced with multiplication with the wieights into the local a_ij entry.
	//
	//Note that it is the responsibility of the evaluator method to correctly handle the jacobian.
	//The jacobian depends only on the mapping from the reference element to the quadrature element.
	//For an octree voxel mesh, this depends only on the depth of the quadrature element and the dimensions of the domain.




	//information required for each evaluation method.
	//bilinear forms will be constructed and passed to the kernel. the kernel will handle dispatching evaluations
	//to accumulate the local matrix for each bilinear form. The storage of the local matrix and logic for accessing values
	//is stored in the bilinear form, as this changes if the form is symmetric or not.
	//additionally, based on boundary conditions, the bilinear form may be responsible for applying boundary conditions to the
	//local matrix after it is assembled by the kernel (with 'natural' BC).
	//for better convenience when applying BC as a post processing step, the full local matrix is stored, even in the symmetric case.
	template<typename Mesh_type,
			 typename TrialDOF_type,
			 typename TestDOF_type,
			 bool IS_SYMMETRIC_=false>
	struct BilinearForm {
		using Mesh_t     = Mesh_type;
		using TestDOF_t  = TestDOF_type;
		using TrialDOF_t = TrialDOF_type;
		
		static constexpr bool IS_SYMMETRIC = IS_SYMMETRIC_;
		static_assert(!IS_SYMMETRIC || (IS_SYMMETRIC && std::same_as<TrialDOF_type, TestDOF_type>),
			"BilinearForm - The test and trial spaces/dofs must be the same for a symmetric bilinear form.");

		using QuadElem_t = typename TrialDOF_t::QuadElem_t::NonPeriodicType;
		static_assert(std::same_as<QuadElem_t, typename TestDOF_t::QuadElem_t::NonPeriodicType>,
			"BilinearForm - The test and trial spaces/dofs must have compatible quadrature elements.");

		BilinearForm(const Mesh_t& mesh) : mesh(mesh) {}

		const Mesh_t&               mesh;  		//link to mesh to project reference quadrature points to geometric points for evaluating weights
		std::vector<double>      	loc_m_v; 	//local matrix values (n_test by m_trial)
		std::span<const TestDOF_t>  test_dofs;	//local test basis functions (row dofs) (note a span is non-owning)
		std::span<const TrialDOF_t> trial_dofs; //local trial basis functions (column dofs)

		using MatStorage_t = CSR_COO<TestDOF_t,TrialDOF_t>;
		using MatRow_t = typename MatStorage_t::Row_t;
		MatStorage_t global_mat; //stores non-zero interaction between all dofs in a hybrid csr-coo format
		
		uint64_t n_test, m_trial;

		void set_basis(const std::vector<TestDOF_t>& test_dofs_, const std::vector<TrialDOF_t>& trial_dofs_) requires (!IS_SYMMETRIC) {
			trial_dofs = trial_dofs_;
			test_dofs  = test_dofs_;
			n_test  = test_dofs.size();
			m_trial = trial_dofs.size();
			loc_m_v.resize(n_test*m_trial, 0.0);
		}

		void set_basis(const std::vector<TestDOF_t>& test_dofs_, const std::vector<TrialDOF_t>& trial_dofs_) requires (IS_SYMMETRIC) {
			trial_dofs = trial_dofs_;
			test_dofs = trial_dofs_;
			m_trial = trial_dofs.size();
			n_test  = m_trial;
			loc_m_v.resize(n_test*m_trial, 0.0);
		}

		inline double& local_mat(uint64_t i, uint64_t j) {
			assert(i<n_test);
			assert(j<m_trial);
			return loc_m_v[j + i*m_trial]; //row-major is better for BC setting
		}

		inline double local_mat(uint64_t i, uint64_t j) const {
			assert(i<n_test);
			assert(j<m_trial);
			return loc_m_v[j + i*m_trial]; //row-major is better for BC setting
		}

		void scatter() {
			//add the results of the local matrix to the global matrix
			//preserves sorted and accumulated
			for (uint64_t i=0; i<n_test; ++i) {
				MatRow_t& row = global_mat.get_row(test_dofs[i]);
				row.reserve(row.size()+m_trial);
				for (uint64_t j=0; j<m_trial; ++j) {
					row.emplace_back(trial_dofs[j], local_mat(i,j));
				}
				row.accumulate();
			}
		}

		inline auto to_eigen_csr(const std::vector<TestDOF_t>& test_dofs_, const std::vector<TrialDOF_t>& trial_dofs_) const {
			return global_mat.to_eigen_csr(test_dofs_, trial_dofs_);
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
