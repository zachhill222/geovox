#include "fem/problems/base_problem.hpp" //includes dofs, mesh, and other standard things. defines convenient aliases

using namespace GV;

using Mesh_t = VoxelMesh<8,false>;

struct CahnHilliard : BaseProblem<Mesh_t>
{
	using BASE 		= BaseProblem<Mesh_t>;
	using DofKey_t  = typename Mesh_t::VoxelVertex::OtherPeriodicType<7>; //periodic BC
	using DOF_t     = VoxelQ1<DofKey_t>;
	using Elem_t    = typename Mesh_t::VoxelElement;
	using Vert_t    = typename Mesh_t::VoxelVertex;

	using H1      = SymmetricH1<DOF_t>; //unweighted bilinear form grad u/w * grad phi for both u and w
	using L2      = SymmetricL2<DOF_t>; //unweighted bilinear form u/w * phi for both u and w
	
	using Vec_t		= typename BASE::Vec_t;		//Eigen::VectorXd for solutions
	using SpMat_t	= typename BASE::SpMat_t;	//Eigen::SparseMat in RowMajor (CSR)

	Mesh_t 						mesh;		//primary mesh

	DofHandler<Mesh_t,DOF_t> 	dofs;		//dof handler for both u and w

	H1							stif_form;	//bilinear form for grad w * grad phi
	L2 							mass_form;	//bilinear form for w * phi

	Vec_t 						u, w;		//current solutions
	SpMat_t 					A, M;		//current mass/stiffness matrices (same for u and w)

	//problem parameters
	double alpha=0.01, k=0.1, dt=0.001;

	CahnHilliard() :
		mesh{{0,0,0},{1,1,1}},
		dofs{mesh},
		stif_form{mesh},
		mass_form{mesh},
		{}


	//refine dofs associated with an element
	void refine_element(const Elem_t el) {
		const auto basis = dofs.basis_s(el);
		dofs.refine(basis);

		//call mesh.process_request_active() to refine the mesh after
		//call transfer solution to update u and w to the new dofs
	}

	//update solutions to a refined/coarsened mesh
	void transfer_solution() {
		const auto n_dofs = dofs.n_dofs();
		Vec_t u_new(n_dofs), w_new(n_dofs);

		dofs.update_coefs(u_new, u);
		dofs.update_coefs(w_new, w);
	}

	template<typename Fun>
	void set_ic(Fun&& fun, uint64_t depth) {
		mesh.set_depth(depth);
		dofs.set_depth(depth);

		auto eval = [this](DOF_t dof) {
			return fun(mesh.ref2geo(static_cast<Vert_t>(dof.key)));
		}

		dofs.init_coefs_by_dof(u,std::forward(eval));
	}

	

};