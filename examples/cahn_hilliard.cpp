#include "fem/problems/problem_base.hpp" //includes dofs, mesh, and other standard things. defines convenient aliases
#include "util/log_time.hpp" //time certain routines

#include <cmath>

#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>

using namespace GV;

using Mesh_t = VoxelMesh<8,false>;

struct CahnHilliard : BaseProblem<Mesh_t>
{
	using BASE 		= BaseProblem<Mesh_t>;
	using DofKey_t  = typename Mesh_t::VoxelVertex::OtherPeriodicType<7>; //periodic BC
	using DOF_t     = VoxelQ1<DofKey_t>;
	using Elem_t    = typename Mesh_t::VoxelElement;
	using Vert_t    = typename Mesh_t::VoxelVertex;

	using H1      	= SymmetricH1<Mesh_t,DOF_t>; //unweighted bilinear form grad u/w * grad phi for both u and w
	using L2      	= SymmetricL2<Mesh_t,DOF_t>; //unweighted bilinear form u/w * phi for both u and w
	using Kernel_t	= Kernel<4,TypeList<H1,L2>,TypeList<>>; //kernel for integrating mass and stiffness matrices

	using Vec_t		= typename BASE::Vec_t;		//Eigen::VectorXd for solutions
	using SpMat_t	= typename BASE::SpMat_t;	//Eigen::SparseMat in RowMajor (CSR)

	Mesh_t 						mesh;		//primary mesh

	DofHandler<Mesh_t,DOF_t> 	dofhandler;	//dof handler for both u and w

	H1							stif_form;	//bilinear form for grad w * grad phi
	L2 							mass_form;	//bilinear form for w * phi

	Vec_t 						u, w;		//current solutions
	SpMat_t 					LHS,A,M;	//current mass/stiffness and left hand side matrices (A,M are the same for u and w)

	using Precon_t = Eigen::DiagonalPreconditioner<double>;
	Eigen::GMRES<SpMat_t, Precon_t> solver;



	//problem parameters
	double alpha=0.01, k=0.1, dt=0.01;
	int t_step = 0;

	CahnHilliard() :
		mesh{{0,0,0},{1,1,1}},
		dofhandler{mesh},
		stif_form{mesh},
		mass_form{mesh}
		{
			#ifdef _OPENMP
			omp_set_max_active_levels(2);
			omp_set_nested(1);
			#endif

			Logger::log("CahnHilliard initialized");
		}


	//refine dofs associated with an element
	void refine_element(const Elem_t el) {
		const auto basis = dofhandler.basis_s(el);
		dofhandler.refine(basis);

		//call mesh.process_request_active() to refine the mesh after
		//call transfer solution to update u and w to the new dofs
	}

	//update solutions to a refined/coarsened mesh
	void transfer_solution() {
		LogTime time("CahnHilliard::transfer_solution");

		const auto n_dofs = dofhandler.n_dofs();
		Vec_t u_new(n_dofs), w_new(n_dofs);

		dofhandler.update_coefs(u, u_new);
		dofhandler.update_coefs(w, w_new);

		u = std::move(u_new);
		w = std::move(w_new);
	}

	template<typename Fun>
	void set_ic(Fun&& fun, uint64_t depth) {
		LogTime time("CahnHilliard::set_ic");

		mesh.set_depth(depth);
		dofhandler.set_depth(depth);
		dofhandler.save_dof_list();
		dofhandler.compress_dof_numbers();

		auto eval = [this,&fun](DOF_t dof) {
			return fun(mesh.ref2geo(static_cast<Vert_t>(dof.key)));
		};

		u.resize(dofhandler.n_dofs());
		dofhandler.init_coefs_by_dof(u, eval);
		w = Vec_t::Zero(u.size());
	}

	//compute the mass and stiffness matricies from scratch
	//over the current active basis functions
	void integrate_all() {
		LogTime time("CahnHilliard::integrate_all");

		//clear old data
		stif_form.global_mat.clear();
		mass_form.global_mat.clear();

		//initialize kernel for numerical integration. TODO: change to per-thread and loop by element color
		const auto diag = mesh.high - mesh.low;
		Kernel_t kernel(diag[0], diag[1], diag[2], stif_form, mass_form);
		
		//define operations to do on each active element
		auto action = [this, &kernel](Elem_t el) {
			const auto el_basis = dofhandler.basis_active(el);
			kernel.set_element(el);
			kernel.template B_set_basis<0>(el_basis,el_basis);
			kernel.template B_set_basis<1>(el_basis,el_basis);

			kernel.template B_compute_scatter<0>();
			kernel.template B_compute_scatter<1>();
		};

		//define which elements to integrate over
		auto pred = [this](Elem_t el) {return mesh.is_active(el);};

		//loop over the elements. TODO: log the maximum depth so that we don't have to loop over elements where we know the predicate is false
		mesh.template for_each<Elem_t>(action, false, pred);
	}


	//assemble LHS matrix
	void assemble_mats() {
		LogTime timer{"CahnHilliard::assemble_mats"};
		const auto& dofs = dofhandler.last_compressed_dofs();

		#ifdef _OPENMP
		#pragma omp parallel
		#pragma omp single
		#endif
		{
			#ifdef _OPENMP
			#pragma omp task
			#endif
			{
				A = stif_form.to_eigen_csr(dofs,dofs);
			}

			#ifdef _OPENMP
			#pragma omp task
			#endif
			{
				M = mass_form.to_eigen_csr(dofs,dofs);
			}
		}

		//assemble block matrix left hand side
		//
		//	LHS = [M       k*dt*A]
		//		  [M-a*A   M     ]
		//
		//note A and M have the same sparsity structure
		const auto n = A.rows();
		const auto nnz = A.nonZeros();

		LHS = SpMat_t{2*n,2*n};
		LHS.resizeNonZeros(4*nnz);
		int* RO = LHS.outerIndexPtr();
		int* CI = LHS.innerIndexPtr();
		double* V = LHS.valuePtr();

		//build row offsets
		RO[0] = 0;		//start of top block
		RO[n] = 2*nnz;	//start of bottom block
		for (int r=0; r<n; ++r) {
			//get number of nonzeros in each horizontal block
			const int row_nnz = A.outerIndexPtr()[r+1] - A.outerIndexPtr()[r];
			RO[r+1]   = RO[r]   + 2*row_nnz; //top block
			RO[n+r+1] = RO[n+r] + 2*row_nnz; //bottom block
		}
		assert(RO[n]   == 2*nnz);
		assert(RO[2*n] == 4*nnz);

		//fill column indices and values
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for (int r=0; r<n; ++r) {
			//start/end of outer indices for the blocks
			const int b_start = A.outerIndexPtr()[r];
			const int b_end   = A.outerIndexPtr()[r+1];
			
			//top blocks
			int j = RO[r];
			for (int idx=b_start; idx<b_end; ++idx) {
				CI[j] = M.innerIndexPtr()[idx];
				V[j]  = M.valuePtr()[idx];
				++j;
			}
			for (int idx=b_start; idx<b_end; ++idx) {
				CI[j] = A.innerIndexPtr()[idx] + n; //column indices increased by n for the top right block
				V[j]  = k*dt*A.valuePtr()[idx];
				++j;
			}

			//bottom blocks
			j = RO[r+n];
			for (int idx=b_start; idx<b_end; ++idx) {
				CI[j] = M.innerIndexPtr()[idx];
				V[j]  = M.valuePtr()[idx] - alpha*A.valuePtr()[idx];
				++j;
			}
			for (int idx=b_start; idx<b_end; ++idx) {
				CI[j] = M.innerIndexPtr()[idx] + n; //column indices increased by n for the top right block
				V[j]  = M.valuePtr()[idx];
				++j;
			}
		}

		//initialize solver to the new LHS
		init_solver();
	}

	void init_solver() {
		LogTime time("CahnHilliard::init_solver");
		solver.setMaxIterations(1000);
		solver.setTolerance(1e-10);
		solver.compute(LHS);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("CahnHilliard::init_solver - solver could not be initialized");
		}
	}

	void step_forward() {
		LogTime time("CahnHilliard::step_forward " + std::to_string(t_step));

		//print sizes of matrices and vectors
		// std::cout << "A: " << A.rows() << " x " << A.cols() << " nnz=" << A.nonZeros() << "\n";
		// std::cout << "M: " << M.rows() << " x " << M.cols() << " nnz=" << M.nonZeros() << "\n";
		// std::cout << "LHS: " << LHS.rows() << " x " << LHS.cols() << " nnz=" << LHS.nonZeros() << "\n";
		// std::cout << "u: " << u.size() << "\n";
		// std::cout << "w: " << w.size() << "\n";

		//assemble rhs
		const int n = A.rows();
		Vec_t rhs(2*n);

		Vec_t u3 = u.array().cube().matrix();
		rhs.head(n) = M*u;
		rhs.tail(n) = M*u3;

		//solve system
		Vec_t x(2*n);
		x.head(n) = u;
		x.tail(n) = w;
		x = solver.solveWithGuess(rhs, x);

		//check solution. TODO: wrap fail to converge into a refinement step?
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("CahnHilliard::step_forward - solver failed to converge");
		}

		//store solution
		Vec_t u0 = x.head(n);
		Vec_t w0 = x.tail(n);

		//TODO: check if we need to refine
		u = std::move(u0);
		w = std::move(w0);
		++t_step;
	}


	void save_solution(const std::string simulation_name) const {
		LogTime timer{"CahnHilliard::save_solution"};

		const std::string filename = simulation_name + "_" + std::to_string(t_step) + ".vtk";
		std::ofstream file(filename);
		if (!file.is_open()) {
			throw std::runtime_error("CahnHilliard::save_solution - could not open file: " + filename);
		}

		//write the mesh
		const auto n_verts = mesh.write_unstructured_vtk(file);

		//interpolate the solution to the vertex values
		auto vert_vals = dofhandler.interpolate_to_vertices(u, n_verts);

		//append solution and header
		file << "POINT_DATA " << n_verts << "\n";
		mesh.append_unstructured_point_data_vtk(
			file,
			"SCALARS u float 1\nLOOKUP_TABLE default",
			n_verts,
			[&vert_vals](Vert_t vtx) {return vert_vals[vtx.linear_index()];}
		);

		vert_vals = dofhandler.interpolate_to_vertices(w, n_verts);
		mesh.append_unstructured_point_data_vtk(
			file,
			"SCALARS w float 1\nLOOKUP_TABLE default",
			n_verts,
			[&vert_vals](Vert_t vtx) {return vert_vals[vtx.linear_index()];}
		);
	}
};


int main(int argc, char* argv[]) {
	LogTime time{"Program Time"};

	CahnHilliard problem{};

	auto ic_fun = [](Point<3,double> X) {
		X[0]-=0.5;
		X[1]-=0.5;
		X[2]-=0.5;
		const double r = std::sqrt(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);

		double val = 0.5*std::cos(6.3*r);
		val += 0.05*std::sin(10*r);
		return val;
	};
	problem.set_ic(ic_fun,5);

	problem.integrate_all();
	problem.assemble_mats();

	for (int i=0; i<60; ++i) {
		if (i%10 == 0) {
			problem.save_solution("cahn_hilliard");
		}
		problem.step_forward();
	}
}