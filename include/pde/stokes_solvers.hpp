#pragma once


namespace GV {


	/////////////////////////////////////////////////////////////////////
	/// Solve the stokes problem on a diffuse domain using a monolithic solver
	/////////////////////////////////////////////////////////////////////
	template<typename StokesProblem>
	inline void ddm_stokes_monolithic(StokesProblem& stokes) {
		GUTIL_TIMER("Monolithic solve");
		std::cout << "Pressure dofs:\n" << stokes.p_handler << "\n";
		std::cout << "velocity dofs:\n" << stokes.u_handler << "\n";

		using Scalar_t   = typename StokesProblem::Scalar_t;
		using EigenSpMat = typename StokesProblem::EigenSpMat;
		using EigenVec   = typename StokesProblem::EigenVec;

		stokes.update_eps();

		const size_t nv = stokes.u_handler.n_dofs();
		const size_t np = stokes.p_handler.n_dofs();
		const size_t n  = 3*nv + np;
		GUTIL_LOG("Build LHS");
		EigenSpMat A(n,n);
		{
			auto triplets = stokes.make_monolithic_mat_coo();
			A.setFromTriplets(triplets.begin(), triplets.end());
		}

		std::vector<Scalar_t> rhs = stokes.make_monolithic_rhs();
		Eigen::Map<EigenVec> b(rhs.data(), rhs.size());

		//get initial guess
		std::vector<Scalar_t> x; x.reserve(b.size());
		x.insert(x.end(), stokes.u_coefs.get_coefs(0).begin(), stokes.u_coefs.get_coefs(0).end());
		x.insert(x.end(), stokes.u_coefs.get_coefs(1).begin(), stokes.u_coefs.get_coefs(1).end());
		x.insert(x.end(), stokes.u_coefs.get_coefs(2).begin(), stokes.u_coefs.get_coefs(2).end());
		x.insert(x.end(), stokes.p_coefs.get_coefs(0).begin(), stokes.p_coefs.get_coefs(0).end());
		GUTIL_ASSERT(x.size()== (size_t) b.size());

		//set up solver
		GUTIL_LOG("Build Preconditioner");
		auto A_vv = stokes.make_v_v_block();
		auto Mp   = stokes.make_pressure_mass_mat();

		typename StokesProblem::MonolithicSolver solver;
		solver.preconditioner().setup(A_vv, Mp, stokes.viscosity, stokes.inner_iter.max_iter, stokes.inner_iter.rel_tol);
		solver.compute(A);

		//solve
		GUTIL_LOG("Solve");
		solver.setMaxIterations(stokes.outer_iter.max_iter);
		Eigen::Map<EigenVec> xm(x.data(), x.size());
		xm = solver.solveWithGuess(b,xm);
		
		//copy back
		for (int j=0; j<3; ++j) {
			std::copy(x.data() + j*nv, x.data()+(j+1)*nv, stokes.u_coefs.get_coefs(j).data());
		}
		std::copy(x.data() + 3*nv, x.data()+3*nv+np, stokes.p_coefs.get_coefs(0).data());

		//convergence information
		if (solver.info() != Eigen::Success) {
			GUTIL_ERROR("WARNING: monolothic solver did not converge");
		}

		GUTIL_LOG("solver_iterations= ", solver.iterations(), " Rdiv= ", stokes.Rdiv(Mp, stokes.u_coefs.get_coefs(0), stokes.u_coefs.get_coefs(1), stokes.u_coefs.get_coefs(2)));
	}




	/////////////////////////////////////////////////////////////////////
	/// Solve the stokes problem on a diffuse domain using the standard Uzawa
	/// method.
	/////////////////////////////////////////////////////////////////////
	template<typename StokesProblem>
	inline void ddm_stokes_standard_uzawa(StokesProblem& stokes) {
		GUTIL_TIMER("Standard Uzawa solve");
		std::cout << "Pressure dofs:\n" << stokes.p_handler << "\n";
		std::cout << "velocity dofs:\n" << stokes.u_handler << "\n";

		using Scalar_t = typename StokesProblem::Scalar_t;
		using CompMat_t = typename StokesProblem::EigenCompMat;

		//set up linear and bilinear forms
		stokes.update_eps();

		//copy the velocity components to a dense column-major matrix
		//so that each component can be solved simultaneously
		CompMat_t X = stokes.get_velocity_as_mat();
		
		//build the velocity-velcity matrix
		auto A = stokes.make_v_v_block();

		//set up solver
		typename StokesProblem::InnerSolver solver;
		solver.compute(A);
		solver.setMaxIterations(stokes.inner_iter.max_iter);

		//build pressure mass matrix for projections and convergence
		auto Mp = stokes.make_pressure_mass_mat();
		Eigen::ConjugateGradient<typename StokesProblem::EigenSpMat, Eigen::Lower|Eigen::Upper, typename StokesProblem::DiagPrecon> p_solver;
		p_solver.compute(Mp);
		p_solver.setMaxIterations(stokes.inner_iter.max_iter);

		for (int i=0; i<stokes.outer_iter.max_iter; ++i) {
			GUTIL_TIMER("Uzawa iteration ", i+1, "/", stokes.outer_iter.max_iter);
			//set up RHS
			CompMat_t B = stokes.assemble_momentum_rhs(stokes.p_coefs.get_coefs(0));

			//solve velocity part of uzawa iteration
			X = solver.solveWithGuess(B, X);
			GUTIL_LOG("  velocity solve: iterations=", solver.iterations(), " error=", solver.error());
			if (solver.info() != Eigen::Success) { GUTIL_LOG("  WARNING: velocity solve did not converge"); }

			//update pressure part of uzawa iteration
			auto div_u = stokes.apply_Gt(X);
			auto q     = p_solver.solve(div_u);   // Mp*q = div_u
			GUTIL_LOG("  pressure solve: iterations=", p_solver.iterations(), " error=", p_solver.error());
			if (p_solver.info() != Eigen::Success) { GUTIL_LOG("  WARNING: pressure solve did not converge"); }

			std::span<Scalar_t> p_coefs = stokes.p_coefs.get_coefs(0);
			const Scalar_t update_scale = stokes.viscosity * stokes.relax_w;
			GUTIL_OMP(parallel)
			{
				gutil::OmpIndexRange range(p_coefs.size());
				GUTIL_SIMD()
				for (size_t idx=range.begin; idx<range.end; ++idx) {
					p_coefs[idx] += update_scale * q[idx];
				}
			}
			stokes.pin_pressure();

			//check for convergence
			Scalar_t rdiv = stokes.Rdiv(Mp,X);
			GUTIL_LOG("Rdiv= ", rdiv);
			if ( rdiv < stokes.outer_iter.rel_tol) {break;}
		}
		
		//copy back to the coefficient handlers
		stokes.copy_back_velocity(X);
	}


	template<typename StokesProblem>
	inline void ddm_stokes_cd_uzawa(StokesProblem& stokes) {
		GUTIL_TIMER("CD-U solve");
		std::cout << "Pressure dofs:\n" << stokes.p_handler << "\n";
		std::cout << "velocity dofs:\n" << stokes.u_handler << "\n";
		using Scalar_t  = typename StokesProblem::Scalar_t;
		using CompMat_t = typename StokesProblem::EigenCompMat;
		using EigenVec  = typename StokesProblem::EigenVec;

		stokes.update_eps();

		auto G_form0  = stokes.template make_G_form<0>();
		auto G_form1  = stokes.template make_G_form<1>();
		auto G_form2  = stokes.template make_G_form<2>();
		auto Gt_form0 = stokes.template make_Gt_form<0>();
		auto Gt_form1 = stokes.template make_Gt_form<1>();
		auto Gt_form2 = stokes.template make_Gt_form<2>();
		auto F_form0  = stokes.template make_acc_form<0>();
		auto F_form1  = stokes.template make_acc_form<1>();
		auto F_form2  = stokes.template make_acc_form<2>();
		auto Mv_form  = stokes.make_M_velocity_form();   // needed for the genuine <Gd,z> L2 inner product

		const size_t n   = stokes.u_handler.n_dofs();
		const size_t n_p = stokes.p_handler.n_dofs();

		CompMat_t X(n, 3);
		for (int i=0; i<3; ++i) {
			X.col(i) = Eigen::Map<EigenVec>(stokes.u_coefs.get_coefs(i).data(), n);
		}

		auto A = stokes.make_v_v_block();

		typename StokesProblem::InnerSolver solver;
		solver.compute(A);
		solver.setMaxIterations(stokes.inner_iter.max_iter);

		auto Mp = stokes.make_pressure_mass_mat();
		Eigen::ConjugateGradient<typename StokesProblem::EigenSpMat, Eigen::Lower|Eigen::Upper, DiagonalPreconditioner<std::vector<Scalar_t>>> p_solver;
		p_solver.compute(Mp);
		p_solver.setMaxIterations(stokes.inner_iter.max_iter);

		// --- Initialization (paper eq. 6): K*u^1 = f - G*p^0, q^1 = h - G^t*u^1, d^1 = -q^1 ---
		{
			CompMat_t B = CompMat_t::Zero(n,3);
			G_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(0).data(), n), stokes.p_coefs.get_coefs(0));
			if (stokes.body_acceleration[0]!=Scalar_t{0}) { F_form0.evaluate_vector(std::span<Scalar_t>(B.col(0).data(), n)); }
			G_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(1).data(), n), stokes.p_coefs.get_coefs(0));
			if (stokes.body_acceleration[1]!=Scalar_t{0}) { F_form1.evaluate_vector(std::span<Scalar_t>(B.col(1).data(), n)); }
			G_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(2).data(), n), stokes.p_coefs.get_coefs(0));
			if (stokes.body_acceleration[2]!=Scalar_t{0}) { F_form2.evaluate_vector(std::span<Scalar_t>(B.col(2).data(), n)); }
			X = solver.solveWithGuess(B, X);
		}

		EigenVec div_u = EigenVec::Zero(n_p);
		Gt_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), n_p), std::span<const Scalar_t>(X.col(0).data(), n));
		Gt_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), n_p), std::span<const Scalar_t>(X.col(1).data(), n));
		Gt_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), n_p), std::span<const Scalar_t>(X.col(2).data(), n));
		EigenVec q = -p_solver.solve(div_u);
		EigenVec d = -q;

		// <q,q> = q^t*Mp*q = q^t*(Mp*q) = q^t*(-div_u), since Mp*q=-div_u by
		// construction -- avoids a second mass-matrix application entirely
		Scalar_t q_l2 = -( q.dot(div_u) );

		for (int i=0; i<stokes.outer_iter.max_iter; ++i) {
			GUTIL_TIMER("CD-U iteration ", i+1, "/", stokes.outer_iter.max_iter);

			CompMat_t Gd = CompMat_t::Zero(n,3);
			G_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(Gd.col(0).data(), n), std::span<const Scalar_t>(d.data(), n_p));
			G_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(Gd.col(1).data(), n), std::span<const Scalar_t>(d.data(), n_p));
			G_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(Gd.col(2).data(), n), std::span<const Scalar_t>(d.data(), n_p));

			CompMat_t Z = solver.solve(Gd);
			GUTIL_LOG("  auxiliary solve: iterations=", solver.iterations(), " error=", solver.error());
			if (solver.info() != Eigen::Success) { GUTIL_LOG("  WARNING: auxiliary solve did not converge"); }

			// <Gd,z> = sum over 3 components of the genuine velocity-mass-weighted
			// cross term (eq. 11's <nabla*d,z>) -- NOT a plain Euclidean dot product
			Scalar_t denom = Scalar_t{0};
			for (int c=0; c<3; ++c) {
				denom += Mv_form.evaluate_form(std::span<const Scalar_t>(Gd.col(c).data(), n), std::span<const Scalar_t>(Z.col(c).data(), n));
			}
			GUTIL_ASSERT(denom != Scalar_t{0});
			const Scalar_t alpha = q_l2 / denom;
			GUTIL_LOG("  alpha=", alpha);

			std::span<Scalar_t> p_coefs = stokes.p_coefs.get_coefs(0);
			GUTIL_OMP(parallel)
			{
				gutil::OmpIndexRange range(p_coefs.size());
				GUTIL_SIMD()
				for (size_t idx=range.begin; idx<range.end; ++idx) { p_coefs[idx] += alpha * d[idx]; }
			}
			X -= alpha * Z;
			stokes.pin_pressure();

			EigenVec div_u_new = EigenVec::Zero(n_p);
			Gt_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u_new.data(), n_p), std::span<const Scalar_t>(X.col(0).data(), n));
			Gt_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u_new.data(), n_p), std::span<const Scalar_t>(X.col(1).data(), n));
			Gt_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u_new.data(), n_p), std::span<const Scalar_t>(X.col(2).data(), n));
			EigenVec q_new = -p_solver.solve(div_u_new);
			GUTIL_LOG("  pressure solve: iterations=", p_solver.iterations(), " error=", p_solver.error());

			const Scalar_t q_new_l2 = -( q_new.dot(div_u_new) );   // same shortcut: q_new^t*Mp*q_new = q_new^t*(-div_u_new)
			const Scalar_t beta = q_new_l2 / q_l2;
			GUTIL_LOG("  beta=", beta);
			d = -q_new + beta * d;
			q = q_new;
			q_l2 = q_new_l2;

			Scalar_t rdiv = stokes.Rdiv(Mp, std::span<Scalar_t>(X.col(0).data(), n), std::span<Scalar_t>(X.col(1).data(), n), std::span<Scalar_t>(X.col(2).data(), n));
			GUTIL_LOG("Rdiv= ", rdiv);
			if (rdiv < stokes.outer_iter.rel_tol) {break;}
		}

		for (int j=0; j<3; ++j) {
			std::copy(X.col(j).data(), X.col(j).data()+n, stokes.u_coefs.get_coefs(j).data());
		}
	}
}