#pragma once


namespace GV {


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
		std::cout << "eps= " << stokes.read_eps() << "\n";

		auto A_form   = stokes.make_A_form();			//velocity-velocity block (same for each component)
		auto Ap_form  = stokes.make_A_penalty_form();	//penalty portion of velocity-velocity block

		auto G_form0  = stokes.template make_G_form<0>();		//upper-right velocity-pressure block
		auto G_form1  = stokes.template make_G_form<1>();		//upper-right velocity-pressure block
		auto G_form2  = stokes.template make_G_form<2>();		//upper-right velocity-pressure block

		auto Gt_form0 = stokes.template make_Gt_form<0>();		//lower-left pressure-velocity block
		auto Gt_form1 = stokes.template make_Gt_form<1>();		//lower-left pressure-velocity block
		auto Gt_form2 = stokes.template make_Gt_form<2>();		//lower-left pressure-velocity block

		auto F_form0  = stokes.template make_acc_form<0>();		//body acceleration linear form
		auto F_form1  = stokes.template make_acc_form<1>();		//body acceleration linear form
		auto F_form2  = stokes.template make_acc_form<2>();		//body acceleration linear form

		//copy the velocity components to a dense column-major matrix
		//so that each component can be solved simultaneously
		const size_t n = stokes.u_handler.n_dofs();

		CompMat_t X(n, 3);
		for (int i=0; i<3; ++i) {
			X.col(i) = Eigen::Map<typename StokesProblem::EigenVec>(stokes.u_coefs.get_coefs(i).data(), n); //copies into X
		}

		//build the velocity-velcity matrix
		std::vector<typename StokesProblem::Triplet_t> triplets;
		A_form.build_triplets(triplets);
		Ap_form.build_triplets(triplets);

		typename StokesProblem::EigenSpMat A(n,n);
		A.setFromTriplets(triplets.begin(), triplets.end());
		triplets.clear(); triplets.shrink_to_fit();


		//set up solver
		typename StokesProblem::InnerSolver solver;
		solver.compute(A);
		solver.setMaxIterations(stokes.inner_iter.max_iter);

		//build pressure mass matrix for projections and convergence
		auto Mp = stokes.make_pressure_mass_mat();
		typename StokesProblem::InnerSolver p_solver;
		p_solver.compute(Mp);
		p_solver.setMaxIterations(stokes.inner_iter.max_iter);

		for (int i=0; i<stokes.outer_iter.max_iter; ++i) {
			GUTIL_TIMER("Uzawa iteration ", i+1, "/", stokes.outer_iter.max_iter);
			//set up RHS
			CompMat_t B = CompMat_t::Zero(n,3);
			G_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(0).data(), n), stokes.p_coefs.get_coefs(0));
			if (stokes.body_acceleration[0]!=Scalar_t{0}) {
				F_form0.evaluate_vector(std::span<Scalar_t>(B.col(0).data(), n));
			}
			G_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(1).data(), n), stokes.p_coefs.get_coefs(0));
			if (stokes.body_acceleration[1]!=Scalar_t{0}) {
				F_form1.evaluate_vector(std::span<Scalar_t>(B.col(1).data(), n));
			}
			G_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(B.col(2).data(), n), stokes.p_coefs.get_coefs(0));
			if (stokes.body_acceleration[2]!=Scalar_t{0}) {
				F_form2.evaluate_vector(std::span<Scalar_t>(B.col(2).data(), n));
			}


			//solve velocity part of uzawa iteration
			X = solver.solveWithGuess(B, X);
			GUTIL_LOG("  velocity solve: iterations=", solver.iterations(), " error=", solver.error());
			if (solver.info() != Eigen::Success) { GUTIL_LOG("  WARNING: velocity solve did not converge"); }

			//update pressure part of uzawa iteration
			typename StokesProblem::EigenVec div_u = StokesProblem::EigenVec::Zero(stokes.p_handler.n_dofs());
			Gt_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), div_u.size()), std::span<const Scalar_t>(X.col(0).data(), n));
			Gt_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), div_u.size()), std::span<const Scalar_t>(X.col(1).data(), n));
			Gt_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), div_u.size()), std::span<const Scalar_t>(X.col(2).data(), n));

			typename StokesProblem::EigenVec q = p_solver.solve(div_u);   // Mp*q = div_u
			GUTIL_LOG("  pressure solve: iterations=", p_solver.iterations(), " error=", p_solver.error());
			if (p_solver.info() != Eigen::Success) { GUTIL_LOG("  WARNING: pressure solve did not converge"); }

			std::span<Scalar_t> p_coefs = stokes.p_coefs.get_coefs(0);
			const Scalar_t update_scale = stokes.viscosity * stokes.relax_w;
			GUTIL_OMP(parallel)
			{
				OmpIndexRange range(p_coefs.size());
				GUTIL_SIMD()
				for (size_t idx=range.begin; idx<range.end; ++idx) {
					p_coefs[idx] += update_scale * q[idx];
				}
			}
			stokes.pin_pressure();

			//check for convergence
			Scalar_t rdiv = stokes.Rdiv(Mp, std::span<Scalar_t>(X.col(0).data(), n), std::span<Scalar_t>(X.col(1).data(), n), std::span<Scalar_t>(X.col(2).data(), n));
			GUTIL_LOG("Rdiv= ", rdiv);
			if ( rdiv < stokes.outer_iter.rel_tol) {break;}
		}

		//copy back to the coefficient handlers
		for (int j=0; j<3; ++j) {
			std::copy(X.col(j).data(), X.col(j).data()+n, stokes.u_coefs.get_coefs(j).data());
		}
	}


	template<typename StokesProblem>
	inline void ddm_stokes_adaptive_uzawa(StokesProblem& stokes) {
		GUTIL_TIMER("Adaptive Uzawa solve");
		std::cout << "Pressure dofs:\n" << stokes.p_handler << "\n";
		std::cout << "velocity dofs:\n" << stokes.u_handler << "\n";
		using Scalar_t  = typename StokesProblem::Scalar_t;
		using CompMat_t = typename StokesProblem::EigenCompMat;
		using EigenVec  = typename StokesProblem::EigenVec;

		stokes.update_eps();

		auto A_form   = stokes.make_A_form();
		auto Ap_form  = stokes.make_A_penalty_form();
		auto G_form0  = stokes.template make_G_form<0>();
		auto G_form1  = stokes.template make_G_form<1>();
		auto G_form2  = stokes.template make_G_form<2>();
		auto Gt_form0 = stokes.template make_Gt_form<0>();
		auto Gt_form1 = stokes.template make_Gt_form<1>();
		auto Gt_form2 = stokes.template make_Gt_form<2>();
		auto F_form0  = stokes.template make_acc_form<0>();
		auto F_form1  = stokes.template make_acc_form<1>();
		auto F_form2  = stokes.template make_acc_form<2>();

		const size_t n   = stokes.u_handler.n_dofs();
		const size_t n_p = stokes.p_handler.n_dofs();

		CompMat_t X(n, 3);
		for (int i=0; i<3; ++i) {
			X.col(i) = Eigen::Map<EigenVec>(stokes.u_coefs.get_coefs(i).data(), n);
		}

		std::vector<typename StokesProblem::Triplet_t> triplets;
		A_form.build_triplets(triplets);
		Ap_form.build_triplets(triplets);
		typename StokesProblem::EigenSpMat A(n,n);
		A.setFromTriplets(triplets.begin(), triplets.end());
		triplets.clear(); triplets.shrink_to_fit();

		typename StokesProblem::InnerSolver solver;
		solver.compute(A);
		solver.setMaxIterations(stokes.inner_iter.max_iter);

		auto Mp = stokes.make_pressure_mass_mat();
		typename StokesProblem::InnerSolver p_solver;
		p_solver.compute(Mp);
		p_solver.setMaxIterations(stokes.inner_iter.max_iter);

		// initial residual/velocity setup: solve K*u^1 = f - G*p^0 once, to get
		// a genuine starting point matching the paper's own initialization
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

		for (int i=0; i<stokes.outer_iter.max_iter; ++i) {
			GUTIL_TIMER("Adaptive Uzawa iteration ", i+1, "/", stokes.outer_iter.max_iter);

			// q^k: L2-projected residual, matching Rdiv's own established formula (Mp*q = -G^t*u, h=0)
			EigenVec div_u = EigenVec::Zero(n_p);
			Gt_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), n_p), std::span<const Scalar_t>(X.col(0).data(), n));
			Gt_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), n_p), std::span<const Scalar_t>(X.col(1).data(), n));
			Gt_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(div_u.data(), n_p), std::span<const Scalar_t>(X.col(2).data(), n));
			EigenVec q = p_solver.solve(div_u);
			q = -q;   // q^k = h - G^t*u^k = -div_u's own projection, per eq.7/8 with h=0

			// K*z^k = G*q^k -- auxiliary velocity solve, same structure as the main momentum solve
			CompMat_t Gq = CompMat_t::Zero(n,3);
			G_form0.mat_vec_multiply_accumulate(std::span<Scalar_t>(Gq.col(0).data(), n), std::span<const Scalar_t>(q.data(), n_p));
			G_form1.mat_vec_multiply_accumulate(std::span<Scalar_t>(Gq.col(1).data(), n), std::span<const Scalar_t>(q.data(), n_p));
			G_form2.mat_vec_multiply_accumulate(std::span<Scalar_t>(Gq.col(2).data(), n), std::span<const Scalar_t>(q.data(), n_p));

			CompMat_t Z = CompMat_t::Zero(n,3);   // starting guess 0 for the auxiliary solve each iteration
			Z = solver.solveWithGuess(Gq, Z);
			GUTIL_LOG("  auxiliary solve: iterations=", solver.iterations(), " error=", solver.error());

			// omega^k = (q,q) / (Gq . z), summed across all 3 velocity components for the denominator
			const Scalar_t numerator = q.dot(q);
			Scalar_t denominator = Scalar_t{0};
			for (int c=0; c<3; ++c) {
				denominator += Eigen::Map<EigenVec>(Gq.col(c).data(), n).dot(Eigen::Map<EigenVec>(Z.col(c).data(), n));
			}
			GUTIL_ASSERT(denominator != Scalar_t{0});
			const Scalar_t omega = numerator / denominator;
			GUTIL_LOG("  omega=", omega);

			// p^{k+1} = p^k - omega*q^k ; u^{k+1} = u^k + omega*z^k
			std::span<Scalar_t> p_coefs = stokes.p_coefs.get_coefs(0);
			GUTIL_OMP(parallel)
			{
				OmpIndexRange range(p_coefs.size());
				GUTIL_SIMD()
				for (size_t idx=range.begin; idx<range.end; ++idx) { p_coefs[idx] -= omega * q[idx]; }
			}
			stokes.pin_pressure();
			X += omega * Z;

			Scalar_t rdiv = stokes.Rdiv(Mp, std::span<Scalar_t>(X.col(0).data(), n), std::span<Scalar_t>(X.col(1).data(), n), std::span<Scalar_t>(X.col(2).data(), n));
			GUTIL_LOG("Rdiv= ", rdiv);
			if (rdiv < stokes.outer_iter.rel_tol) {break;}
		}

		for (int j=0; j<3; ++j) {
			std::copy(X.col(j).data(), X.col(j).data()+n, stokes.u_coefs.get_coefs(j).data());
		}
	}
}