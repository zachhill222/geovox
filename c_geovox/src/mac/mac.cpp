#include "mac/mac.hpp"

namespace GeoVox::mac{
	/////////////////////////////////////////////////////////////////
	////////// PRESSURE DERIVATIVES (B_transpose ~ grad()) //////////
	/////////////////////////////////////////////////////////////////


	//derivative of P at U DOFs
	VectorXd MacMesh::dPdX(const VectorXd& variable) const{
		VectorXd result(N.prod());
		double h_1 = 1.0/H[0];

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					result[index(i,j,k)] = h_1*(variable[index(i,j,k)]-variable[index(i-1,j,k)]);
				}
			}
		}

		return result;
	}

	//derivative of P at V DOFs
	VectorXd MacMesh::dPdY(const VectorXd& variable) const{
		VectorXd result(N.prod());
		double h_1 = 1.0/H[1];

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					result[index(i,j,k)] = h_1*(variable[index(i,j,k)]-variable[index(i,j-1,k)]);
				}
			}
		}

		return result;
	}

	//derivative of P at W DOFs
	VectorXd MacMesh::dPdZ(const VectorXd& variable) const{
		VectorXd result(N.prod());
		double h_1 = 1.0/H[2];

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					result[index(i,j,k)] = h_1*(variable[index(i,j,k)]-variable[index(i,j,k-1)]);
				}
			}
		}

		return result;
	}



	/////////////////////////////////////////////////////////////////
	////////// VELOCITY DERIVATIVES (B ~ -div() /////////////////////
	/////////////////////////////////////////////////////////////////

	//derivative of U at P DOFs
	VectorXd MacMesh::dUdX(const VectorXd& variable) const{
		VectorXd result(N.prod());
		double h_1 = 1.0/H[0];

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					result[index(i,j,k)] = h_1*(variable[index(i,j,k)]-variable[index(i+1,j,k)]);
				}
			}
		}

		return result;
	}

	//derivative of V at P DOFs
	VectorXd MacMesh::dVdY(const VectorXd& variable) const{
		VectorXd result(N.prod());
		double h_1 = 1.0/H[1];

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					result[index(i,j,k)] = h_1*(variable[index(i,j,k)]-variable[index(i,j+1,k)]);
				}
			}
		}

		return result;
	}

	//derivative of W at P DOFs
	VectorXd MacMesh::dWdZ(const VectorXd& variable) const{
		VectorXd result(N.prod());
		double h_1 = 1.0/H[2];

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					result[index(i,j,k)] = h_1*(variable[index(i,j,k)]-variable[index(i,j,k+1)]);
				}
			}
		}

		return mu*result;
	}


	/////////////////////////////////////////////////////////////////
	////////// NEGATIVE LAPLACIANS A, Ap ~ -laplace()  //////////////
	/////////////////////////////////////////////////////////////////
	//-laplacian of u,v,w
	VectorXd MacMesh::A(const VectorXd& variable) const{
		VectorXd result(N.prod());
		VectorXd h_2 = (H.array()*H.array()).inverse();

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					long unsigned int P = index(i,j,k);

					result[P] = h_2[0]*(2*variable[P] - variable[index(i-1,j,k)] - variable[index(i+1,j,k)]);
					result[P]+= h_2[1]*(2*variable[P] - variable[index(i,j-1,k)] - variable[index(i,j+1,k)]);
					result[P]+= h_2[2]*(2*variable[P] - variable[index(i,j,k-1)] - variable[index(i,j,k+1)]);
				}
			}
		}

		return result;
	}

	//-laplacian of p
	VectorXd MacMesh::Ap(const VectorXd& variable) const {
		return dUdX(dPdX(variable)) + dVdY(dPdY(variable)) + dWdZ(dPdZ(variable));
	}




	void MacMesh::GS_relax_velocity(){
		Point3 h_1 = H.cwiseInverse(); // 1/h
		Point3 H_2 = 2.0*(H.array()*H.array()).inverse(); //  2/(h*h)
		double C   = 1.0/H_2.sum();

		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){

					//VELOCITY INDICES
					long unsigned int P = index(i,j,k);
					long unsigned int W = index(i-1,j,k);
					long unsigned int E = index(i+1,j,k);
					long unsigned int S = index(i,j-1,k);
					long unsigned int N = index(i,j+1,k);
					long unsigned int B = index(i,j,k-1);
					long unsigned int T = index(i,j,k+1);


					//UPDATE VELOCITY
					if (u_mask(i,j,k)==MAC_DOMAIN_MARKER){
						u[P] = C*( f1[P] + mu*(H_2[0]*(u[E]+u[W]) + H_2[1]*(u[N]+u[S]) + H_2[2]*(u[T]+u[B])) + h_1[0]*(p[W]-p[P]) );
					}
					
					if (v_mask(i,j,k)==MAC_DOMAIN_MARKER){
						v[P] = C*( f2[P] + mu*(H_2[0]*(v[E]+v[W]) + H_2[1]*(v[N]+v[S]) + H_2[2]*(v[T]+v[B])) + h_1[1]*(p[S]-p[P]) );
					}
					
					if (w_mask(i,j,k)==MAC_DOMAIN_MARKER){
						w[P] = C*( f3[P] + mu*(H_2[0]*(w[E]+w[W]) + H_2[1]*(w[N]+w[S]) + H_2[2]*(w[T]+w[B])) + h_1[2]*(p[B]-p[P]) );
					}
				}
			}
		}
	}

	


	VectorXd MacMesh::GS_relax_p() const{
		//Compute residual
		VectorXd res = g - dUdX(u) - dVdY(v) - dWdZ(w);


		//Relax
		VectorXd Ep = VectorXd::Zero(p.size());

		Point3 H_2 = 2.0*(H.array()*H.array()).inverse(); //  2/(h*h)
		double C   = 1.0/H_2.sum();

		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){

				//BLACK GROUP
				for (long unsigned int i=0; i<N[0]; i++){

					long unsigned int P = index(i,j,k);
					long unsigned int W = index(i-1,j,k);
					long unsigned int E = index(i+1,j,k);
					long unsigned int S = index(i,j-1,k);
					long unsigned int N = index(i,j+1,k);
					long unsigned int B = index(i,j,k-1);
					long unsigned int T = index(i,j,k+1);
					
					Ep[P] = C*( res[P] + H_2[0]*(Ep[E]+Ep[W]) + H_2[1]*(Ep[N]+Ep[S]) + H_2[2]*(Ep[T]+Ep[B]) );
				}
			}
		}

		return Ep;
	}

	void MacMesh::DGS(){
		//RELAX VELOCITY
		GS_relax_velocity();

		//RELAX PRESSURE
		VectorXd Ep = GS_relax_p();

		//UPDATE VELOCITY
		u += dPdX(Ep);
		v += dPdY(Ep);
		w += dPdZ(Ep);

		//UPDATE PRESSURE
		p -= Ap(Ep);
		p = (p.array()-p.mean()).matrix(); //FORCE MEAN PRESSURE TO ZERO


		//SET SOLUTION TO ZERO IN ROCK DOMAIN
		setRockVelocity();
	}
	
	void MacMesh::setRockVelocity(){
		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					long unsigned int P = index(i,j,k);
					if (!u_mask.markers[P]==MAC_DOMAIN_MARKER) {u[P]=0;}
					if (!v_mask.markers[P]==MAC_DOMAIN_MARKER) {v[P]=0;}
					if (!w_mask.markers[P]==MAC_DOMAIN_MARKER) {w[P]=0;}
				}
			}
		}
	}

	void MacMesh::solve(const int max_iter){
		for (int iter_count=0; iter_count<max_iter; iter_count++){
			//UPDATE
			DGS();


			// if (iter_count%100 == 0){
			// 	//COMPUTE RESIDUALS
			// 	VectorXd res = VectorXd::Zero(4);

			// 	VectorXd res_u = A(u)+dPdX(p)-f1;
			// 	VectorXd res_v = A(v)+dPdY(p)-f2;
			// 	VectorXd res_w = A(w)+dPdZ(p)-f3;
			// 	VectorXd res_p = dUdX(u)+dVdY(v)+dWdZ(w)-g;


			// 	#pragma omp parallel for collapse(3)
			// 	for (long unsigned int k=0; k<N[2]; k++){
			// 		for (long unsigned int j=0; j<N[1]; j++){
			// 			for (long unsigned int i=0; i<N[0]; i++){
			// 				long unsigned int P = index(i,j,k);
			// 				if (!u_mask.markers[P]==MAC_DOMAIN_MARKER) {res[0] += res_u[P]*res_u[P];}
			// 				if (!v_mask.markers[P]==MAC_DOMAIN_MARKER) {res[1] += res_v[P]*res_v[P];}
			// 				if (!w_mask.markers[P]==MAC_DOMAIN_MARKER) {res[2] += res_w[P]*res_w[P];}
			// 				if (!p_mask.markers[P]==MAC_DOMAIN_MARKER) {res[3] += res_p[P]*res_p[P];}
			// 			}
			// 		}
			// 	}

			// 	for (int i=0; i<4; i++){
			// 		res[i] = sqrt(res[i]);
			// 	}

			// 	std::cout << "DGS iteration " << iter_count << ": res=" << res.transpose() << std::endl;
			// }
		}
	}

	void MacMesh::solve_multigrid(int m){
		//check to see if mesh can be coarsened
		bool coarsest = false;
		if (N[0]%2 or N[1]%2 or N[2]%2==0){
			coarsest = true;
		}

		if (N[0]<16 or N[1]<16 or N[2]<16){
			coarsest = true;
		}

		if (coarsest){
			solve(100);
			return;
		}

		//presmooth
		for (int i=0; i<m; i++){
			DGS();
		}

		//compute residual
		VectorXd res_u = A(u)+dPdX(p)-f1;
		VectorXd res_v = A(v)+dPdY(p)-f2;
		VectorXd res_w = A(w)+dPdZ(p)-f3;
		VectorXd res_p = dUdX(u)+dVdY(v)+dWdZ(w)-g;

		//setup problem on coarse mesh
		long unsigned int M[3] = {N[0]/2, N[1]/2, N[2]/2};
		MacMesh coarse_mesh(box, M, assembly);
		coarse_mesh.mu = mu;
		coarsen(coarse_mesh.f1,coarse_mesh.f2,coarse_mesh.f3,coarse_mesh.g);

		//solve problem on coarse mesh
		coarse_mesh.solve_multigrid(m);

		//prolongation
		VectorXd dU = VectorXd::Zero(p.size());
		VectorXd dV = VectorXd::Zero(p.size());
		VectorXd dW = VectorXd::Zero(p.size());
		VectorXd dP = VectorXd::Zero(p.size());

		coarse_mesh.refine(dU,dV,dW,dP);
		u+=dU;
		v+=dV;
		w+=dW;
		p+=dP;

		//postsmooth
		solve(m); //should write GS in reverse order
	}

	void MacMesh::coarsen(VectorXd& U, VectorXd& V, VectorXd& W, VectorXd& P){
		long unsigned int N_2[3] {N[0]/2, N[1]/2, N[2]/2};

		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N_2[2]; k++){
			for (long unsigned int j=0; j<N_2[1]; j++){
				for (long unsigned int i=0; i<N_2[0]; i++){
					long unsigned int idx = i + N_2[0]*(j + N_2[1]*k);

					long unsigned int ii = 2*i;
					long unsigned int jj = 2*j;
					long unsigned int kk = 2*k;

					U[idx] =    u[index(ii-1,jj,kk)] + u[index(ii-1,jj+1,kk)] + u[index(ii-1,jj+1,kk+1)] + u[index(ii-1,jj,kk+1)];
					U[idx]+= 2*(u[index(ii  ,jj,kk)] + u[index(ii  ,jj+1,kk)] + u[index(ii  ,jj+1,kk+1)] + u[index(ii  ,jj,kk+1)]);
					U[idx]+=    u[index(ii+1,jj,kk)] + u[index(ii+1,jj+1,kk)] + u[index(ii+1,jj+1,kk+1)] + u[index(ii+1,jj,kk+1)];
					U[idx]*=0.0625; // 1/16

					V[idx] =    v[index(ii-1,jj,kk)] + v[index(ii-1,jj+1,kk)] + v[index(ii-1,jj+1,kk+1)] + v[index(ii-1,jj,kk+1)];
					V[idx]+= 2*(v[index(ii  ,jj,kk)] + v[index(ii  ,jj+1,kk)] + v[index(ii  ,jj+1,kk+1)] + v[index(ii  ,jj,kk+1)]);
					V[idx]+=    v[index(ii+1,jj,kk)] + v[index(ii+1,jj+1,kk)] + v[index(ii+1,jj+1,kk+1)] + v[index(ii+1,jj,kk+1)];
					V[idx]*=0.0625; // 1/16

					W[idx] =    w[index(ii-1,jj,kk)] + w[index(ii-1,jj+1,kk)] + w[index(ii-1,jj+1,kk+1)] + w[index(ii-1,jj,kk+1)];
					W[idx]+= 2*(w[index(ii  ,jj,kk)] + w[index(ii  ,jj+1,kk)] + w[index(ii  ,jj+1,kk+1)] + w[index(ii  ,jj,kk+1)]);
					W[idx]+=    w[index(ii+1,jj,kk)] + w[index(ii+1,jj+1,kk)] + w[index(ii+1,jj+1,kk+1)] + w[index(ii+1,jj,kk+1)];
					W[idx]*=0.0625; // 1/16

					P[idx] = 0.125*(p[index(ii,jj,kk)]       + p[index(ii+1,jj,kk)]   + p[index(ii,jj+1,kk)]   + p[index(ii,jj,kk+1)] /
						          + p[index(ii+1,jj+1,kk+1)] + p[index(ii,jj+1,kk+1)] + p[index(ii+1,jj,kk+1)] + p[index(ii+1,jj+1,kk)]);
				}
			}
		}
	}

	void MacMesh::refine(VectorXd& U, VectorXd& V, VectorXd& W, VectorXd& P){
		long unsigned int N2[3] {N[0]*2, N[1]*2, N[2]*2};


		//PRESSURE AND FIRST INTERPOLATIONS
		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){

					long unsigned int idx;
					long unsigned int ii = 2*i;
					long unsigned int jj = 2*j;
					long unsigned int kk = 2*k;
					
					

					idx = ii + N2[0]*(jj  +N[1]*(kk  ));
					U[idx] = 0.75*u[index(i,j,k)] + 0.25*u[index(i,j-1,k-1)];
					V[idx] = 0.75*v[index(i,j,k)] + 0.25*v[index(i-1,j,k-1)];
					W[idx] = 0.75*w[index(i,j,k)] + 0.25*w[index(i-1,j-1,k)];
					P[idx] = p[index(i,j,k)];
					P[idx+1] = p[index(i,j,k)];

					idx = ii + N2[0]*(jj+1+N[1]*(kk  ));
					U[idx] = 0.75*u[index(i,j,k)] + 0.25*u[index(i,j+1,k-1)];
					V[idx] = 0.75*v[index(i,j,k)] + 0.25*v[index(i+1,j,k-1)];
					W[idx] = 0.75*w[index(i,j,k)] + 0.25*w[index(i-1,j+1,k)];
					P[idx] = p[index(i,j,k)];
					P[idx+1] = p[index(i,j,k)];
					
					idx = ii + N2[0]*(jj  +N[1]*(kk+1));
					U[idx] = 0.75*u[index(i,j,k)] + 0.25*u[index(i,j-1,k+1)];
					V[idx] = 0.75*v[index(i,j,k)] + 0.25*v[index(i-1,j,k+1)];
					W[idx] = 0.75*w[index(i,j,k)] + 0.25*w[index(i+1,j-1,k)];
					P[idx] = p[index(i,j,k)];
					P[idx+1] = p[index(i,j,k)];
					
					idx = ii + N2[0]*(jj+1+N[1]*(kk+1));
					U[idx] = 0.75*u[index(i,j,k)] + 0.25*u[index(i,j+1,k+1)];
					V[idx] = 0.75*v[index(i,j,k)] + 0.25*v[index(i+1,j,k+1)];
					W[idx] = 0.75*w[index(i,j,k)] + 0.25*w[index(i+1,j+1,k)];
					P[idx] = p[index(i,j,k)];
					P[idx+1] = p[index(i,j,k)];
				}
			}
		}

		//SECOND INTERPOLATIONS
		#pragma omp parallel for collapse(3)
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){

					long unsigned int idx;
					long unsigned int ii = 2*i;
					long unsigned int jj = 2*j;
					long unsigned int kk = 2*k;
					
					idx = ii + N2[0]*(jj  +N[1]*(kk  ));
					U[idx+1] = 0.5*(U[idx]+U[idx+2]);
					V[idx+1] = 0.5*(V[idx]+V[idx+2]);
					W[idx+1] = 0.5*(W[idx]+W[idx+2]);

					idx = ii + N2[0]*(jj+1+N[1]*(kk  ));
					U[idx+1] = 0.5*(U[idx]+U[idx+2]);
					V[idx+1] = 0.5*(V[idx]+V[idx+2]);
					W[idx+1] = 0.5*(W[idx]+W[idx+2]);
					
					idx = ii + N2[0]*(jj  +N[1]*(kk+1));
					U[idx+1] = 0.5*(U[idx]+U[idx+2]);
					V[idx+1] = 0.5*(V[idx]+V[idx+2]);
					W[idx+1] = 0.5*(W[idx]+W[idx+2]);
					
					idx = ii + N2[0]*(jj+1+N[1]*(kk+1));
					U[idx+1] = 0.5*(U[idx]+U[idx+2]);
					V[idx+1] = 0.5*(V[idx]+V[idx+2]);
					W[idx+1] = 0.5*(W[idx]+W[idx+2]);
				}
			}
		}
	}




	void MacMesh::saveas(const std::string filename) const{
		//////////////// OPEN FILE ////////////////
		std::ofstream solutionfile(filename);

		if (not solutionfile.is_open()){
			std::cout << "Couldn't write to " << filename << std::endl;
			solutionfile.close();
			return;
		}

		//////////////// WRITE TO FILE ////////////////
		std::stringstream buffer;

		//HEADER
		buffer << "# vtk DataFile Version 2.0\n";
		buffer << "Mesh Data\n";
		buffer << "ASCII\n\n";

		//POINTS (CENTROIDS)
		buffer << "DATASET STRUCTURED_POINTS\n";
		buffer << "DIMENSIONS " << N[0]+1 << " " << N[1]+1 << " " << N[2]+1 << "\n";
		buffer << "ORIGIN " << p_mask.box.low()-0.5*p_mask.H << "\n"; //OFFSET FOR CENTROID PRESENTATION
		buffer << "SPACING " << p_mask.H << "\n\n";

		solutionfile << buffer.rdbuf();
		buffer.str("");


		//PRESSURE
		// std::cout << "pressure\n"; 
		buffer << "CELL_DATA " << N.prod() << "\n";
		buffer << "SCALARS pressure double\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					buffer << p[index(i,j,k)] << "\n";
				}
			}
		}
		buffer << "\n";
		
		solutionfile << buffer.rdbuf();
		buffer.str("");


		//VELOCITY
		// std::cout << "velocity\n";
		buffer << "VECTORS colocated_velocity double\n";
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					double vx = 0.5*(u[index(i,j,k)]+u[index(i+1,j,k)]);
					double vy = 0.5*(v[index(i,j,k)]+v[index(i+1,j,k)]);
					double vz = 0.5*(w[index(i,j,k)]+w[index(i+1,j,k)]);
					
					buffer << vx << "\t" << vy << "\t" << vz << "\n";
				}
			}
		}
		buffer << "\n";
		
		solutionfile << buffer.rdbuf();
		buffer.str("");


		//MASK
		// std::cout << "mask\n";
		buffer << "SCALARS mask integer\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					buffer << p_mask(i,j,k) << "\n";
				}
			}
		}
		buffer << "\n";


		//DIVERGENCE
		VectorXd neg_divergence = dUdX(u)+dVdY(v)+dWdZ(w);
		buffer << "SCALARS velocity_divergence double\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<N[2]; k++){
			for (long unsigned int j=0; j<N[1]; j++){
				for (long unsigned int i=0; i<N[0]; i++){
					buffer << neg_divergence[index(i,j,k)] << "\n";
				}
			}
		}
		buffer << "\n";
		
		solutionfile << buffer.rdbuf();
		buffer.str("");

		//////////////// CLOSE FILE ////////////////
		solutionfile.close();
	}
}