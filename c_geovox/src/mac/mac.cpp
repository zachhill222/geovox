#include "mac/mac.hpp"

namespace GeoVox::mac{
	std::vector<double> MacMesh::discrete_laplacian(const StructuredPoints &mask, const std::vector<double> &vals) const{
		//mask corresponds to vals
		std::vector<double> result(mask.N[0]*mask.N[1]*mask.N[2], 0);
		Point3 H_2 = Point3(1.0/(mask.H[0]*mask.H[0]), 1.0/(mask.H[1]*mask.H[1]), 1.0/(mask.H[2]*mask.H[2])); //  1/(h*h)

		for (long unsigned int k=0; k<mask.N[2]; k++){
			for (long unsigned int j=0; j<mask.N[1]; j++){
				for (long unsigned int i=0; i<mask.N[0]; i++){
					
					long unsigned int P = mask.index(i  , j  , k  );
					if (mask.pointMarkers[P] != MAC_DOMAIN_MARKER){
						continue;
					}

					long unsigned int E = mask.index((i+1)%mask.N[0], j  , k  );
					long unsigned int W = mask.index((i-1)%mask.N[0], j  , k  );
					
					long unsigned int N = mask.index(i  , (j+1)%mask.N[1], k  );
					long unsigned int S = mask.index(i  , (j-1)%mask.N[1], k  );
					
					long unsigned int T = mask.index(i  , j  , (k+1)%mask.N[2]);
					long unsigned int B = mask.index(i  , j  , (k-1)%mask.N[2]);
					
					
					result[P] = (vals[E] - 2*vals[P] + vals[W])*H_2[0];
					result[P]+= (vals[N] - 2*vals[P] + vals[S])*H_2[1];
					result[P]+= (vals[T] - 2*vals[P] + vals[B])*H_2[2];
				}
			}
		}

		return result;
	}



	std::vector<double> MacMesh::discrete_partial_t(const StructuredPoints &eval_dof_mask, const StructuredPoints &given_dof_mask, const std::vector<double> &vals, const int direction) const{
		std::vector<double> result(eval_dof_mask.N[0]*eval_dof_mask.N[1]*eval_dof_mask.N[2], 0);
		Point3 H_2 = Point3(1.0/(given_dof_mask.H[0]), 1.0/(given_dof_mask.H[1]), 1.0/(given_dof_mask.H[2])); //  1/h

		for (long unsigned int k=0; k<eval_dof_mask.N[2]; k++){
			for (long unsigned int j=0; j<eval_dof_mask.N[1]; j++){
				for (long unsigned int i=0; i<eval_dof_mask.N[0]; i++){
					
					long unsigned int P = eval_dof_mask.index(i  , j  , k  );
					long unsigned int B;
					double h_2;
					if (eval_dof_mask.pointMarkers[P] != MAC_DOMAIN_MARKER){
						continue;
					}

					long unsigned int F = given_dof_mask.index(i%given_dof_mask.N[0], j%given_dof_mask.N[1], k%given_dof_mask.N[2]);
					switch (direction){
					case 0:
						B = given_dof_mask.index((i-1)%given_dof_mask.N[0], j  , k  );
						h_2 = H_2[0];
						break;
					case 1:
						B = given_dof_mask.index(i  , (j-1)%given_dof_mask.N[1], k  );
						h_2 = H_2[1];
						break;
					case 2:
						B = given_dof_mask.index(i  , j  , (k-1)%given_dof_mask.N[2]);
						h_2 = H_2[2];
						break;
					default:
						h_2 = 0;
						throw std::range_error("unknown direction\n");
					}
					
					result[P] = (vals[F] - vals[B])*h_2;
				}
			}
		}

		return result;
	}



	std::vector<double> MacMesh::discrete_partial(const StructuredPoints &eval_dof_mask, const StructuredPoints &given_dof_mask, const std::vector<double> &vals, const int direction) const{
		std::vector<double> result(eval_dof_mask.N[0]*eval_dof_mask.N[1]*eval_dof_mask.N[2], 0);
		Point3 H_2 = Point3(1.0/(given_dof_mask.H[0]), 1.0/(given_dof_mask.H[1]), 1.0/(given_dof_mask.H[2])); //  1/h

		for (long unsigned int k=0; k<eval_dof_mask.N[2]; k++){
			for (long unsigned int j=0; j<eval_dof_mask.N[1]; j++){
				for (long unsigned int i=0; i<eval_dof_mask.N[0]; i++){
					
					long unsigned int P = eval_dof_mask.index(i  , j  , k  );
					if (eval_dof_mask.pointMarkers[P] != MAC_DOMAIN_MARKER){
						continue;
					}

					
					long unsigned int F = given_dof_mask.index(i%given_dof_mask.N[0], j%given_dof_mask.N[1], k%given_dof_mask.N[2]);
					long unsigned int B;
					double h_2;
					switch (direction){
					case 0:
						B = given_dof_mask.index((i+1)%given_dof_mask.N[0], j  , k  );
						h_2 = H_2[0];
						break;
					case 1:
						B = given_dof_mask.index(i  , (j+1)%given_dof_mask.N[1], k  );
						h_2 = H_2[1];
						break;
					case 2:
						B = given_dof_mask.index(i  , j  , (k+1)%given_dof_mask.N[2]);
						h_2 = H_2[2];
						break;
					default:
						h_2 = 0;
						throw std::range_error("unknown direction\n");
					}
					
					result[P] = (vals[F] - vals[B])*h_2;
				}
			}
		}

		return result;
	}


	std::vector<double> MacMesh::BB_t(const std::vector<double> &vals) const{
		std::vector<double> result = Bz(Bz_t(vals));
		
		std::vector<double> temp = By(By_t(vals));
		GeoVox::solvers::sum(result, temp);

		temp = Bx(Bx_t(vals));
		GeoVox::solvers::sum(result, temp);
		return result;
	}

	void MacMesh::DGS(){
		//RELAX X-VELOCITY
		std::vector<double> temp = Bx_t(p);
		std::vector<double> rhs = GeoVox::solvers::sum_times(f1, temp, -1.0);
		rhs = GeoVox::solvers::sum_times(rhs, Au(u), -1.0);

		std::function<std::vector<double>(std::vector<double>)> times_mat = std::bind(&MacMesh::Au, this, std::placeholders::_1);
		std::vector<double> step(u.size(),0);
		GeoVox::solvers::cg(step, times_mat, rhs, 10);

		GeoVox::solvers::sum(u, step);

		//RElAX Y-VELOCITY
		temp = By_t(p);
		rhs = GeoVox::solvers::sum_times(f2, temp, -1.0);
		temp = Av(v);
		rhs = GeoVox::solvers::sum_times(rhs, temp, -1.0);

		
		times_mat = std::bind(&MacMesh::Av, this, std::placeholders::_1);
		step = std::vector<double>(v.size(),0);
		GeoVox::solvers::cg(step, times_mat, rhs, 10);

		GeoVox::solvers::sum(v, step);

		//RElAX Z-VELOCITY
		temp = Bz_t(p);
		rhs = GeoVox::solvers::sum_times(f3, temp, -1.0);
		temp = Aw(w);
		rhs = GeoVox::solvers::sum_times(rhs, temp, -1.0);

		times_mat = std::bind(&MacMesh::Aw, this, std::placeholders::_1);
		step = std::vector<double>(w.size(),0);
		GeoVox::solvers::cg(step, times_mat, rhs, 10);

		GeoVox::solvers::sum(w, step);


		//RELAX MASS
		temp = Bx(u);
		rhs = GeoVox::solvers::sum_times(g, temp, -1.0);
		temp = By(v);
		rhs = GeoVox::solvers::sum_times(rhs, temp, -1.0);
		temp = Bz(w);
		rhs = GeoVox::solvers::sum_times(rhs, temp, -1.0);
		times_mat = std::bind(&MacMesh::BB_t, this, std::placeholders::_1);
		
		std::vector<double> ep(p.size(),0);
		GeoVox::solvers::cg(ep, times_mat, rhs, 10);


		//UPDATE VELOCITY AND PRESSURE
		temp = Bx_t(ep);
		GeoVox::solvers::sum(u, temp);
		temp = By_t(ep);
		GeoVox::solvers::sum(v, temp);
		temp = Bz_t(ep);
		GeoVox::solvers::sum(w, temp);
		temp = BB_t(ep);
		p = GeoVox::solvers::sum_times(p, temp, -1.0);
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
		buffer << "DIMENSIONS " << p_mask.N[0] << " " << p_mask.N[1] << " " << p_mask.N[2] << "\n";
		buffer << "ORIGIN " << p_mask.box.low().tostring() << "\n";
		buffer << "SPACING " << p_mask.H.tostring() << "\n\n";

		solutionfile << buffer.rdbuf();
		buffer.str("");


		//PRESSURE
		buffer << "POINT_DATA " << p_mask.N[0]*p_mask.N[1]*p_mask.N[2] << "\n";
		buffer << "SCALARS pressure float\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<p_mask.N[2]; k++){
			long unsigned int start_idx = p_mask.N[0]*p_mask.N[1]*k;
			for (long unsigned int ij=0; ij<p_mask.N[0]*p_mask.N[1]; ij++){
				switch (p_mask.pointMarkers[start_idx+ij]){
				case MAC_DOMAIN_MARKER:
					buffer << p[start_idx+ij] << " ";
					break;
				default:
					buffer << 0 << " ";
				}
			}
			buffer << "\n";
		}
		buffer << "\n";
		
		solutionfile << buffer.rdbuf();
		buffer.str("");


		//VELOCITY
		buffer << "POINT_DATA " << p_mask.N[0]*p_mask.N[1]*p_mask.N[2] << "\n";
		buffer << "VECTORS pressure float\n";
		buffer << "LOOKUP_TABLE default\n";
		for (long unsigned int k=0; k<p_mask.N[2]; k++){
			for (long unsigned int j=0; j<p_mask.N[1]; j++){
				for (long unsigned int i=0; i<p_mask.N[0]; i++){
					double vx = 0.5*(u[p_mask.index(i,j,k)+u[p_mask.index(i+1,j,k)]]);
					double vy = 0.5*(v[p_mask.index(i,j,k)+v[p_mask.index(i,j+1,k)]]);
					double vz = 0.5*(w[p_mask.index(i,j,k)+w[p_mask.index(i,j,k+1)]]);
					
					buffer << vx << " " << vy << " " << vz << " \t";
				}
			}
			buffer << "\n";
		}
		buffer << "\n";
		
		solutionfile << buffer.rdbuf();
		buffer.str("");

		//////////////// CLOSE FILE ////////////////
		solutionfile.close();

		

	}
}