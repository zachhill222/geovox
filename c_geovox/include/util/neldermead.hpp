#ifndef NELDERMEAD_H
#define NELDERMEAD_H

#include "util/polytope.hpp"
#include "util/point.hpp"
#include <cmath>


#define NELDERMEAD_ALPHA 1.0
#define NELDERMEAD_GAMMA 1.25
#define NELDERMEAD_RHO 0.5
#define NELDERMEAD_SIGMA 0.125
#define NELDERMEAD_MAXITER 2

namespace GeoVox::util{
	template<size_t dim, typename F, typename Args>
	GeoVox::util::Point<dim> neldermead(F fun, GeoVox::util::Simplex<dim>& simplex, const Args& args);

	template<size_t dim>
	void sortSimplex(double (&fvals)[dim+1], GeoVox::util::Simplex<dim>& simplex);


	template<size_t dim, typename F, typename Args>
	GeoVox::util::Point<dim> neldermead(F fun, GeoVox::util::Simplex<dim>& simplex, const Args& args){
		// std::cout << "starting neldermead\n";
		//INITIALIZE SIMPLEX AND FVALS
		double fvals[dim+1];
		for (long unsigned int i=0; i<dim+1; i++){
			fvals[i] = fun(simplex[i], args);
		}
		

		//MAIN LOOP
		for (int loop_idx=0; loop_idx<NELDERMEAD_MAXITER; loop_idx++){
			//PREPARE LOOP
			bool shrink = false;
			sortSimplex(fvals, simplex);


			//COMPUTE CENTROID
			GeoVox::util::Point<dim> centroid = simplex.center();


			//REFLECTED CHECK
			GeoVox::util::Point<dim> reflected = centroid + NELDERMEAD_ALPHA*(centroid - simplex[dim]);
			double f_reflected = fun(reflected, args);

			if (fvals[0] <= f_reflected && f_reflected < fvals[dim-1]){
				fvals[dim] = f_reflected;
				simplex[dim] = reflected;
				continue;
			}


			//EXPANSION CHECK
			if (f_reflected < fvals[0]){
				GeoVox::util::Point<dim> expanded = centroid + NELDERMEAD_GAMMA*(reflected - centroid);
				double f_expanded = fun(expanded, args);

				if (f_expanded < f_reflected){
					fvals[dim] = f_expanded;
					simplex[dim] = expanded;
					continue;
				}else{
					fvals[dim] = f_reflected;
					simplex[dim] = reflected;
					continue;
				}
			}


			//CONTRACTION CHECK
			if (f_reflected < fvals[dim]){
				GeoVox::util::Point<dim> contracted = centroid + NELDERMEAD_RHO*(reflected-centroid);
				double f_contracted = fun(contracted, args);

				if (f_contracted < fvals[dim]){
					fvals[dim] = f_contracted;
					simplex[dim] = contracted;
					continue;
				}else{
					shrink = true;
				}
			}else{
				GeoVox::util::Point<dim> contracted = centroid + NELDERMEAD_RHO*(simplex[dim]-centroid);
				double f_contracted = fun(contracted, args);

				if (f_contracted < fvals[dim]){
					fvals[dim] = f_contracted;
					simplex[dim] = contracted;
					continue;
				}else{
					shrink = true;
				}
			}


			//SHRINK STEP
			if (shrink){
				for (long unsigned int i=1; i<dim+1; i++){
					simplex[i] = simplex[0] + NELDERMEAD_SIGMA*(simplex[i] - simplex[0]);
					fvals[i] = fun(simplex[i], args);
				}
			}
		}

		sortSimplex(fvals, simplex);
		// std::cout << fvals[0] << std::endl;
		return simplex[0];
	}




	template<size_t dim>
	void sortSimplex(double (&fvals)[dim+1], GeoVox::util::Simplex<dim>& simplex){
		bool ordered = false;
		for (long unsigned int n=0; n<dim+1; n++){
			ordered = true;
			for (long unsigned int i=0; i<dim; i++){
				if (fvals[i] > fvals[i+1]){
					//swap fvals
					double temp = fvals[i+1];
					fvals[i+1]  = fvals[i];
					fvals[i]    = temp;

					//swap points in simplex
					GeoVox::util::Point<dim> tempPoint = simplex[i+1];
					simplex[i+1] = simplex[i];
					simplex[i]   = tempPoint;

					ordered = false;
				}
			}

			if (ordered){
				return;
			}
		}

	}

}


#endif