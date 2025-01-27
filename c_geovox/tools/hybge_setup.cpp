#include "util_module.hpp"
#include "geometry_module.hpp"

#include <stdio.h>
#include <getopt.h>
#include <string>
#include <sstream>
#include <fstream>

//DEFAULT SETTINGS
#define DEFAULT_N 100
#define DEFAULT_R 0.025
#define DEFAULT_DIR "./"
#define DEFAULT_NORMALIZE false
#define DEFAULT_VERBOSE false

#define DEFAULT_VISCOSITY 0.001
#define DEFAULT_HYBGE_SOLVER_MAX_ITER 2000
#define DEFAULT_HYBGE_SOLVER_ABSOLUTE_TOL 1e-8
#define DEFAULT_HYBGE_SOLVER_RELATIVE_TOL 1e-6
#define DEFAULT_HYBGE_SOLVER_VERBOSE 2

//SETTINGS CONTAINERS
struct hybge_parameters {
	double length;
	double width;
	double height;
	int solver_max_iterations;
	double solver_absolute_tolerance;
	double solver_relative_tolerance;
	int solver_verbose;
};

struct hybge_extra_parameters {
	double viscosity;
	bool user_specified_viscosity;
};

struct geovox_options {
	long unsigned int N;
	double trim;
	std::string dir;
	bool normalize;
	std::string assembly_file;
};


void help(void) {
	std::cout << "usage: hybge_setup [options] file\n";
	std::cout << "Options:\n";
	std::cout << "\t-h, --help\t\t\tdisplay this help message\n";
	std::cout << "\t-N <integer>\t\tdiscretization size (default of " << DEFAULT_N << ")\n";
	std::cout << "\t-r, --trim <float>\t\ttrim percent (default of " << DEFAULT_R << "), must be between 0 and 1.\n";
	std::cout << "\t-O, --out <directory>\t\toutput directory (default of " << DEFAULT_DIR << ")\n";
	std::cout << "\t-n, --normalize\t\tsets viscosity to 1 and the length scale of the domain to O(1), while keeping the correct aspect ratio\n";
	std::cout << "\t--viscosity <float>\t\tsets the viscosity of the fluid (default of " << DEFAULT_VISCOSITY << ")\n";
	std::cout << "\t--max_iter <int>\t\tsets the maximum number of outer iterations for the hybge solver (GMRES with restart, default of " << DEFAULT_HYBGE_SOLVER_MAX_ITER << ")\n";
	std::cout << "\t--abs_tol <float>\t\tsets the absolute tolerance for the hybge solver (default of " << DEFAULT_HYBGE_SOLVER_ABSOLUTE_TOL << ")\n";
	std::cout << "\t--rel_tol <float>\t\tsets the relative tolerance for the hybge solver (default of " << DEFAULT_HYBGE_SOLVER_RELATIVE_TOL << ")\n";
	std::cout << "\t--verbose <int>\t\t\tsets the verbose setting for the hybge solver (default of " << DEFAULT_HYBGE_SOLVER_VERBOSE << ")\n";

	std::cout << "\nOutput: in the directory specified by -O\n";
	std::cout << "\tGeometry.dat\t\ttext file that specifies the rock (1) phase and fluid (0) phase\n";
	std::cout << "\tParameters.dat\t\ttext file that specifies the physical dimensions of the domain and solver settings for HYBGE\n";
}


std::string print_options(const hybge_parameters &hybge_par, const hybge_extra_parameters &hybge_extra_par, const geovox_options &geovox_opts){
	std::stringstream options_text;

	options_text << "HYBGE PARAMETERS:\n";
	options_text << "length= " << hybge_par.length << std::endl;
	options_text << "width= " << hybge_par.width << std::endl;
	options_text << "height= " << hybge_par.height << std::endl;
	options_text << "solver_max_iterations= " << hybge_par.solver_max_iterations << std::endl;
	options_text << "solver_absolute_tolerance= " << hybge_par.solver_absolute_tolerance << std::endl;
	options_text << "solver_relative_tolerance= " << hybge_par.solver_relative_tolerance << std::endl;
	options_text << "solver_verbose= " << hybge_par.solver_verbose << std::endl;


	options_text << "\nHYBGE EXTRA PARAMETERS:\n";
	options_text << "viscosity= " << hybge_extra_par.viscosity << std::endl;
	options_text << "user_specified_viscosity= " << hybge_extra_par.user_specified_viscosity << std::endl;


	options_text << "\nGEOVOX OPTIONS\n";
	options_text << "N= " << geovox_opts.N << std::endl;
	options_text << "trim= " << geovox_opts.trim << std::endl;
	options_text << "dir= " << geovox_opts.dir << std::endl;
	options_text << "normalize= " << geovox_opts.normalize << std::endl;
	options_text << "assembly_file= " << geovox_opts.assembly_file << std::endl;

	return options_text.str();
}


int parse_options(int argc, char *const argv[], hybge_parameters &hybge_par, hybge_extra_parameters &hybge_extra_par, geovox_options &geovox_opts){
	//hybge solver defaults
	hybge_par.solver_max_iterations     = DEFAULT_HYBGE_SOLVER_MAX_ITER;
	hybge_par.solver_absolute_tolerance = DEFAULT_HYBGE_SOLVER_ABSOLUTE_TOL;
	hybge_par.solver_relative_tolerance = DEFAULT_HYBGE_SOLVER_RELATIVE_TOL;
	hybge_par.solver_verbose            = DEFAULT_HYBGE_SOLVER_VERBOSE;

	//hybge extra defaults
	hybge_extra_par.viscosity = DEFAULT_VISCOSITY;
	hybge_extra_par.user_specified_viscosity = false;

	//geovox defaults
	geovox_opts.N         = DEFAULT_N;
	geovox_opts.trim      = DEFAULT_R;
	geovox_opts.dir       = DEFAULT_DIR;
	geovox_opts.normalize = DEFAULT_NORMALIZE;


	//SET UP OPTION PARSING
	//track parse error
	bool opt_err = false;

	//long options
	static struct option long_options[] = {
		{"normalize", no_argument,       0, 'n'},
		{"help",	  no_argument,       0, 'h'},
		{"N",	  	  required_argument, 0, 'N'},
		{"trim",	  required_argument, 0, 'r'},
		{"out",	 	  required_argument, 0, 'O'},
		{"viscosity", required_argument, 0,  0 },
		{"max_iter",  required_argument, 0,  0 },
		{"abs_tol",   required_argument, 0,  0 },
		{"rel_tol",   required_argument, 0,  0 },
		{"verbose",   required_argument, 0,  0 },
		{0,           0,                 0,  0 }
	};
	int opt;
	int option_index = 0;


	//PARSE OPTIONS FROM COMMANDLINE
	while ((opt = getopt_long(argc, argv, "nhN:r:O:", long_options, &option_index)) != -1){
		switch (opt) {
		case 'n':
			geovox_opts.normalize = true;
			break;
		case 'h':
			help();
			return 0;
			break;
		case 'N':
			geovox_opts.N = atoi(optarg);
			break;
		case 'r':
			geovox_opts.trim = strtod(optarg, NULL);
			if (geovox_opts.trim <0 or geovox_opts.trim>= 1){
				std::cout << "ERROR: the trim value must be in the interval [0,1).\n";
				return 1;
			}
			break;
		case 'O':
			geovox_opts.dir = optarg;
			if (geovox_opts.dir[geovox_opts.dir.size()-1] != '/'){
				geovox_opts.dir += '/';
			}
			break;
		case 0: //long only opts
			if (strcmp("viscosity", long_options[option_index].name) == 0){
				hybge_extra_par.viscosity = strtod(optarg, NULL);
				hybge_extra_par.user_specified_viscosity = true;
			}else if (strcmp("max_iter", long_options[option_index].name) == 0){
				hybge_par.solver_max_iterations = atoi(optarg);
			}else if (strcmp("abs_tol", long_options[option_index].name) == 0){
				hybge_par.solver_absolute_tolerance = strtod(optarg, NULL);
			}else if (strcmp("rel_tol", long_options[option_index].name) == 0){
				hybge_par.solver_relative_tolerance = strtod(optarg, NULL);
			}else if (strcmp("verbose", long_options[option_index].name) == 0){
				hybge_par.solver_verbose = atoi(optarg);
			}
		case '?':
			opt_err = true;
		}
	}

	//GET FILENAME
	if (optind < argc) {
		geovox_opts.assembly_file = argv[optind++];
	}else{
		std::cout << "ERROR: an assembly file must be specified\n";
		help();
		return 1;
	}


	//display help message if there was a parse error
	if (opt_err){
		help();
		return 1;
	}

	return 0;
}




int main(int argc, char *argv[]){
	//get options
	static hybge_parameters hybge_par;
	static hybge_extra_parameters hybge_extra_par;
	static geovox_options geovox_opts;
	int flag;
	flag = parse_options(argc, argv, hybge_par, hybge_extra_par, geovox_opts);
	if (flag) {return 1;}

	// std::cout << print_options(hybge_par, hybge_extra_par, geovox_opts);


	//make geometry file
	std::string geometry_file = geovox_opts.dir+"Geometry.dat";
	GeoVox::geometry::Assembly A(geovox_opts.assembly_file, "-rrr-eps-xyz-q");
	A.divide(5); //create octree for faster sampling of points
	GeoVox::util::Box domain = (1-geovox_opts.trim) * A.box;
	long unsigned int N[3] {geovox_opts.N, geovox_opts.N, geovox_opts.N};
	A.save_geometry(geometry_file, domain, N);

	//make parameter file
	std::string parameter_file = geovox_opts.dir+"Parameters.dat";
	hybge_par.length = domain.sidelength()[0];
	hybge_par.width = domain.sidelength()[1];
	hybge_par.height = domain.sidelength()[2];

	if (geovox_opts.normalize){
		double L = std::max(hybge_par.length, hybge_par.width);
		L = std::max(L, hybge_par.height);

		hybge_par.length /= L;
		hybge_par.width  /= L;
		hybge_par.height /= L;
	}

	std::ofstream parameters(parameter_file);
	if (not parameters.is_open()){
		std::cout << "Couldn't write to " << parameter_file << std::endl;
		parameters.close();
		return 1;
	}

	parameters << "length= " << hybge_par.length << std::endl;
	parameters << "width= " << hybge_par.width << std::endl;
	parameters << "height= " << hybge_par.height << std::endl;
	parameters << "solver_max_iterations= " << hybge_par.solver_max_iterations << std::endl;
	parameters << "solver_absolute_tolerance= " << hybge_par.solver_absolute_tolerance << std::endl;
	parameters << "solver_relative_tolerance= " << hybge_par.solver_relative_tolerance << std::endl;
	parameters << "solver_verbose= " << hybge_par.solver_verbose << std::endl;
	parameters.close();


	//make extra parameter file
	std::string extra_parameter_file = geovox_opts.dir+"ExtraParameters.dat";
	std::ofstream extra_parameters(extra_parameter_file);
	if (not extra_parameters.is_open()){
		std::cout << "Couldn't write to " << parameter_file << std::endl;
		extra_parameters.close();
		return 1;
	}

	if (hybge_extra_par.user_specified_viscosity or !geovox_opts.normalize){
		extra_parameters << "viscosity= " << hybge_extra_par.viscosity << std::endl;
	}else{
		extra_parameters << "viscosity= " << 1 << std::endl;
	}
	extra_parameters.close();

}


