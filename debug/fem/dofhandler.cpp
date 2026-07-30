#include "gutil.hpp"

#include "mesh/voxel_mesh_unstructured(NEW).hpp"
#include "diffuse_domain/signed_distance.hpp"
#include "fem/handlers/dofhandler.hpp"
#include "fem/dofs/voxel_dof_Q1.hpp"

/////////////////////////////////////////////////////////////////
/// Collect compile-time options.
///		GV_TEST_DOMAIN_PERIOD:
///				0bzyx (e.g., 6=0b110 is periodic in z and y but not in x)
///
/// 	GV_TEST_MAX_DEPTH:
///				the maximum depth of the voxel mesh, larger than 10 is 
///				not recommended due to memory. larger than 15 is not supported.
///
///		GV_TEST_DOMAIN_SIZE:
///				the size of the domain (half sidelength) in each axis
///				the domain is centered at the origin
/////////////////////////////////////////////////////////////////

#ifndef GV_TEST_DOMAIN_PERIOD
	#define GV_TEST_DOMAIN_PERIOD 7
#endif

#ifndef GV_TEST_MAX_DEPTH
	#define GV_TEST_MAX_DEPTH 8
#endif

#ifndef GV_TEST_DOMAIN_SIZE
	#define GV_TEST_DOMAIN_SIZE 1
#endif

/////////////////////////////////////////////////////////////////
/// Assemble aliases for this test.
/////////////////////////////////////////////////////////////////
using Point_t		= gutil::Point<3,double>;
using Box_t 		= gutil::Box<3,double>;
using Sphere_t		= gutil::Sphere<3,double>;
using Mesh_t		= GV::UnstructuredVoxelMesh<GV_TEST_MAX_DEPTH>;
using Vert_t		= typename Mesh_t::VoxelVertex;
using Elem_t		= typename Mesh_t::VoxelElement;
using Assembly_t	= GV::SignedDistanceSpheres<double,GV_TEST_DOMAIN_PERIOD>;

using D_KEY_t       = Vert_t;
using DOF_t         = GV::VoxelQ1<D_KEY_t>;
using Handler_t     = GV::DofHandler<Mesh_t,DOF_t>;

inline constexpr double DOMAIN_SIZE = GV_TEST_DOMAIN_SIZE;
inline constexpr size_t MAX_DEPTH = GV_TEST_MAX_DEPTH;
inline constexpr Box_t domain{ {-DOMAIN_SIZE,-DOMAIN_SIZE,-DOMAIN_SIZE},
							   { DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE}};


/////////////////////////////////////////////////////////////////
/// Generate a pseudo-random periodic collection of spheres with
/// no intersections.
/////////////////////////////////////////////////////////////////
Assembly_t generate_assembly(const std::string& test_name, size_t n_spheres, size_t seed, double min_r, double max_r) {
	gutil::Logger::log("Generating Periodic Assembly: START");
	std::cout << "\ntest= " << test_name << "\n"
			  << "\tn_spheres= " << n_spheres << "\n"
			  << "\tseed= " << seed << "\n"
			  << "\tmin_r= " << min_r << "\n"
			  << "\tmax_r= " << max_r << "\n" << std::flush;
	gutil::LogTime timer{"Generating Periodic Assembly: DONE"};
	

	// set up assembly and rng
	Assembly_t assembly{domain};

	auto random_point = gutil::UniformRandomPoint<Point_t,true>();
	random_point.set_parameters(-DOMAIN_SIZE,DOMAIN_SIZE);
	random_point.set_seed(seed);
	
	auto random_radius = gutil::UniformRandomPoint<Point_t,true>();
	random_radius.set_parameters(min_r,max_r);
	random_radius.set_seed(seed);

	//insert spheres
	const size_t max_attempts = 500 * n_spheres;
	size_t attempts = 0;

	while(assembly.size() < n_spheres && attempts<max_attempts) {
		++attempts;
		Sphere_t candidate(random_point(), random_radius.scalar());
		if (!assembly.collides(candidate)) {
			assembly.push_back(std::move(candidate));
		}
	}

	if (assembly.size() < n_spheres) {
		gutil::Logger::log("WARNING: only placed ", assembly.size(), "/", n_spheres,
							" disjoint periodic spheres after ", max_attempts, " attempts.\n",
							"Try a smaller number of spheres or radii.");
	}

	std::string filename = test_name + "_spheres.txt";
	gutil::write_spheres_to_file(filename, assembly.as_cspan());

	return assembly;
}


/////////////////////////////////////////////////////////////////
/// Generate a mesh of the spheres, assign dofs
/////////////////////////////////////////////////////////////////
void test_dof_handler(const std::string& test_name, const Assembly_t& assembly, size_t initial_depth) {
	Mesh_t mesh(domain, initial_depth);
	mesh.update_unstructured();
	mesh.collect_vertices();
	Handler_t handler(mesh);
	handler.init_dofs();
	handler.collect_dofs();
	handler.sort_dofs();

	

	//save mesh and record dof information
	const std::string filename = test_name + "_dof.vtk";
	mesh.save_as_binary(filename);

	auto pt_dof_active_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				const DOF_t dof(static_cast<D_KEY_t>(vtx));
				const size_t idx = handler.global_number(dof);
				return (idx<handler.n_dofs()) ? static_cast<int>(idx) : -1;
			}, "active_dof_index");

	auto el_dof_count_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				std::vector<DOF_t> dofs;
				std::vector<size_t> idxs;
				handler.get_active_dofs_full_hierarchical(el,dofs,idxs);
				return dofs.size();
			}, "n_active_dofs");

	mesh.append_point_data_field_binary(filename, "point", pt_dof_active_lookup);
	mesh.append_cell_data_field_binary(filename, "element", el_dof_count_lookup);
}










int main(int argc, char* argv[]) {
	std::vector<std::string> args(argv, argv+argc);
	size_t seed  = 0;
	size_t n_spheres = 100;
	double min_r = 0.01;
	double max_r = 0.5;
	size_t initial_depth = MAX_DEPTH/3;
	std::string test_name = "dofhandler";

	for (size_t i=0; i<args.size(); ++i) {
		if      (args[i] == "-N") { n_spheres = atoi(args[++i].c_str());}
		else if (args[i] == "-R0") { min_r = atof(args[++i].c_str());}
		else if (args[i] == "-R1") { max_r = atof(args[++i].c_str());}
		else if (args[i] == "-S")  { seed = atoi(args[++i].c_str());}
		else if (args[i] == "-name") {test_name = args[++i];}
		else if (args[i] == "-ID") {initial_depth = atoi(args[++i].c_str());}
	}


	Assembly_t assembly = generate_assembly(test_name, n_spheres, seed, min_r, max_r);
	test_dof_handler(test_name, assembly, initial_depth);
	return 0;
}