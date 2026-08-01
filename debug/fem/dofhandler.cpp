#include "gutil.hpp"

#include "simd_keys/mesh_key.hpp"
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
	#define GV_TEST_MAX_DEPTH 5
#endif

#ifndef GV_TEST_DOMAIN_SIZE
	#define GV_TEST_DOMAIN_SIZE 1
#endif

#ifndef GV_TEST_DOF_PERIOD
	#define GV_TEST_DOF_PERIOD 0
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

using D_KEY         = Vert_t::PeriodicVariant<GV_TEST_DOF_PERIOD>;
using DOF_t         = GV::VoxelQ1<D_KEY>;
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
	gutil::Logger::log("test_dof_handler (start)");
	gutil::Logger::log("initializing mesh (start)");
	gutil::LogTime* timer = new gutil::LogTime{"initializing mesh (done)"};
	
	Mesh_t mesh(domain, initial_depth);
	mesh.update_unstructured();
	Handler_t handler(mesh);
	handler.init_dofs();
	handler.collect_dofs();

	gutil::Logger::log("dofhandler has ", handler.n_dofs(), " dofs");
	gutil::Logger::log("mesh has ", mesh.n_elements(), " elements");
	delete timer;

	{
		gutil::Logger::log("refining dofs (start)");
		gutil::LogTime t{"refining dofs (done)"};

		//refine mesh near the boundary
		auto action = [&](std::span<DOF_t> dofs, double tol) {
			for (DOF_t dof : dofs) {
				GUTIL_ASSERT(dof.is_valid())
				Vert_t vtx = static_cast<Vert_t>(dof.key);
				if (std::abs( assembly.signed_distance(mesh.geo_coord(vtx)) ) < tol) {
					handler.refine_quasi_hierarchical(dof);
				}
			}
		};

		double tol = 0.1;
		for (size_t i=initial_depth; i<MAX_DEPTH; ++i) {
			handler.dispatch_parallel_active_dof(action, tol);
			handler.wait();

			//refining dofs request mesh refinement
			handler.collect_dofs();
			mesh.process_refine();
			mesh.update_unstructured();
		}

		handler.refine_all_quasi_hierarchical();
		mesh.process_refine();
		mesh.update_unstructured();
		gutil::Logger::log("dofhandler has ", handler.n_dofs(), " dofs");
		gutil::Logger::log("mesh has ", mesh.n_elements(), " elements");
	}

	{
		gutil::Logger::log("unrefining dofs (start)");
		gutil::LogTime t{"unrefining dofs (done)"};

		for (auto it=mesh.element_begin(); it!=mesh.element_end(); ++it) {
			GUTIL_ASSERT(mesh.is_active(*it));
			Point_t v0 = mesh.geo_coord(it->vertex(0));
			Point_t v1 = mesh.geo_coord(it->vertex(7));
			if (std::abs( assembly.signed_distance(0.5*(v0+v1)) > 0.01 )) {
				for (Vert_t vtx : it->vertices()) {
					GUTIL_ASSERT(vtx.is_valid())
					D_KEY key = static_cast<D_KEY>(vtx);
					GUTIL_ASSERT(DOF_t{key}.is_valid())
					handler.unrefine_quasi_hierarchical(DOF_t{key});
				}
			}
		}

		handler.wait();
		handler.collect_dofs();

		gutil::Logger::log("dofhandler has ", handler.n_dofs(), " dofs");
		gutil::Logger::log("mesh has ", mesh.n_elements(), " elements");
	}

	{
		gutil::Logger::log("remove mesh elements with no conformal dofs dofs (start)");
		gutil::LogTime t{"remove mesh elements with no conformal dofs dofs (done)"};

		for (auto it=mesh.element_begin(); it!=mesh.element_end(); ++it) {
			std::vector<DOF_t> dofs;
			std::vector<size_t> numbers;
			handler.get_active_dofs_conformal(*it,dofs,numbers);
			if (dofs.empty()) {
				std::cout << "no dofs\n";
				mesh.unrefine(*it);
			}
		}

		mesh.process_unrefine();
		mesh.update_unstructured();
		gutil::Logger::log("dofhandler has ", handler.n_dofs(), " dofs");
		gutil::Logger::log("mesh has ", mesh.n_elements(), " elements");
	}







	gutil::Logger::log("saving file");

	//save mesh and record dof information
	const std::string filename = test_name + "_dof.vtk";
	mesh.update_unstructured();
	mesh.collect_vertices();
	mesh.save_as_binary(filename);

	auto pt_sd_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {return assembly.signed_distance(mesh.geo_coord(vtx));},
			"signed_distance");

	auto pt_dof_active_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				vtx = handler.get_dof_vertex(vtx);
				return vtx.exists() ? handler.global_number(DOF_t{static_cast<D_KEY>(vtx)}) : -1;
			}, "active_dof_index");

	auto pt_dof_refinable_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				vtx = handler.get_dof_vertex(vtx);
				return vtx.exists() ? handler.is_refinable(DOF_t{static_cast<D_KEY>(vtx)}) : -1;
			}, "dof_refinable");

	auto pt_dof_depth_lookup = GV::make_feature_lookup<Vert_t>(
			[&](Vert_t vtx) {
				vtx = handler.get_dof_vertex(vtx);
				return vtx.exists() ? vtx.depth() : -1;
			}, "depth");

	auto el_dof_count_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				std::vector<DOF_t> dofs;
				std::vector<size_t> idxs;
				// handler.get_active_dofs_conformal(el,dofs,idxs);
				handler.get_active_dofs_full_hierarchical(el,dofs,idxs);
				return dofs.size();
			}, "n_active_dofs");

	auto el_depth_lookup = GV::make_feature_lookup<Elem_t>(
			[&](Elem_t el) {
				return el.depth();
			}, "depth");

	mesh.append_point_data_field_binary(filename, "point", pt_sd_lookup, pt_dof_active_lookup, pt_dof_refinable_lookup,pt_dof_depth_lookup);
	mesh.append_cell_data_field_binary(filename, "element", el_dof_count_lookup,el_depth_lookup);
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