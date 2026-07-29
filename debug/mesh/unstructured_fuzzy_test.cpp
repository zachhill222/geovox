#include "gutil.hpp"

#include "mesh/voxel_mesh_unstructured(NEW).hpp"
#include "diffuse_domain/signed_distance.hpp"

#include <string>
#include <vector>
#include <cstdlib>
#include <iostream>

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

inline constexpr double DOMAIN_SIZE = GV_TEST_DOMAIN_SIZE;
inline constexpr size_t MAX_DEPTH = GV_TEST_MAX_DEPTH;

inline constexpr Box_t domain{
						{-DOMAIN_SIZE,-DOMAIN_SIZE,-DOMAIN_SIZE},
						{ DOMAIN_SIZE, DOMAIN_SIZE, DOMAIN_SIZE}};

/////////////////////////////////////////////////////////////////
/// Generate a pseudo-random periodic collection of spheres with
/// no intersections.
/////////////////////////////////////////////////////////////////
Assembly_t generate_assembly(const std::string& test_name, size_t n_spheres=100, size_t seed=0, double min_r=0.01, double max_r=0.5) {
	gutil::Logger::log("Generating Periodic Assembly: START");
	std::cout << "\ntest= " << test_name << "\n"
			  << "\tn_spheres= " << n_spheres << "\n"
			  << "\tseed= " << seed << "\n"
			  << "\tmin_r= " << min_r << "\n"
			  << "\tmax_r= " << max_r << "\n" << std::flush;
	gutil::LogTime timer{"Generating Periodic Assembly: DONE"};
	

	// set up assembly and rng
	Assembly_t assembly{domain};

	auto random_point = gutil::UniformRandomPoint<Point_t,false>();
	random_point.set_parameters(-DOMAIN_SIZE,DOMAIN_SIZE);
	random_point.set_seed(seed);
	
	auto random_radius = gutil::UniformRandomPoint<Point_t,false>();
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
/// Generate a mesh of the spheres, randomly refine, refine the boundary,
/// and then randomly refine again.
/////////////////////////////////////////////////////////////////
Mesh_t generate_and_refine_mesh(const std::string& test_name, const Assembly_t& assembly, size_t seed=0) {
	gutil::Logger::log("Generating And Refining Mesh: START");
	std::cout << "\ntest= " << test_name << "\n"
			  << "\tn_spheres= " << assembly.size() << "\n"
			  << "\tseed= " << seed << "\n" << std::flush;
	gutil::LogTime timer{"Generating And Refining Mesh: DONE"};

	auto random_index = gutil::UniformRandomPoint<gutil::Point<1,size_t>,false>();
	random_index.set_parameters(size_t{0}, size_t{1}<<MAX_DEPTH);
	random_index.set_seed(seed);

	const size_t initial_depth = random_index.scalar() % 6;
	const size_t n_random_refines = random_index.scalar() % 100;
	const size_t target_depth = MAX_DEPTH;
	Mesh_t mesh(domain);
	{
		gutil::Logger::log("setting initial mesh (start)");
		gutil::LogTime t{"(done)"};
		mesh.set_depth(initial_depth);
	}

	{
		gutil::Logger::log("making ", n_random_refines, " random refinements (start)");
		gutil::LogTime t{"(done)"};
		for (size_t n=0; n<n_random_refines; ++n) {
			mesh.update_unstructured();

			//find a random active element
			size_t d, idx;
			while (true) {
				d = random_index.scalar() % (MAX_DEPTH+1);
				idx = random_index.scalar() % (mesh.get_layer(d).n_elements());
				if (mesh.get_layer(d).is_active(idx)) { break; }
			}

			//refine
			mesh.refine(Elem_t{d,idx});
			mesh.process_refine();
		}
	}

	{
		gutil::Logger::log("refining sphere surfaces to target depth ", target_depth, " (start)");
		gutil::LogTime t{"(done)"};
		auto pred = [&](Elem_t el) {
			if (el.depth() >= target_depth) {return false;}

			Point_t pt = mesh.geo_coord(el.vertex(0));
			Point_t diag = mesh.geo_coord(el.vertex(7)) - pt;
			pt += 0.5*diag;

			double dist = assembly.signed_distance(pt);
			return dist*dist < 0.5*gutil::squared_norm(diag);
		};

		for (size_t n=0; n<target_depth; ++n) {
			mesh.update_unstructured();
			mesh.refine(pred);
			mesh.process_refine();
		}
	}

	{
		gutil::Logger::log("making ", n_random_refines, " random refinements (start)");
		gutil::LogTime t{"(done)"};
		for (size_t n=0; n<n_random_refines; ++n) {
			mesh.update_unstructured();

			//find a random active element
			size_t d, idx;
			while (true) {
				d = random_index.scalar() % (MAX_DEPTH+1);
				idx = random_index.scalar() % (mesh.get_layer(d).n_elements());
				if (mesh.get_layer(d).is_active(idx)) { break; }
			}

			//refine
			mesh.refine(Elem_t{d,idx});
			mesh.process_refine();
		}
	}

	{
		gutil::Logger::log("saving mesh (start)");
		gutil::LogTime t{"(done)"};
		
		const double sdf_eps = gutil::ldexp(DOMAIN_SIZE, -int{target_depth}); //double the target depth cell size
		std::cout << "\tsdf_eps= " << sdf_eps << "\n" << std::flush;

		const std::string filename = test_name + "_initial_mesh.vtk";
		mesh.save_as_binary(filename);

		auto pt_sd_lookup = GV::make_feature_lookup<Vert_t>(
			[&assembly, &mesh](Vert_t vtx) {return assembly.signed_distance(mesh.geo_coord(vtx));},
			"signed_distance");

		auto pt_heaviside_lookup = GV::make_feature_lookup<Vert_t>(
				[&assembly, &mesh, sdf_eps](Vert_t vtx) {return assembly.heaviside(mesh.geo_coord(vtx), sdf_eps);},
				"heaviside");

		auto pt_dirac_lookup = GV::make_feature_lookup<Vert_t>(
				[&assembly, &mesh, sdf_eps](Vert_t vtx) {return assembly.dirac(mesh.geo_coord(vtx), sdf_eps);},
				"dirac");

		auto pt_depth_lookup = GV::make_feature_lookup<Vert_t>(
				[](Vert_t vtx) {return vtx.depth();},
				"depth"
			);

		auto pt_ijk_lookup = GV::make_feature_lookup<Vert_t>(
				[](Vert_t vtx) { return std::array<uint64_t,3>{vtx.i(), vtx.j(), vtx.k()}; },
				"ijk"
			);

		auto pt_d_lin_idx_lookup = GV::make_feature_lookup<Vert_t>(
				[](Vert_t vtx) { return vtx.depth_linear_index(); },
				"depth_linear_index"
			);


		auto el_depth_lookup = GV::make_feature_lookup<Elem_t>(
				[](Elem_t el) {return el.depth();},
				"depth"
			);

		auto el_ijk_lookup = GV::make_feature_lookup<Elem_t>(
				[](Elem_t el) { return std::array<uint64_t,3>{el.i(), el.j(), el.k()}; },
				"ijk"
			);

		auto el_d_lin_idx_lookup = GV::make_feature_lookup<Elem_t>(
				[](Elem_t el) { return el.depth_linear_index(); },
				"depth_linear_index"
			);

		auto el_color_lookup = GV::make_feature_lookup<Elem_t>(
				[](Elem_t el) { return el.color(); },
				"color"
			);

		mesh.append_point_data_field_binary(filename, "point_data", pt_sd_lookup, pt_heaviside_lookup, pt_dirac_lookup, pt_depth_lookup, pt_ijk_lookup,pt_d_lin_idx_lookup);
		mesh.append_cell_data_field_binary(filename, "element_data", el_depth_lookup, el_ijk_lookup, el_d_lin_idx_lookup, el_color_lookup);
	}

	return mesh;
}




