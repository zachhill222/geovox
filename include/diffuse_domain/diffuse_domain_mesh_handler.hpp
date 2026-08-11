#pragma once

#include "gutil.hpp"

#include "mesh/mesh.hpp"

#include <cstdint>
#include <string>


namespace GV {


	//////////////////////////////////////////////////////////////////
	/// A class to coordinate an assembly of particles with an unstructured
	/// voxel mesh.
	//////////////////////////////////////////////////////////////////
	template<typename AssemblyType>
	struct DiffuseDomainMeshHandler {
		
		
		//////////////////////////////////////////////////////////////
		/// Aliases and constants
		//////////////////////////////////////////////////////////////
		using Assembly_t 	= AssemblyType;
		using Particle_t    = typename Assembly_t::Particle_t;
		using Scalar_t 		= typename Assembly_t::Scalar_t;
		using Point_t       = typename Assembly_t::Point_t;
		using Box_t		  	= typename Assembly_t::Box_t;


		using Mesh_t		= UnstructuredVoxelMesh<Scalar_t>;
		static_assert(std::same_as<Box_t, typename Mesh_t::Box_t>);
		static_assert(std::same_as<Point_t, typename Mesh_t::GeoPoint_t>);


		static constexpr uint8_t PERIOD = Assembly_t::PERIOD;


		///////////////////////////////////////////////////////////////
		/// Own the assembly and the mesh
		///////////////////////////////////////////////////////////////
		Assembly_t 	assembly;
		Mesh_t		mesh;


		///////////////////////////////////////////////////////////////
		/// Construcors
		///////////////////////////////////////////////////////////////
		DiffuseDomainMeshHandler() = delete;
		DiffuseDomainMeshHandler(const Box_t& domain, uint8_t max_depth) : 
				assembly(domain), mesh(domain, max_depth) {}
		DiffuseDomainMeshHandler(DiffuseDomainMeshHandler&&) = default;
		DiffuseDomainMeshHandler(const DiffuseDomainMeshHandler&) = delete;
		DiffuseDomainMeshHandler& operator=(DiffuseDomainMeshHandler&&) = delete;
		DiffuseDomainMeshHandler& operator=(const DiffuseDomainMeshHandler&) = delete;


		///////////////////////////////////////////////////////////////
		/// Read the assembly from a file
		///////////////////////////////////////////////////////////////
		void build_assembly(const std::string& filename) noexcept {
			std::vector<Particle_t> list;

			if constexpr (std::same_as<Particle_t, gutil::Sphere<3,Scalar_t>>) {
				list = gutil::read_spheres_from_file<3,Scalar_t>(filename);
			}
			else {
				GUTIL_ABORT("Unsupported particle type");
			}

			if (list.empty()) {GUTIL_ERROR("No particles found");}
			assembly.push_back_range(std::move(list));

			GUTIL_LOG("Assembly with period ", PERIOD, " tracking ", assembly.size(),  " particles");
		}


		///////////////////////////////////////////////////////////////
		/// Get the finest element size in the mesh to use with the heaviside/dirac function
		///////////////////////////////////////////////////////////////
		[[nodiscard]] Scalar_t min_element_size() const noexcept {
			GUTIL_ASSERT(mesh.is_depth_sorted());
			for (int dd=(int)mesh.max_depth; dd>=0; --dd) {
				if (mesh.get_depth(dd).size()>0) {
					Scalar_t scale = gutil::ldexp(Scalar_t{1},-dd);
					return scale * gutil::norm2(mesh.diag);
				}
			}

			GUTIL_ERROR("No elements found");
			return -1;
		}
	};
}



