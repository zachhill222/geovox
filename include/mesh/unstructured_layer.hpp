#pragma once



#include "gutil.hpp"

#include "mesh/keys/voxel_key.hpp"
#include "mesh/vtk_file_io.hpp"

#include <algorithm>
#include <vector>




namespace GV {
	/////////////////////////////////////////////////////////////////////////////////////////////////////
	/// A helper class to better loop through meshes.
	/// This is designed to be used only at a single mesh depth.
	/// The expected interface to be able to use the vtk file io is also supplied.
	/////////////////////////////////////////////////////////////////////////////////////////////////////	
	template<uint64_t MaxDepth>
	struct UnstructuredLayer {


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Aliases and storage
		/////////////////////////////////////////////////////////////////////////////////////////////////
		using VoxelElement = VoxelElementKey<MaxDepth,0>;
		using VoxelVertex  = VoxelVertexKey<MaxDepth,0>;
		using VoxelFace    = VoxelFaceKey<MaxDepth,0>;
		using GeoPoint_t   = gutil::Point<3,double>;
		using Box_t        = gutil::Box<3,double>;
		using Mesh_t       = UnstructuredLayer<MaxDepth>;

		Box_t box;
		std::vector<VoxelElement> elements;
		std::vector<VoxelVertex> vertices;
		std::vector<uint64_t> color_offsets;


		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Helpful interface
		/////////////////////////////////////////////////////////////////////////////////////////////////
		void color() noexcept {
			//at a single depth, we need 8 colors.
			gutil::BinSort<VoxelElement> sorter(std::span<VoxelElement>{elements.begin(), elements.end()}, 8);
			sorter.sort( [](VoxelElement el) {
				int clr=0;
				if (el.i() & 1) {clr += 1;}
				if (el.j() & 1) {clr += 2;}
				if (el.k() & 1) {clr += 4;}
				return clr;}  );

			color_offsets.clear();
			for (int i=0; i<8; ++i) {
				color_offsets.push_back(sorter.bin_start(i));
			}
			color_offsets.push_back(sorter.bin_end(8));
		}

		void collect_vertices() noexcept {
			vertices.clear();
			for (VoxelElement el : elements) {
				for (VoxelVertex vtx : el.vertices()) {
					vertices.push_back(vtx.reduced_key());
				}
			}
			std::sort(vertices.begin(), vertices.end());
			auto last = std::unique(vertices.begin(), vertices.end());
			vertices.erase(last, vertices.end());
			vertices.shrink_to_fit();
		}

		[[nodiscard]] uint64_t vertex_index(VoxelVertex vtx) const noexcept {
			auto it = std::lower_bound(vertices.begin(), vertices.end(), vtx);
			if (it == vertices.end() || *it!=vtx) {return uint64_t(-1);}
			return std::distance(vertices.begin(), it);
		}

		[[nodiscard]] GeoPoint_t geo_coord(VoxelVertex vtx) const noexcept {
			{return box.low + (box.high-box.low)*vtx.normalized_coordinate();}
		}

		auto element_begin() const { return elements.cbegin(); }
		auto element_end()   const { return elements.cend();   }
		auto vertex_begin()  const { return vertices.cbegin(); }
		auto vertex_end()    const { return vertices.cend();   }

		[[nodiscard]] uint64_t n_elements() const noexcept { return elements.size(); }
		[[nodiscard]] uint64_t n_vertices() const noexcept { return vertices.size(); }

		void push_back(VoxelElement el) noexcept { elements.push_back(el); }
		void clear() noexcept {
			elements.clear();
			vertices.clear();
			color_offsets.clear();
		}



		/////////////////////////////////////////////////////////////////////////////////////////////////
		/// Methods to write to a VTK file
		/////////////////////////////////////////////////////////////////////////////////////////////////
		void save_as_ascii(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,true>(filename, *this, description);}
		void save_as_binary(const std::string& filename, const std::string& description = "") const {print_topology_vtk<Mesh_t,false>(filename, *this, description);}

		template<typename... Lookup_ts>
		inline void append_cell_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_cell_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		inline void append_cell_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_cell_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		inline void append_point_data_field_ascii(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_point_data_field_vtk<Mesh_t,true>(filename, *this, field_name, lookups...);
		}
		template<typename... Lookup_ts>
		inline void append_point_data_field_binary(const std::string& filename, const std::string field_name, const Lookup_ts&... lookups) const {
			append_point_data_field_vtk<Mesh_t,false>(filename, *this, field_name, lookups...);
		}
	};
}