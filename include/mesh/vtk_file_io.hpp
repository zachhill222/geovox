#pragma once

#include "gutil.hpp"
#include "util/big_endian_helpers.hpp"

#include <cstring>
#include <cassert>
#include <string>
#include <fstream>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>

namespace GV
{
	template<typename OS, typename Mesh_t, bool ASCII>
	void write_point_field(OS& buffer, const Mesh_t& mesh) {
		if (mesh.vertex_begin()==mesh.vertex_end()) {
			gutil::Logger::error("ERROR: no vertices found. Maybe mesh.collect_vertices() was forgotten?");
			throw;
		}

		using PrintPoint_t = gutil::Point<3,float>;

		buffer << "POINTS " << mesh.n_vertices() << " float\n";
		for (auto it=mesh.vertex_begin(); it!=mesh.vertex_end(); ++it) {
			const PrintPoint_t coord = static_cast<PrintPoint_t>(mesh.geo_coord(*it));
			
			if constexpr (ASCII) {
				buffer << static_cast<PrintPoint_t>(coord) << "\n";
			}
			else {
				WRITE_BIG_ENDIAN(buffer, coord);
			}
		}
		buffer << "\n";
	}

	template<typename OS, typename Mesh_t, bool ASCII>
	void write_cell_field(OS& buffer, const Mesh_t& mesh) {
		if (mesh.element_begin()==mesh.element_end()) {
			gutil::Logger::error("ERROR: no elements found. Maybe mesh.update_unstructured() was forgotten?");
			throw;
		}

		buffer << "CELLS " << mesh.n_elements() << " " << 9*mesh.n_elements() << "\n";
		for (auto it=mesh.element_begin(); it!=mesh.element_end(); ++it) {
			std::array<uint64_t,8> v_idx;
			for (int i=0; i<8; ++i) {
				v_idx[i] = mesh.vertex_index(it->vertex(i).reduced_key());
			}

			if constexpr (ASCII) {
				buffer << 8;
				for (auto idx : v_idx) {buffer << " " << idx;}
				buffer << "\n";
			}
			else {
				WRITE_BIG_ENDIAN<int32_t>(buffer,8);
				WRITE_BIG_ENDIAN<int32_t>(buffer,v_idx);
			}
		}
		
		buffer << "\nCELL_TYPES " << mesh.n_elements() << "\n";
		for (uint32_t idx=0; idx<mesh.n_elements(); ++idx) {
			if constexpr (ASCII) {
				buffer << "11 ";
			}
			else {
				WRITE_BIG_ENDIAN<int32_t>(buffer,11);
			}
		}

		buffer << "\n";
	}



	template<typename Mesh_t, bool ASCII>
	void print_topology_vtk(const std::string& filename, const Mesh_t& mesh, const std::string description="Mesh Data") {
		const auto mode = ASCII ? std::ios::out : (std::ios::out | std::ios::binary);
		std::ofstream file(filename, mode);

		if (!file.is_open()) {
			throw std::runtime_error("print_topology_vtk - Could not open file: " + filename);
		}

		//HEADER
		file << "# vtk DataFile Version 2.0\n";
		file << description + "\n";
		if constexpr (ASCII) {file << "ASCII\n\n";}
		else {file << "BINARY\n\n";}
		file << "DATASET UNSTRUCTURED_GRID\n";

		//TOPOLOGY
		if constexpr (ASCII) {
			std::stringstream p_buff, c_buff;

			std::thread p_thread([&](){write_point_field<std::stringstream,Mesh_t,ASCII>(p_buff,mesh);});
			std::thread c_thread([&](){write_cell_field<std::stringstream,Mesh_t,ASCII>(c_buff,mesh);});

			p_thread.join();
			file << p_buff.rdbuf();

			c_thread.join();
			file << c_buff.rdbuf();
		}
		else {
			std::ostringstream p_buff(std::ios::out | std::ios::binary);
			std::ostringstream c_buff(std::ios::out | std::ios::binary);

			std::thread p_thread([&](){write_point_field<std::ostringstream,Mesh_t,ASCII>(p_buff,mesh);});
			std::thread c_thread([&](){write_cell_field<std::ostringstream,Mesh_t,ASCII>(c_buff,mesh);});

			p_thread.join();
			file.write(p_buff.str().data(), p_buff.str().size());

			c_thread.join();
			file.write(c_buff.str().data(), c_buff.str().size());
		}
	}


	template<typename Mesh_t, bool ASCII, typename... Lookups> 
	void append_cell_data_field_vtk(const std::string& filename, const Mesh_t& mesh, const std::string field_name, const Lookups&... lookups) {
		//sanity check
		static_assert(sizeof...(lookups)>0, "no data lookup provided");
		static_assert( ((std::same_as<typename Mesh_t::Elem_t, typename Lookups::MeshFeature_t> || std::same_as<void, typename Lookups::MeshFeature_t>)&& ... ), "all lookups must be for mesh elements");

		//open file
		constexpr auto mode = ASCII ? std::ios::app : (std::ios::app | std::ios::binary);
		constexpr size_t N_LOOKUPS = sizeof...(lookups);
		std::ofstream file(filename, mode);

		if (!file.is_open()) {
			throw std::runtime_error("append_cell_data_field_vtk - Could not open file: " + filename);
		}
		
		//write section header and get feature count
		const uint64_t f_count = mesh.n_elements();
		file << "CELL_DATA " << f_count << "\n";
		file << "FIELD " << field_name << " " << sizeof...(lookups) << "\n";

		//write a lambda to do the printing for each lookup
		auto write_var = [&]<typename Lookup_t>(std::ostringstream& buf, const Lookup_t& lookup) {
			buf << lookup.header(f_count);

			uint64_t idx = 0;
			for (auto it = mesh.element_begin(); it != mesh.element_end(); ++it, ++idx) {
				const auto val = [&]() {
					if constexpr (std::same_as<typename Lookup_t::MeshFeature_t, void>) {return lookup(idx);}
					else {return lookup(*it);}
				}();
				if constexpr (ASCII) {
					if constexpr (Lookup_t::IS_SCALAR) {buf << val << "\n";}
					else {
						for (int i=0; i<Lookup_t::N; i++) {buf << val[i] << " ";}
						buf << "\n";
					}
				}
				else {
					WRITE_BIG_ENDIAN<typename Lookup_t::OutScalar_t>(buf, val);
				}
			}

			if constexpr (ASCII) {buf << "\n";}
		};

		//set up one buffer per lookup, then dispatch one thread per lookup and combine
		std::array<std::ostringstream, N_LOOKUPS> buffers;
		if constexpr (!ASCII) {
			for (auto& b : buffers) { b = std::ostringstream(std::ios::out | std::ios::binary); }
		}

		auto lookup_refs = std::tie(lookups...);

		[&]<std::size_t... I>(std::index_sequence<I...>) {
			std::array<std::thread, N_LOOKUPS> threads {
				std::thread( [&](){ write_var(buffers[I], std::get<I>(lookup_refs)); } )...
			};
			for (auto& t : threads) {t.join();}
		}(std::index_sequence_for<Lookups...>{});

		for (auto& buf : buffers) {
			const std::string s = buf.str();
			if constexpr (ASCII) {file << s;}
			else {file.write(s.data(),s.size());}
		}
	}

	template<typename Mesh_t, bool ASCII, typename... Lookups> 
	void append_point_data_field_vtk(const std::string& filename, const Mesh_t& mesh, const std::string field_name, const Lookups&... lookups) {
		//sanity check
		static_assert(sizeof...(lookups)>0, "no data lookup provided");
		static_assert( ((std::same_as<typename Mesh_t::Vert_t, typename Lookups::MeshFeature_t> || std::same_as<void, typename Lookups::MeshFeature_t>)&& ... ), "all lookups must be for mesh vertices");

		//open file
		constexpr auto mode = ASCII ? std::ios::app : (std::ios::app | std::ios::binary);
		constexpr size_t N_LOOKUPS = sizeof...(lookups);
		std::ofstream file(filename, mode);

		if (!file.is_open()) {
			throw std::runtime_error("append_point_data_field_vtk - Could not open file: " + filename);
		}

		//write section header and get feature count
		const uint64_t f_count = mesh.n_vertices();
		file << "POINT_DATA " << f_count << "\n";
		file << "FIELD " << field_name << " " << N_LOOKUPS << "\n";

		//write a lambda to do the printing for each lookup
		auto write_var = [&]<typename Lookup_t>(std::ostringstream& buf, const Lookup_t& lookup) {
			buf << lookup.header(f_count);

			uint64_t idx = 0;
			for (auto it = mesh.vertex_begin(); it != mesh.vertex_end(); ++it, ++idx) {
				const auto val = [&]() {
					if constexpr (std::same_as<typename Lookup_t::MeshFeature_t, void>) {return lookup(idx);}
					else {return lookup(*it);}
				}();
				if constexpr (ASCII) {
					if constexpr (Lookup_t::IS_SCALAR) {buf << val << "\n";}
					else {
						for (int i=0; i<Lookup_t::N; i++) {buf << val[i] << " ";}
						buf << "\n";
					}
				}
				else {
					WRITE_BIG_ENDIAN<typename Lookup_t::OutScalar_t>(buf, val);
				}
			}

			if constexpr (ASCII) {buf << "\n";}
		};

		//set up one buffer per lookup, then dispatch one thread per lookup and combine
		std::array<std::ostringstream, N_LOOKUPS> buffers;
		if constexpr (!ASCII) {
			for (auto& b : buffers) { b = std::ostringstream(std::ios::out | std::ios::binary); }
		}

		auto lookup_refs = std::tie(lookups...);

		[&]<std::size_t... I>(std::index_sequence<I...>) {
			std::array<std::thread, N_LOOKUPS> threads {
				std::thread( [&](){ write_var(buffers[I], std::get<I>(lookup_refs)); } )...
			};
			for (auto& t : threads) {t.join();}
		}(std::index_sequence_for<Lookups...>{});

		for (auto& buf : buffers) {
			const std::string s = buf.str();
			if constexpr (ASCII) {file << s;}
			else {file.write(s.data(),s.size());}
		}
	}



	template<typename Feature_t, typename Lambda_t> 
	using LambdaReturn_t = std::invoke_result_t<Lambda_t, Feature_t>;

	template<typename T>
	struct scalar_type {using type=T;};

	template<typename T> requires requires{typename T::value_type;}
	struct scalar_type<T> {using type = typename T::value_type;};

	template<typename T>
	using scalar_type_t = typename scalar_type<T>::type;

	template<typename Feature_t, typename Lambda_t>
	struct FeatureDataLookup {
		Lambda_t lookup;
		std::string varname;
		using MeshFeature_t = Feature_t;
		using Return_t = LambdaReturn_t<Feature_t, Lambda_t>;
		static constexpr bool IS_SCALAR = std::is_arithmetic_v<Return_t>;
		
		static constexpr int N = []{
			if constexpr (IS_SCALAR) {return 1;}
			else {return std::tuple_size_v<Return_t>;}
		}();

		using Scalar_t = decltype([] {
			if constexpr (IS_SCALAR) {return Return_t{};}
			else return typename Return_t::value_type{};
		}());

		using OutScalar_t = std::conditional_t<std::is_floating_point_v<Scalar_t>, float,
								std::conditional_t<std::is_integral_v<Scalar_t>, int32_t, void>>;
		static_assert(!std::is_same_v<OutScalar_t,void>, "unknown return type");

		FeatureDataLookup(Lambda_t&& lookup_, std::string varname_ = "var") 
			: lookup(std::forward<Lambda_t>(lookup_)), varname(varname_) {}

		std::string header(const uint64_t n_data) const {
			std::string hdr = varname + " " + std::to_string(N) + " " + std::to_string(n_data);
			if constexpr (std::same_as<OutScalar_t, float>) {return hdr + " float\n";}
			else if constexpr (std::same_as<OutScalar_t, int32_t>) {return hdr + " integer\n";}
			else {assert(false && "unknown scalar");}
		}

		Return_t operator()(Feature_t feature) const {return lookup(feature);}
		template<typename OtherFeature_t>
		Return_t operator()(OtherFeature_t feature) const {return lookup(static_cast<Feature_t>(feature));}
	};

	//factory to help make lookups a bit easier
	template<typename Feature_t, typename Lambda_t>
	auto make_feature_lookup(Lambda_t&& lambda, std::string varname="var") {
		return FeatureDataLookup<Feature_t, std::decay_t<Lambda_t>>{std::forward<Lambda_t>(lambda), varname};
	}


	//struct to lookup data by the feature index
	template<typename Return_t_in, typename Lambda_t>
	struct IndexDataLookup {
		Lambda_t lookup;
		std::string varname;
		using MeshFeature_t = void;
		using Return_t = Return_t_in;
		static constexpr bool IS_SCALAR = std::is_arithmetic_v<Return_t>;
		
		static constexpr int N = []{
			if constexpr (IS_SCALAR) {return 1;}
			else {return std::tuple_size_v<Return_t>;}
		}();

		using Scalar_t = decltype([] {
			if constexpr (IS_SCALAR) {return Return_t{};}
			else return typename Return_t::value_type{};
		}());

		using OutScalar_t = std::conditional_t<std::is_floating_point_v<Scalar_t>, float,
								std::conditional_t<std::is_integral_v<Scalar_t>, int32_t, void>>;
		static_assert(!std::is_same_v<OutScalar_t,void>, "unknown return type");

		IndexDataLookup(Lambda_t&& lookup_, std::string varname_ = "var") 
			: lookup(std::forward<Lambda_t>(lookup_)), varname(varname_) {}

		std::string header(const uint64_t n_data) const {
			std::string hdr = varname + " " + std::to_string(N) + " " + std::to_string(n_data);
			if constexpr (std::same_as<OutScalar_t, float>) {return hdr + " float\n";}
			else if constexpr (std::same_as<OutScalar_t, int32_t>) {return hdr + " integer\n";}
			else {assert(false && "unknown scalar");}
		}

		Return_t operator()(uint64_t idx) const {return lookup(idx);}
	};

	//factory to help make lookups a bit easier
	template<typename Return_t_in, typename Lambda_t>
	auto make_index_lookup(Lambda_t&& lambda, std::string varname="var") {
		return IndexDataLookup<Return_t_in, std::decay_t<Lambda_t>>{std::forward<Lambda_t>(lambda), varname};
	}
}