#include "gutil.hpp"

#include <iostream>

#ifdef _OPENMP
	#include <omp.h>
#endif

#ifndef DIMENSION
	#define DIMENSION 2
#endif
static_assert(DIMENSION==2 or DIMENSION==3, "test is enabled for DIMENSION 2 or 3");

using Scalar_t = gutil::FixedPoint<int32_t>;
// using Scalar_t = float;
using Octree_t = gutil::PointOctree<DIMENSION,Scalar_t,128>;
using Index_t  = gutil::Point<DIMENSION,size_t>;
using Point_t  = gutil::Point<DIMENSION,Scalar_t>;
using Box_t    = gutil::Box<DIMENSION,Scalar_t>;

void generate_points2(Octree_t& octree, const Index_t& N) {
	assert(N>Index_t{0});

	#ifdef _OPENMP
		//if openmp is enabled and data is being inserted in parallel, then
		//the space must be allocated ahead of time.
		octree.resize(prod(N));
		#pragma omp parallel for collapse(2)
	#else
		//if openmp is not enabled, reserving space is better
		octree.reserve(prod(N));
	#endif

	for (size_t i=0; i<N[0]; i++) {
		for (size_t j=0; j<N[1]; j++) {
			//push_back_async can be used even if openmp is not enabled
			//when openmp is enabled, it is the user's responsibility to ensure
			//that no duplicate data is inserted. If openmp is not enambled, then
			//push_back_async defaults to push_back. This can still be called in parallel, but
			//it locks the entire tree with a lock_guard mutex. Data can be searched in parallel
			//if no data is currently being inserted.
			#ifdef _OPENMP
				octree.push_back_async(Point_t{i,j});
			#else
				octree.push_back(Point_t{i,j});
			#endif
		}
	}

	//if data is inserted in parallel, the buffer must be flushed to ensure
	//that it is up to date. If openmp is not enabled or octree.push_back() is called, then
	//this does not need to be called.
	octree.flush();

	//this isn't needed with precise reserve/resize usage
	//when pushing data back in parallel, it may be necessary (or more efficient) in practice to overestimate
	//the number of data entries to be added when calling resize() and then free unused space after.
	octree.shrink_to_fit();
}

void generate_points3(Octree_t& octree, const Index_t& N) {
	assert(N>Index_t{0});

	#ifdef _OPENMP
		octree.resize(prod(N));
		#pragma omp parallel for collapse(3)
	#else
		octree.reserve(prod(N));
	#endif
	
	for (size_t i=0; i<N[0]; i++) {
		for (size_t j=0; j<N[1]; j++) {
			for (size_t k=0; k<N[2]; k++) {
				#ifdef _OPENMP
					octree.push_back_async(Point_t{i,j,k});
				#else
					octree.push_back(Point_t{i,j,k});
				#endif
			}
		}
	}

	octree.flush();
	octree.shrink_to_fit();
}

void check_octree_find(const Octree_t& octree) {
	size_t n_miss = 0;

	#ifdef _OPENMP
		#pragma omp parallel for reduction(+:n_miss)
	#endif
	for (size_t i=0; i<octree.size(); i++) {
		const size_t j = octree.find(octree[i]);
		if (i!=j) {n_miss++;}
	}

	std::cout << "The octree had " << n_miss << " find() misses ";
	if (n_miss==0) {std::cout << "SUCCESS\n";}
	else {std::cout << "FAIL\n";}
}

void check_octree_move_data(Octree_t& octree) {
	Box_t region_to_move = 0.5*octree.bbox(); //center of bounding box
	std::vector<size_t> indices_to_move = octree.get_data_in_box(region_to_move);

	std::cout << "The octree found " << indices_to_move.size() << " points in the box: " << region_to_move << std::endl;

	#ifdef _OPENMP
		#pragma omp parallel for
	#endif
	for (size_t i=0; i<indices_to_move.size(); i++) {
		const size_t idx = indices_to_move[i];
		Point_t new_data = -octree[idx];
		octree.replace(std::move(new_data), idx);
	}

	//ensure that the tree is still valid after changing data
	//most likely unnecessary data located at a single point.
	octree.rebuild_tree();

	indices_to_move = octree.get_data_in_box(region_to_move);
	std::cout << "After moving data, the octree found " << indices_to_move.size() << " points in the box: " << region_to_move << std::endl;
}

void print_octree_summary(const Octree_t& octree) {
	//print basic stats from the octree
	auto stats = octree.get_tree_stats();
	std::cout << "\nThe octree has " << stats.n_nodes << " octree nodes with " << stats.n_leafs << " leafs\n";
	std::cout << "The octree is storing " << stats.n_used_indices << "/" << stats.n_indices_capacity << " data indices\n";
	std::cout << "The octree has a maximum depth of " << stats.max_depth << std::endl;

	double storage_memory_in_bytes  = octree.size() * sizeof(typename Octree_t::Data_t);
	double byte2MiB = std::pow(0.5, 20);

	std::cout << "The octree structure itself is using " << stats.memory_used_bytes*byte2MiB << "/" << stats.memory_reserved_bytes*byte2MiB << " MiB\n";
	std::cout << "The data itself is using " << storage_memory_in_bytes*byte2MiB << " MiB\n";

	std::cout << "The octree bounding box is " << octree.bbox() << std::endl;
}


int main(int argc, char* argv[]) {
	#if DIMENSION==2
		Index_t N{100,100};
	#elif DIMENSION==3
		Index_t N{100,100,100};
	#endif

	if (argc>1) {N[0]=std::stoull(argv[1]);}
	if (argc>2) {N[1]=std::stoull(argv[2]);}
	if (argc>3 and DIMENSION==3) {N[2]=std::stoull(argv[3]);}

	//the bounding box of the octree will automatically be expanded when needed.
	//this is fairly fast (adds nodes to the top of the tree structure rather than rebuilding)
	//but if the bounding box is known ahead of time, it should be used.
	//to help avoid floating point errors, the bounding box will be expanded to have
	//power of 2 coordinates.
	Octree_t octree(Box_t{Point_t{0}, Point_t{N}});
	// Octree_t octree;

	//generate points and verify the correct number were generated
	if constexpr (DIMENSION==2) {
		generate_points2(octree, N);
	} else if constexpr (DIMENSION==3) {
		generate_points3(octree, N);
	}
	
	std::cout << "The octree has " << octree.size() << " elements ";
	if (octree.size() == prod(N)) {std::cout << "SUCCESS\n";}
	else {std::cout << "FAIL\n";}

	//verify that i==octre.find(octree[i]) for all i
	check_octree_find(octree);

	//print octree summary
	print_octree_summary(octree);

	//test moving data
	check_octree_move_data(octree);
	
	//print octree summary
	print_octree_summary(octree);

	return 0;
}


