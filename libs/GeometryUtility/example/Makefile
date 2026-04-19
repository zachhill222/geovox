INCLUDE = -I../include

#default build (debug)
CXX        = g++ -std=c++20
OPT        = -O0
PAR        =
DEBUG      = -pg -g -Wall

#redefine for testing optimizations
OPT        = -O3
PAR        = -fopenmp
DEBUG      = -DNDEBUG

#final compiler command
CC = $(CXX) $(OPT) $(PAR) $(DEBUG)

#testing defines
DIM        = -DDIMENSION=2

#final defines
DEFINES = ${DIM}

point:
	${CC} ${INCLUDE} ${DEFINES} point_octree_example.cpp -o point_octree_example

polygon:
	${CC} ${INCLUDE} ${DEFINES} polygon_octree_example.cpp -o polygon_octree_example


clean:
	rm -f point_octree_example polygon_octree_example *.out