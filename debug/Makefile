#get source/target names of all files in the specified directory
ifndef DIR
	DIR = mesh
endif
SRC     = $(wildcard $(DIR)/*.cpp)
TARGETS = $(SRC:.cpp=)

#header files
INCLUDE = -I../include -I/usr/include/eigen3 -I../external/gutil/include

#default build (debug)
CXX     = g++ -std=c++23

#explicit defines
DEFINES = 
ifdef PROFILE
	DEFINES += -DPROFILE=$(PROFILE)
endif
ifdef GUTIL_ORTHTREE_MAX_DEPTH
	DEFINES += -DGUTIL_ORTHTREE_MAX_DEPTH=$(GUTIL_ORTHTREE_MAX_DEPTH)
endif
ifdef GUTIL_N_OCTREE_THREADS
	DEFINES += -DGUTIL_N_OCTREE_THREADS=$(GUTIL_N_OCTREE_THREADS)
endif

#debug vs release mode
ifdef NDEBUG
	CXX += -O3
	DEFINES += -DNDEBUG
else
	CXX += -O0 -Wall -pg
endif

#link to libraries
ifndef NO_OMP
	LIBS = -fopenmp
else
	LIBS = 
endif

#final compile command
COMPILE = ${CXX} ${INCLUDE} ${LIBS} ${DEFINES}

all: $(TARGETS)

$(DIR)/%: $(DIR)/%.cpp FORCE
	$(COMPILE) $< -o $@

FORCE:

.PHONY: clean
clean:
	rm -f  *.out *.vtk massif.* ./outfiles/*

.PHONY: print
print:
	@echo CXX:=     $(CXX)
	@echo DIR:=     $(DIR)
	@echo INCLUDE:= $(INCLUDE)