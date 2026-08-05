#get source/target names of all files in the specified directory
ifndef DIR
	DIR = mesh
endif
SRC     = $(wildcard $(DIR)/*.cpp)
TARGETS = $(SRC:.cpp=)

#header files
INCLUDE = -I../include -I/usr/include/eigen3 -I../external/gutil/include

#default build
CXX     = g++ -std=c++23 -Wall -march=native
# CXX    += -fsanitize=address
CXX    += -fsanitize=thread

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
ifdef GV_TEST_DOMAIN_PERIOD
	DEFINES += -DGV_DOMAIN_PERIOD=$(GV_DOMAIN_PERIOD)
endif
ifdef GV_TEST_DOF_PERIOD
	DEFINES += -DGV_TEST_DOF_PERIOD=$(GV_TEST_DOF_PERIOD)
endif
ifdef GV_TEST_MAX_DEPTH
	DEFINES += -DGV_TEST_MAX_DEPTH=$(GV_TEST_MAX_DEPTH)
endif
ifdef GV_TEST_DOMAIN_SIZE
	DEFINES += -DGV_TEST_DOMAIN_SIZE=$(GV_TEST_DOMAIN_SIZE)
endif

#optimization level
ifdef OPTIMIZE
	CXX += -O${OPTIMIZE}
else ifdef NDEBUG
	CXX += -O3
else
	CXX += -O0
endif
#debug vs release mode
ifdef NDEBUG
	ifneq ($(NDEBUG),0)
	DEFINES += -DNDEBUG
	endif
else
	CXX += -pg -g
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