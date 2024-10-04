INC_DIR    := include
BUILD_DIR  := lib
SRC_DIR    := src

#util module
UTIL_SOURCES := $(wildcard src/util/*.cpp)
UTIL_OBJECTS := $(UTIL_SOURCES:.cpp=.o)

#geometry module
GEO_SOURCES := $(wildcard src/geometry/*.cpp)
GEO_OBJECTS := $(GEO_SOURCES:.cpp=.o)

#mesh module
MESH_SOURCES := $(wildcard src/mesh/*.cpp)
MESH_OBJECTS := $(MESH_SOURCES:.cpp=.o)

#solver module
SOLVER_SOURCES := $(wildcard src/solver/*.cpp)
SOLVER_OBJECTS := $(SOLVER_SOURCES:.cpp=.o)

#MAC module
MAC_SOURCES := $(wildcard src/mac/*.cpp)
MAC_OBJECTS := $(MAC_SOURCES:.cpp=.o)

CC        := g++
CFLAGS    := -O2 -Wall

TARGET    := $(BUILD_DIR)/libcgeovox.a





##############################################
######### BUILD LIBRARY ######################
##############################################
$(TARGET): $(UTIL_OBJECTS) $(GEO_OBJECTS) $(MESH_OBJECTS) $(SOLVER_OBJECTS) $(MAC_OBJECTS)| $(BUILD_DIR)
	ar rcs $@ $^
	ranlib $@





##############################################
######### BUILD OBJECTS ######################
##############################################
.cpp.o:
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@




##############################################
######### DIRECTORY AND CLEANUP ##############
##############################################
$(BUILD_DIR):
	mkdir -p $@

# clean directories
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	find . -name "*.o" -type f -delete

.PHONY: print
print:
	@echo UTIL_SOURCES:= $(UTIL_SOURCES)
	@echo UTIL_OBJECTS:= $(UTIL_OBJECTS)
	@echo GEO_SOURCES:= $(GEO_SOURCES)
	@echo GEO_OBJECTS:= $(GEO_OBJECTS)
	@echo MESH_SOURCES:= $(MESH_SOURCES)
	@echo MESH_OBJECTS:= $(MESH_OBJECTS)